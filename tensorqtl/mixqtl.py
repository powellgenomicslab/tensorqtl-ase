import torch
import numpy as np
import pandas as pd
import scipy.stats
import os
import sys
import time
sys.path.insert(1, os.path.dirname(__file__))
import cis
from core import *


def trc(genotypes_t, counts_t, lib_size_t=None, covariates_t=None, select_covariates=True,
        count_threshold=20, return_af=False):
    """
    Inputs
      genotypes_t: dosages (variants x samples)
      counts_t: total read counts
      lib_size_t: library size
      covariates_t: covariates matrix, first column MUST be intercept
    """
    # Cast to float64 for numerical precision (CUDA and CPU both support it)
    genotypes_t = genotypes_t.to(torch.float64)
    counts_t = counts_t.to(torch.float64)
    if lib_size_t is not None:
        lib_size_t = lib_size_t.to(torch.float64)
    else:
        lib_size_t = torch.ones_like(counts_t)
    if covariates_t is not None:
        covariates_t = covariates_t.to(torch.float64)

    # R: lhs = log(trc / lib_size / 2)
    y_full_t = torch.log(counts_t / (lib_size_t * 2.0))
    device = y_full_t.device
    dtype = y_full_t.dtype

    if covariates_t is not None:
        # R's regress_against_covariate uses all samples that are not NA/inf
        m_cov_t = ~torch.isinf(y_full_t) & ~torch.isnan(y_full_t)
        y_cov_t = y_full_t[m_cov_t]
        c_cov_t = covariates_t[m_cov_t, :]

        if select_covariates:
            # Step 1: select significant covariates
            b_t, b_se_t = linreg(c_cov_t, y_cov_t, dtype=dtype)
            tstat_t = b_t / b_se_t
            # R: selected = abs(out[-1, 3]) > 2 (skipping intercept)
            selected = tstat_t[1:].abs() > 2

            if selected.any():
                # Step 2: calculate predicted response (excluding intercept)
                sel_mask = torch.cat([torch.tensor([True]).to(device), selected])
                sel_covariates_cov_t = c_cov_t[:, sel_mask]
                b_sel_t, _ = linreg(sel_covariates_cov_t, y_cov_t, dtype=dtype)

                # Calculate offset for ALL samples
                offset_full_t = torch.matmul(covariates_t[:, sel_mask][:, 1:], b_sel_t[1:])
            else:
                offset_full_t = torch.zeros_like(y_full_t)



        else:
            b_full_t, _ = linreg(c_cov_t, y_cov_t, dtype=dtype)
            offset_full_t = torch.matmul(covariates_t[:, 1:], b_full_t[1:])
    else:
        offset_full_t = torch.zeros_like(y_full_t)

    # Now filter samples for trcQTL
    y_target_full_t = y_full_t - offset_full_t
    m_samples = (counts_t >= count_threshold) & ~torch.isinf(y_full_t) & ~torch.isnan(y_full_t)
    n_f = m_samples.sum().item()

    if n_f <= 2:
        num_variants = genotypes_t.shape[0]
        nan_t = torch.full([num_variants], np.nan, device=device, dtype=dtype)
        res = (nan_t, nan_t.clone(), nan_t.clone())
        flag = f'low_counts:n={n_f},threshold={count_threshold}'
        if return_af:
            af, ma_samples, ma_counts = get_allele_stats(genotypes_t)
            return *res, n_f, flag, af, ma_samples, ma_counts
        else:
            return *res, n_f, flag

    # R's mixqtl (mixqtl/R/mixqtl.R) does h1[is.na(h1)] = 0.5; h2[is.na(h2)] = 0.5 before summing.
    # We expect genotypes_t to be already imputed if coming from mixqtl().
    # But if called directly, we ensure no NaNs.
    X_f = genotypes_t[:, m_samples].clone()
    X_f[torch.isnan(X_f)] = 1.0 # (0.5 + 0.5)
    X_f = X_f / 2
    
    y_f = y_target_full_t[m_samples]
    
    # mask for monomorphic variants
    is_mono = (X_f.max(1)[0] == X_f.min(1)[0])
    
    valid_mask = ~is_mono
    
    num_variants = X_f.shape[0]
    b1 = torch.full([num_variants], np.nan, device=device, dtype=dtype)
    se1 = torch.full([num_variants], np.nan, device=device, dtype=dtype)
    tstat = torch.full([num_variants], np.nan, device=device, dtype=dtype)
    sample_sizes = torch.zeros([num_variants], dtype=torch.long, device=device)

    if valid_mask.any():
        X_v = X_f[valid_mask]
        
        T1 = torch.matmul(X_v, y_f) # (v_valid,)
        T2 = y_f.sum() # scalar
        S11 = (X_v**2).sum(1) # (v_valid,)
        S12 = X_v.sum(1) # (v_valid,)
        S22 = float(n_f)
        
        delta = S11 * S22 - S12**2
        b1_v = (S22 * T1 - S12 * T2) / delta
        b2_v = (S11 * T2 - S12 * T1) / delta # intercept
        
        y2_sum = (y_f**2).sum()
        rsq = y2_sum - 2*b1_v*T1 - 2*b2_v*T2 + 2*b1_v*b2_v*S12 + b1_v**2*S11 + b2_v**2*S22
        
        sigma = torch.sqrt(torch.clamp(rsq, min=0) / (n_f - 2))
        se1_v = sigma * torch.sqrt(S22 / delta)
        
        b1[valid_mask] = b1_v
        se1[valid_mask] = se1_v
        tstat[valid_mask] = b1_v / se1_v
        sample_sizes[valid_mask] = n_f

    res = (tstat, b1, se1)
    sample_size = n_f
    flag = 'ok'

    if return_af:
        af, ma_samples, ma_counts = get_allele_stats(genotypes_t)
        return *res, sample_size, flag, af, ma_samples, ma_counts
    else:
        return *res, sample_size, flag


def asc(genotypes1_t, genotypes2_t, counts1_t, counts2_t,
        asc_cutoff=5, weight_cap=100, asc_cap=5000):
    """
    Allele-specific QTL mapping (ascQTL).
    Solves log(counts1/counts2) ~ (genotypes1 - genotypes2) using weighted least squares.

    Inputs:
      genotypes1_t: haplotype 1 dosages (variants x samples)
      genotypes2_t: haplotype 2 dosages (variants x samples)
      counts1_t: haplotype 1 read counts (samples)
      counts2_t: haplotype 2 read counts (samples)
    """
    # Cast to float64 for numerical precision (CUDA and CPU both support it)
    genotypes1_t = genotypes1_t.to(torch.float64)
    genotypes2_t = genotypes2_t.to(torch.float64)
    counts1_t = counts1_t.to(torch.float64)
    counts2_t = counts2_t.to(torch.float64)

    device = genotypes1_t.device
    dtype = genotypes1_t.dtype

    h1 = genotypes1_t.clone()
    h2 = genotypes2_t.clone()
    h1[torch.isnan(h1)] = 0.5
    h2[torch.isnan(h2)] = 0.5
    X_t = (h1 - h2)
    
    # Avoid log(0)
    mask_nonzero = (counts1_t > 0) & (counts2_t > 0)
    y_t = torch.zeros_like(counts1_t)
    y_t[mask_nonzero] = torch.log(counts1_t[mask_nonzero] / counts2_t[mask_nonzero])
    
    # Filter samples
    m_t = (counts1_t >= asc_cutoff) & (counts2_t >= asc_cutoff) & \
          (counts1_t <= asc_cap) & (counts2_t <= asc_cap)
    
    no_input = (counts1_t.sum() == 0) and (counts2_t.sum() == 0)
    sample_size = m_t.sum().item()
    if no_input or sample_size <= 2:
        num_variants = genotypes1_t.shape[0]
        nan_t = torch.full([num_variants], np.nan, device=device, dtype=dtype)
        if no_input:
            flag = 'no_ase_snps:ref_and_alt_counts_all_zero'
        else:
            flag = f'low_counts:n={sample_size},cutoff={asc_cutoff}'
        return nan_t, nan_t, nan_t, sample_size, flag

    X_f_t = X_t[:, m_t]
    y_f_t = y_t[m_t]
    c1_f_t = counts1_t[m_t]
    c2_f_t = counts2_t[m_t]
    
    # mask for monomorphic variants
    is_mono = (X_f_t.max(1)[0] == X_f_t.min(1)[0])
    
    valid_mask = ~is_mono
    
    num_variants = X_f_t.shape[0]
    b = torch.full([num_variants], np.nan, device=device, dtype=dtype)
    b_se = torch.full([num_variants], np.nan, device=device, dtype=dtype)
    tstat = torch.full([num_variants], np.nan, device=device, dtype=dtype)

    if valid_mask.any():
        X_v = X_f_t[valid_mask]
        
        # Weights: harmonic sum 1/(1/c1 + 1/c2)
        w_t = 1.0 / (1.0 / c1_f_t + 1.0 / c2_f_t)
        
        # Weight capping logic from R implementation
        weight_cap_val = min(weight_cap, sample_size // 10)
        w_cutoff = w_t.min() * weight_cap_val
        w_t = torch.clamp(w_t, max=w_cutoff)
        
        # Weighted regression via transformation: sqrt(W)y ~ sqrt(W)X (no intercept)
        sw_t = torch.sqrt(w_t)
        y_w_t = y_f_t * sw_t
        X_w_t = X_v * sw_t
        
        # Solve variant-wise: b = (X_w^T X_w)^-1 (X_w^T y_w)
        XtX = (X_w_t**2).sum(1)
        Xty = (X_w_t * y_w_t).sum(1)
        
        b_v = Xty / XtX
        
        # Standard error: se = sqrt(rss / dof / XtX)
        rss = ((y_w_t.unsqueeze(0) - b_v.unsqueeze(1) * X_w_t)**2).sum(1)
        dof = sample_size - 1
        sigma2 = rss / dof
        b_se_v = torch.sqrt(sigma2 / XtX)
        
        b[valid_mask] = b_v
        b_se[valid_mask] = b_se_v
        tstat[valid_mask] = b_v / b_se_v
    
    return tstat, b, b_se, sample_size, 'ok'


def meta_analyze(trc_b, trc_se, trc_n, asc_b, asc_se, asc_n, n_cutoff=15):
    """
    Meta-analyze TRC and ASC results using inverse-variance weighting.
    """
    device = trc_b.device
    dtype = trc_b.dtype
    
    # Initialize with NaNs
    meta_b = torch.full_like(trc_b, np.nan)
    meta_se = torch.full_like(trc_se, np.nan)
    meta_method = np.array(['None'] * len(trc_b), dtype=object)

    # 1. Inverse-variance meta-analysis where both pass n_cutoff
    mask_both = (trc_n >= n_cutoff) & (asc_n >= n_cutoff) & \
                (~torch.isnan(trc_b)) & (~torch.isnan(asc_b))
    
    if mask_both.any():
        w_trc = 1.0 / (trc_se[mask_both]**2)
        w_asc = 1.0 / (asc_se[mask_both]**2)
        meta_b[mask_both] = (w_trc * trc_b[mask_both] + w_asc * asc_b[mask_both]) / (w_trc + w_asc)
        meta_se[mask_both] = torch.sqrt(1.0 / (w_trc + w_asc))
        meta_method[mask_both.cpu().numpy()] = 'meta'

    # 2. For remaining variants, pick the one with more samples (matching R logic)
    # If TRC has more samples or same and meta hasn't been filled
    mask_trc_better = (trc_n >= asc_n) & (torch.from_numpy(meta_method == 'None').to(device)) & (~torch.isnan(trc_b))
    if mask_trc_better.any():
        meta_b[mask_trc_better] = trc_b[mask_trc_better]
        meta_se[mask_trc_better] = trc_se[mask_trc_better]
        meta_method[mask_trc_better.cpu().numpy()] = 'trc'
        
    mask_asc_better = (asc_n > trc_n) & (torch.from_numpy(meta_method == 'None').to(device)) & (~torch.isnan(asc_b))
    if mask_asc_better.any():
        meta_b[mask_asc_better] = asc_b[mask_asc_better]
        meta_se[mask_asc_better] = asc_se[mask_asc_better]
        meta_method[mask_asc_better.cpu().numpy()] = 'asc'

    # Final fallback for cases where one is NaN and the other is not
    mask_trc_only = torch.isnan(meta_b) & (~torch.isnan(trc_b))
    if mask_trc_only.any():
        meta_b[mask_trc_only] = trc_b[mask_trc_only]
        meta_se[mask_trc_only] = trc_se[mask_trc_only]
        meta_method[mask_trc_only.cpu().numpy()] = 'trc'
        
    mask_asc_only = torch.isnan(meta_b) & (~torch.isnan(asc_b))
    if mask_asc_only.any():
        meta_b[mask_asc_only] = asc_b[mask_asc_only]
        meta_se[mask_asc_only] = asc_se[mask_asc_only]
        meta_method[mask_asc_only.cpu().numpy()] = 'asc'

    # Calculate p-values
    tstat = meta_b / meta_se

    # Compute log10(pval) using the appropriate distribution per variant:
    #   - method='meta': z-distribution (n_trc + n_asc >= 2*n_cutoff >> n_cutoff)
    #   - method='trc':  t-distribution (df=trc_n) if trc_n <= n_cutoff, else z
    #   - method='asc':  t-distribution (df=asc_n) if asc_n <= n_cutoff, else z
    # This matches R's get_pval_fast_() which uses t when n <= n_cutoff, z otherwise.
    log10_pval = torch.full_like(tstat, np.nan)
    valid_t = ~torch.isnan(tstat)

    # Partition valid variants into those needing t-dist vs z-dist
    method_arr = meta_method  # numpy object array

    # z-distribution: method='meta', or method='trc'/'asc' with n > n_cutoff
    mask_z = valid_t & (
        (torch.from_numpy(method_arr == 'meta').to(device)) |
        (torch.from_numpy(method_arr == 'trc').to(device) & (trc_n > n_cutoff)) |
        (torch.from_numpy(method_arr == 'asc').to(device) & (asc_n > n_cutoff))
    )

    # t-distribution: method='trc' with trc_n <= n_cutoff
    mask_t_trc = valid_t & torch.from_numpy(method_arr == 'trc').to(device) & (trc_n <= n_cutoff)

    # t-distribution: method='asc' with asc_n <= n_cutoff
    mask_t_asc = valid_t & torch.from_numpy(method_arr == 'asc').to(device) & (asc_n <= n_cutoff)

    # z-distribution branch (numerically stable via log_ndtr)
    if mask_z.any():
        log_pval_z = np.log(2) + torch.special.log_ndtr(-torch.abs(tstat[mask_z]))
        log10_pval[mask_z] = log_pval_z / np.log(10)

    # t-distribution branch for trc-only variants (df = trc_n, scalar per gene)
    if mask_t_trc.any():
        t_arr = tstat[mask_t_trc].cpu().numpy()
        log10_p_t = (np.log(2) + scipy.stats.t.logsf(np.abs(t_arr), df=trc_n)) / np.log(10)
        log10_pval[mask_t_trc] = torch.from_numpy(log10_p_t).to(device=device, dtype=tstat.dtype)

    # t-distribution branch for asc-only variants (df = asc_n, scalar per gene)
    if mask_t_asc.any():
        t_arr = tstat[mask_t_asc].cpu().numpy()
        log10_p_t = (np.log(2) + scipy.stats.t.logsf(np.abs(t_arr), df=asc_n)) / np.log(10)
        log10_pval[mask_t_asc] = torch.from_numpy(log10_p_t).to(device=device, dtype=tstat.dtype)

    # Nominal p-values (will underflow to 0 for extreme t-stats)
    meta_p = torch.pow(10.0, log10_pval)

    # Flag p-values that underflowed to zero despite having a valid t-statistic.
    # float64 smallest normal: ~2.2e-308 → log10 ≈ -307.7
    min_log10 = np.log10(np.finfo(np.float64).tiny)
    pval_underflow = valid_t & (log10_pval < min_log10)

    return meta_b, meta_se, meta_p, meta_method, log10_pval, pval_underflow


def mixqtl(genotypes1_t, genotypes2_t, counts1_t, counts2_t, y_total_t, lib_size_t=None,
           covariates_t=None, trc_cutoff=20, asc_cutoff=5, weight_cap=100, asc_cap=5000,
           n_cutoff=15, logger=None, verbose=True):
    """
    Combined MixQTL model.
    """
    if logger is None:
        logger = SimpleLogger(verbose=verbose)

    # Cast all inputs to float64 for numerical precision (CUDA and CPU both support it)
    genotypes1_t = genotypes1_t.to(torch.float64)
    genotypes2_t = genotypes2_t.to(torch.float64)
    counts1_t = counts1_t.to(torch.float64)
    counts2_t = counts2_t.to(torch.float64)
    y_total_t = y_total_t.to(torch.float64)
    if lib_size_t is not None:
        lib_size_t = lib_size_t.to(torch.float64)
    if covariates_t is not None:
        covariates_t = covariates_t.to(torch.float64)

    device = genotypes1_t.device
    n_variants, n_samples = genotypes1_t.shape
    start_time = time.time()

    logger.write('mixQTL mapping')
    logger.write(f'  * {n_samples} samples')
    logger.write(f'  * {n_variants} variants')
    logger.write(f'  * device: {device}')
    if covariates_t is not None:
        logger.write(f'  * {covariates_t.shape[1] - 1} covariates')
    logger.write(f'  * TRC count threshold: {trc_cutoff}')
    logger.write(f'  * ASC count threshold: {asc_cutoff}  (cap: {asc_cap})')
    logger.write(f'  * weight cap: {weight_cap}')
    logger.write(f'  * meta-analysis n threshold: {n_cutoff}')

    # 1. TRC model
    # Xtrc = (h1 + h2) / 2
    # Impute missing genotypes to 0.5 BEFORE summing, to match R
    logger.write('  * running TRC model')
    h1 = genotypes1_t.clone()
    h2 = genotypes2_t.clone()
    h1[torch.isnan(h1)] = 0.5
    h2[torch.isnan(h2)] = 0.5
    genotypes_t = h1 + h2

    trc_res = trc(genotypes_t, y_total_t, lib_size_t=lib_size_t, covariates_t=covariates_t,
                  count_threshold=trc_cutoff, return_af=False)
    trc_tstat, trc_b, trc_se, trc_n, trc_flag = trc_res
    logger.write(f'    * {trc_n} samples passed TRC count threshold ({trc_flag})')

    # 2. ASC model
    logger.write('  * running ASC model')
    asc_res = asc(genotypes1_t, genotypes2_t, counts1_t, counts2_t,
                  asc_cutoff=asc_cutoff, weight_cap=weight_cap, asc_cap=asc_cap)
    asc_tstat, asc_b, asc_se, asc_n, asc_flag = asc_res
    logger.write(f'    * {asc_n} samples passed ASC count threshold ({asc_flag})')

    # 3. Meta-analysis
    logger.write('  * running meta-analysis')
    meta_b, meta_se, meta_p, meta_method, log10_pval, pval_underflow = meta_analyze(
        trc_b, trc_se, trc_n, asc_b, asc_se, asc_n, n_cutoff=n_cutoff)
    method_counts = {m: (meta_method == m).sum() for m in ['meta', 'trc', 'asc', 'None']}
    logger.write(f'    * meta: {method_counts["meta"]}  trc-only: {method_counts["trc"]}  '
                 f'asc-only: {method_counts["asc"]}  no result: {method_counts["None"]}')

    logger.write(f'  Time elapsed: {(time.time()-start_time)/60:.2f} min')
    logger.write('done.')

    n_underflow = pval_underflow.sum().item()
    if n_underflow > 0:
        logger.write(f'  * WARNING: {n_underflow} p-values underflowed to zero '
                     f'(use log10_pval for accurate values)')

    return {
        'trc_b': trc_b, 'trc_se': trc_se, 'trc_n': trc_n, 'trc_flag': trc_flag,
        'asc_b': asc_b, 'asc_se': asc_se, 'asc_n': asc_n, 'asc_flag': asc_flag,
        'meta_b': meta_b, 'meta_se': meta_se, 'meta_p': meta_p, 'meta_method': meta_method,
        'log10_pval': log10_pval, 'pval_underflow': pval_underflow
    }
