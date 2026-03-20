import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(1, os.path.dirname(__file__))
import cis
from core import *


def trc(genotypes_t, counts_t, lib_size_t=None, covariates_t=None, select_covariates=True,
        count_threshold=0, imputation='offset', mode='standard', return_af=False):
    """
    Inputs
      genotypes_t: dosages (variants x samples)
      counts_t: total read counts
      lib_size_t: library size
      covariates_t: covariates matrix, first column must be intercept
      mode: if 'standard', parallel regression for each variant in genotypes_t
            if 'multi', multiple regression for all variants in genotypes_t

    Outputs:
      tstat, beta, beta_se, sample_size {af, ma_samples, ma_counts}  (mode='standard')
      beta, beta_se  (mode='multi')
    """
    if lib_size_t is None:
        lib_size_t = torch.ones_like(counts_t)

    # R: trc = log(trc / 2 / lib_size) - cov
    y_full_t = torch.log(counts_t / 2.0 / lib_size_t).float()
    genotypes_t = genotypes_t.float()
    
    if covariates_t is not None:
        covariates_t = covariates_t.float()
        if select_covariates:
            # select significant covariates using only nonzero counts to avoid log(-inf)
            # but wait, counts_t >= count_threshold is applied later.
            m_valid = ~torch.isinf(y_full_t) & ~torch.isnan(y_full_t)
            b_t, b_se_t = linreg(covariates_t[m_valid, :], y_full_t[m_valid], dtype=torch.float32)
            tstat_t = b_t / b_se_t
            m = tstat_t.abs() > 2
            m[0] = True  # keep intercept
            sel_covariates_t = covariates_t[:, m]
        else:
            sel_covariates_t = covariates_t

        # R's cov_offset is subtracted. Here we can use Residualizer or subtract predicted values.
        # matrix_ls_trc assumes cov is already estimated.
        # For simplicity and consistency with tensorqtl, we use Residualizer on the valid samples.
        residualizer = Residualizer(sel_covariates_t) # Residualizer in core.py handles intercept if not careful
    else:
        residualizer = None

    m_t = counts_t >= count_threshold
    sample_size = m_t.sum().item()

    if mode == 'standard':
        # genotypes_t/2 to match R's Xtrc = (h1+h2)/2
        res = cis.calculate_cis_nominal(genotypes_t[:, m_t] / 2, y_full_t[m_t], residualizer=residualizer, return_af=False)
        if return_af:
            af, ma_samples, ma_counts = get_allele_stats(genotypes_t)
            return *res, sample_size, af, ma_samples, ma_counts
        else:
            return *res, sample_size

    elif mode.startswith('multi'):
        X_t = torch.cat([torch.ones([m_t.sum(), 1], dtype=bool).to(genotypes_t.device), genotypes_t[:, m_t].T / 2], axis=1)
        b_t, b_se_t = linreg(X_t, y_t[m_t], dtype=torch.float32)
        return b_t[1:], b_se_t[1:]


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
    device = genotypes1_t.device
    X_t = (genotypes1_t - genotypes2_t).float()
    counts1_t = counts1_t.float()
    counts2_t = counts2_t.float()
    
    # Avoid log(0)
    mask_nonzero = (counts1_t > 0) & (counts2_t > 0)
    y_t = torch.zeros_like(counts1_t)
    y_t[mask_nonzero] = torch.log(counts1_t[mask_nonzero] / counts2_t[mask_nonzero])
    
    # Filter samples
    m_t = (counts1_t >= asc_cutoff) & (counts2_t >= asc_cutoff) & \
          (counts1_t <= asc_cap) & (counts2_t <= asc_cap)
    
    sample_size = m_t.sum().item()
    if sample_size <= 2:
        num_variants = X_t.shape[0]
        nan_t = torch.full([num_variants], np.nan, device=device)
        return nan_t, nan_t, nan_t, sample_size

    X_f_t = X_t[:, m_t]
    y_f_t = y_t[m_t]
    c1_f_t = counts1_t[m_t]
    c2_f_t = counts2_t[m_t]
    
    # Weights: harmonic sum 1/(1/c1 + 1/c2)
    w_t = 1.0 / (1.0 / c1_f_t + 1.0 / c2_f_t)
    
    # Weight capping logic from R implementation
    weight_cap_val = min(weight_cap, sample_size // 10)
    w_cutoff = w_t.min() * weight_cap_val
    w_t = torch.clamp(w_t, max=w_cutoff)
    
    # Weighted regression via transformation: sqrt(W)y ~ sqrt(W)X (no intercept)
    sw_t = torch.sqrt(w_t)
    y_w_t = y_f_t * sw_t
    X_w_t = X_f_t * sw_t
    
    # Solve variant-wise: b = (X_w^T X_w)^-1 (X_w^T y_w)
    # X_w_t is (variants x samples)
    XtX = (X_w_t**2).sum(1)
    Xty = (X_w_t * y_w_t).sum(1)
    
    mask_valid = XtX > 0
    b = torch.full_like(XtX, np.nan)
    b[mask_valid] = Xty[mask_valid] / XtX[mask_valid]
    
    # Standard error: se = sqrt(rss / dof / XtX)
    # rss = sum((y_w - b*X_w)^2)
    # Using unsqueeze for broadcasting: (variants, samples) - (variants, 1) * (variants, samples)
    rss = ((y_w_t.unsqueeze(0) - b.unsqueeze(1) * X_w_t)**2).sum(1)
    dof = sample_size - 1
    sigma2 = rss / dof
    b_se = torch.sqrt(sigma2 / XtX)
    tstat = b / b_se
    
    return tstat, b, b_se, sample_size


def meta_analyze(trc_b, trc_se, trc_n, asc_b, asc_se, asc_n, n_cutoff=15):
    """
    Meta-analyze TRC and ASC results using inverse-variance weighting.
    """
    device = trc_b.device
    trc_b = trc_b.float()
    trc_se = trc_se.float()
    asc_b = asc_b.float()
    asc_se = asc_se.float()
    
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
    # Normal distribution approximation for p-values
    meta_p = 2 * torch.distributions.Normal(0, 1).cdf(-torch.abs(tstat))
    
    return meta_b, meta_se, meta_p, meta_method


def mixqtl(genotypes1_t, genotypes2_t, counts1_t, counts2_t, y_total_t, lib_size_t=None,
           covariates_t=None, trc_cutoff=20, asc_cutoff=5, weight_cap=100, asc_cap=5000, n_cutoff=15):
    """
    Combined MixQTL model.
    """
    # 1. TRC model
    # Xtrc = (h1 + h2) / 2
    genotypes_t = genotypes1_t + genotypes2_t
    trc_res = trc(genotypes_t, y_total_t, lib_size_t=lib_size_t, covariates_t=covariates_t, 
                  count_threshold=trc_cutoff, return_af=False)
    trc_tstat, trc_b, trc_se, trc_n = trc_res
    
    # 2. ASC model
    asc_res = asc(genotypes1_t, genotypes2_t, counts1_t, counts2_t,
                  asc_cutoff=asc_cutoff, weight_cap=weight_cap, asc_cap=asc_cap)
    asc_tstat, asc_b, asc_se, asc_n = asc_res
    
    # 3. Meta-analysis
    meta_b, meta_se, meta_p, meta_method = meta_analyze(trc_b, trc_se, trc_n, 
                                                        asc_b, asc_se, asc_n, 
                                                        n_cutoff=n_cutoff)
    
    return {
        'trc_b': trc_b, 'trc_se': trc_se, 'trc_n': trc_n,
        'asc_b': asc_b, 'asc_se': asc_se, 'asc_n': asc_n,
        'meta_b': meta_b, 'meta_se': meta_se, 'meta_p': meta_p, 'meta_method': meta_method
    }
