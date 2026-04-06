"""
mixqtl_perm.py — Permutation-based empirical p-values for mixQTL.

Applies the same strategy as tensorqtl cis.map_cis:
  - Permute sample labels using coupled permutations: shared TRC/ASC samples are
    permuted together to preserve their correlation under the null, while
    model-exclusive samples are permuted independently.
  - Track the maximum meta z² across all valid variants per permutation as the
    null distribution of the best-variant statistic.
  - Return a gene-level empirical p-value for the lead variant plus an optional
    beta-distribution approximation (analogous to map_cis pval_beta/pval_perm).

This is a standalone optional module.  It does not modify mixqtl.py.

Typical usage (one gene at a time):
    from tensorqtl import mixqtl_perm
    res = mixqtl_perm.mixqtl_perm(g1_t, g2_t, y1_t, y2_t, y_total_t,
                                   lib_size_t=lib_t, covariates_t=cov_t,
                                   nperm=10000, seed=42)
    print(res['pval_perm'], res['pval_beta'])

Memory notes:
  TRC permutations: O(n_trc_valid × chunk) — tiny, all perms in one matmul.
  ASC permutations: O(chunk × n_asc_valid × n_asc_samples) 3-D intermediate;
    perm_chunk controls this trade-off (default 200 ≈ few hundred MB at most).
"""

import torch
import numpy as np
import scipy.stats
import scipy.optimize
import os
import sys
import time

sys.path.insert(1, os.path.dirname(__file__))
from core import SimpleLogger, calculate_beta_approx_pval, linreg


# ──────────────────────────────────────────────────────────────────────────────
# TRC — pre-compute and batch-permute
# ──────────────────────────────────────────────────────────────────────────────

def _trc_prepare(geno_sum_t, y_total_t, lib_size_t, covariates_t,
                 count_threshold, device, dtype):
    """
    Pre-compute permutation-invariant TRC quantities.

    geno_sum_t : (n_variants × n_samples) — h1 + h2 after NaN imputation
    Returns a dict used by _trc_perm_chunk.
    """
    if lib_size_t is None:
        lib_size_t = torch.ones(y_total_t.shape[0], dtype=dtype, device=device)

    y_full_t = torch.log(y_total_t / (lib_size_t * 2.0))

    # Covariate offset — computed once on observed data, residuals are permuted.
    # Mirrors the select_covariates branch in mixqtl.trc().
    if covariates_t is not None:
        m_cov = ~torch.isinf(y_full_t) & ~torch.isnan(y_full_t)
        y_cov = y_full_t[m_cov]
        c_cov = covariates_t[m_cov]
        b_t, b_se_t = linreg(c_cov, y_cov, dtype=dtype)
        selected = (b_t / b_se_t)[1:].abs() > 2
        if selected.any():
            sel = torch.cat([torch.tensor([True], device=device), selected])
            b_sel, _ = linreg(c_cov[:, sel], y_cov, dtype=dtype)
            offset_t = torch.matmul(covariates_t[:, sel][:, 1:], b_sel[1:])
        else:
            offset_t = torch.zeros_like(y_full_t)
    else:
        offset_t = torch.zeros_like(y_full_t)

    y_target_t = y_full_t - offset_t

    # TRC sample mask
    m_samples = ((y_total_t >= count_threshold) &
                 ~torch.isinf(y_full_t) & ~torch.isnan(y_full_t))
    n_trc = int(m_samples.sum().item())

    # Genotype matrix for TRC-passing samples, NaN→1.0 (0.5+0.5), then /2
    X_f = geno_sum_t[:, m_samples].clone()
    X_f[torch.isnan(X_f)] = 1.0
    X_f = X_f / 2.0

    y_f = y_target_t[m_samples]

    # Filter monomorphic variants
    is_mono = (X_f.max(1)[0] == X_f.min(1)[0])
    valid_ix = torch.where(~is_mono)[0]          # (n_trc_valid,)
    X_v = X_f[valid_ix]                          # (n_trc_valid × n_trc)

    S11   = (X_v ** 2).sum(1)
    S12   = X_v.sum(1)
    S22   = float(n_trc)
    delta = torch.clamp(S11 * S22 - S12 ** 2, min=1e-30)

    return dict(X_v=X_v, y_f=y_f, S11=S11, S12=S12, S22=S22, delta=delta,
                valid_ix=valid_ix, n_trc=n_trc, n_variants=geno_sum_t.shape[0],
                m_samples=m_samples)


def _trc_perm_chunk(prep, perm_ix_chunk):
    """
    Batched OLS for a chunk of TRC permutations.

    perm_ix_chunk : LongTensor (chunk × n_trc)
    Returns (b, se) each of shape (n_trc_valid × chunk).
    """
    X_v, y_f = prep['X_v'], prep['y_f']
    S11, S12, S22, delta = prep['S11'], prep['S12'], prep['S22'], prep['delta']
    n_trc = prep['n_trc']

    y_p   = y_f[perm_ix_chunk]             # (chunk × n_trc)
    y2s   = (y_p ** 2).sum(1)              # (chunk,)
    T1    = X_v @ y_p.T                   # (n_valid × chunk)
    T2    = y_p.sum(1)                     # (chunk,)

    b1 = (S22 * T1 - S12[:, None] * T2[None, :]) / delta[:, None]
    b2 = (S11[:, None] * T2[None, :] - S12[:, None] * T1) / delta[:, None]

    rss = (y2s[None, :]
           - 2 * b1 * T1
           - 2 * b2 * T2[None, :]
           + 2 * b1 * b2 * S12[:, None]
           + b1 ** 2 * S11[:, None]
           + b2 ** 2 * S22)

    sigma = torch.sqrt(torch.clamp(rss, min=0) / (n_trc - 2))
    se1   = sigma * (S22 / delta).sqrt()[:, None]

    return b1, se1   # (n_valid × chunk), (n_valid × chunk)


# ──────────────────────────────────────────────────────────────────────────────
# ASC — pre-compute and batch-permute
# ──────────────────────────────────────────────────────────────────────────────

def _asc_prepare(genotypes1_t, genotypes2_t, counts1_t, counts2_t,
                 asc_cutoff, weight_cap, asc_cap, device, dtype):
    """
    Pre-compute permutation-invariant ASC quantities.

    Returns a dict used by _asc_perm_chunk, or None if too few ASC samples.
    """
    h1 = genotypes1_t.clone(); h2 = genotypes2_t.clone()
    h1[torch.isnan(h1)] = 0.5;  h2[torch.isnan(h2)] = 0.5
    X_t = h1 - h2   # (n_variants × n_samples)

    mask_nz = (counts1_t > 0) & (counts2_t > 0)
    y_t = torch.zeros(counts1_t.shape[0], dtype=dtype, device=device)
    y_t[mask_nz] = torch.log(counts1_t[mask_nz] / counts2_t[mask_nz])

    m_t = ((counts1_t >= asc_cutoff) & (counts2_t >= asc_cutoff) &
           (counts1_t <= asc_cap)    & (counts2_t <= asc_cap))
    n_asc = int(m_t.sum().item())

    if n_asc <= 2:
        return None

    X_f   = X_t[:, m_t]            # (n_variants × n_asc)
    y_f   = y_t[m_t]               # (n_asc,)
    c1_f  = counts1_t[m_t]
    c2_f  = counts2_t[m_t]

    w     = 1.0 / (1.0 / c1_f + 1.0 / c2_f)
    cap_mult = max(1, min(weight_cap, n_asc // 10))
    w_cap = w.min() * cap_mult
    w     = torch.clamp(w, max=w_cap)
    sw    = torch.sqrt(w)           # (n_asc,)

    # Filter monomorphic variants
    is_mono  = (X_f.max(1)[0] == X_f.min(1)[0])
    valid_ix = torch.where(~is_mono)[0]   # (n_asc_valid,)
    X_v      = X_f[valid_ix]             # (n_asc_valid × n_asc)

    return dict(X_v=X_v, y_f=y_f, sw=sw, valid_ix=valid_ix,
                n_asc=n_asc, dof=n_asc - 1, n_variants=genotypes1_t.shape[0],
                m_samples=m_t)


def _asc_perm_chunk(prep, perm_ix_chunk):
    """
    Batched WLS for a chunk of ASC permutations.

    Permuting sample indices permutes both y_f and the weights sw jointly,
    preserving the relationship between read depth and variance.

    perm_ix_chunk : LongTensor (chunk × n_asc)
    Returns (b, se) each of shape (n_asc_valid × chunk).

    Memory: (chunk × n_asc_valid × n_asc) intermediate; set perm_chunk to
    control peak usage (default 200 ≈ ~170 MB at 1000 variants, 421 samples).
    """
    X_v  = prep['X_v']   # (n_valid × n_asc)
    y_f  = prep['y_f']   # (n_asc,)
    sw   = prep['sw']    # (n_asc,)
    dof  = prep['dof']

    sw_p    = sw[perm_ix_chunk]          # (chunk × n_asc)
    y_p     = y_f[perm_ix_chunk]         # (chunk × n_asc)
    yw_p    = y_p * sw_p                 # (chunk × n_asc)

    # (chunk × n_valid × n_asc)
    Xw = X_v[None, :, :] * sw_p[:, None, :]

    XtX  = (Xw ** 2).sum(2)                          # (chunk × n_valid)
    Xty  = (Xw * yw_p[:, None, :]).sum(2)            # (chunk × n_valid)
    b    = Xty / XtX

    yw2s = (yw_p ** 2).sum(1)                        # (chunk,)
    rss  = yw2s[:, None] - 2 * b * Xty + b ** 2 * XtX
    se   = torch.sqrt(rss / dof / XtX)

    # Transpose to (n_valid × chunk) for consistent layout with TRC
    return b.T, se.T


# ──────────────────────────────────────────────────────────────────────────────
# Coupled permutation index generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_coupled_permutations(trc_mask_np, asc_mask_np, nperm, rng):
    """
    Generate coupled permutation indices for TRC and ASC models.

    Shared samples (passing both TRC and ASC filters) are permuted together
    so that the TRC-ASC correlation under the null is preserved.  Samples
    exclusive to one model are permuted independently within that model.

    Parameters
    ----------
    trc_mask_np : ndarray (n_samples,) bool — TRC-passing sample mask
    asc_mask_np : ndarray (n_samples,) bool — ASC-passing sample mask (or None)
    nperm       : int
    rng         : np.random.Generator

    Returns
    -------
    perm_ix_trc : ndarray (nperm × n_trc) int64
    perm_ix_asc : ndarray (nperm × n_asc) int64, or None if asc_mask_np is None
    """
    trc_idx = np.where(trc_mask_np)[0]
    n_trc = len(trc_idx)

    if asc_mask_np is None:
        return np.array([rng.permutation(n_trc) for _ in range(nperm)]), None

    asc_idx = np.where(asc_mask_np)[0]
    n_asc = len(asc_idx)

    # Identify shared and model-specific samples (by full-sample index)
    both_set = set(trc_idx) & set(asc_idx)

    # Positions within each filtered array that are shared vs exclusive
    trc_both_pos = np.array([i for i, idx in enumerate(trc_idx) if idx in both_set])
    trc_only_pos = np.array([i for i, idx in enumerate(trc_idx) if idx not in both_set])
    asc_both_pos = np.array([i for i, idx in enumerate(asc_idx) if idx in both_set])
    asc_only_pos = np.array([i for i, idx in enumerate(asc_idx) if idx not in both_set])

    n_both = len(trc_both_pos)
    n_trc_only = len(trc_only_pos)
    n_asc_only = len(asc_only_pos)

    perm_ix_trc = np.zeros((nperm, n_trc), dtype=np.int64)
    perm_ix_asc = np.zeros((nperm, n_asc), dtype=np.int64)

    for p in range(nperm):
        # Shared samples: same permutation applied to both models
        both_perm = rng.permutation(n_both)
        perm_ix_trc[p, trc_both_pos] = trc_both_pos[both_perm]
        perm_ix_asc[p, asc_both_pos] = asc_both_pos[both_perm]

        # Model-exclusive samples: permuted independently
        if n_trc_only > 0:
            perm_ix_trc[p, trc_only_pos] = trc_only_pos[rng.permutation(n_trc_only)]
        if n_asc_only > 0:
            perm_ix_asc[p, asc_only_pos] = asc_only_pos[rng.permutation(n_asc_only)]

    return perm_ix_trc, perm_ix_asc


# ──────────────────────────────────────────────────────────────────────────────
# Meta-analysis over permutation chunks
# ──────────────────────────────────────────────────────────────────────────────

def _meta_z2_chunk(trc_b_p, trc_se_p, asc_b_p, asc_se_p,
                   trc_valid_ix, asc_valid_ix,
                   meta_method, n_variants, device, dtype):
    """
    Combine per-permutation TRC and ASC estimates via inverse-variance
    meta-analysis and return max z² across all variants.

    trc_b_p, trc_se_p : (n_trc_valid × chunk) or None
    asc_b_p, asc_se_p : (n_asc_valid × chunk) or None — layout same as TRC after transpose
    meta_method        : np.ndarray (n_variants,) of 'meta'/'trc'/'asc'/'None'

    Returns max_z2 : (chunk,)
    """
    chunk = (trc_b_p.shape[1] if trc_b_p is not None
             else asc_b_p.shape[1])

    # Full z² array, NaN for variants with no valid result
    z2 = torch.full((n_variants, chunk), float('nan'), dtype=dtype, device=device)

    mask_trc_only = torch.from_numpy(meta_method == 'trc').to(device)
    mask_asc_only = torch.from_numpy(meta_method == 'asc').to(device)
    mask_meta     = torch.from_numpy(meta_method == 'meta').to(device)

    # ── TRC-only variants ────────────────────────────────────────────────────
    if trc_b_p is not None and mask_trc_only.any():
        gix = torch.where(mask_trc_only)[0]                 # global variant ix
        lix = torch.where(torch.isin(trc_valid_ix, gix))[0] # local in trc_b_p
        assert len(lix) == len(gix), \
            f"TRC index mismatch: {len(lix)} local vs {len(gix)} global"
        z2[gix] = (trc_b_p[lix] / trc_se_p[lix]) ** 2

    # ── ASC-only variants ────────────────────────────────────────────────────
    if asc_b_p is not None and mask_asc_only.any():
        gix = torch.where(mask_asc_only)[0]
        lix = torch.where(torch.isin(asc_valid_ix, gix))[0]
        assert len(lix) == len(gix), \
            f"ASC index mismatch: {len(lix)} local vs {len(gix)} global"
        z2[gix] = (asc_b_p[lix] / asc_se_p[lix]) ** 2

    # ── Meta variants — full inverse-variance combination ───────────────────
    if trc_b_p is not None and asc_b_p is not None and mask_meta.any():
        gix     = torch.where(mask_meta)[0]                       # (n_meta,)
        trc_lix = torch.where(torch.isin(trc_valid_ix, gix))[0]  # (n_meta,)
        asc_lix = torch.where(torch.isin(asc_valid_ix, gix))[0]  # (n_meta,)
        assert len(trc_lix) == len(gix) == len(asc_lix), \
            f"Meta index mismatch: TRC {len(trc_lix)}, ASC {len(asc_lix)}, global {len(gix)}"

        bt = trc_b_p[trc_lix]   # (n_meta × chunk)
        st = trc_se_p[trc_lix]
        ba = asc_b_p[asc_lix]
        sa = asc_se_p[asc_lix]

        wt = 1.0 / st ** 2      # (n_meta × chunk)
        wa = 1.0 / sa ** 2
        w_sum = wt + wa

        meta_b_p  = (wt * bt + wa * ba) / w_sum
        meta_se_p = torch.sqrt(1.0 / w_sum)
        z2[gix]   = (meta_b_p / meta_se_p) ** 2

    # Max z² across variants, treating NaN as 0 (no contribution)
    z2 = torch.nan_to_num(z2, nan=0.0)
    return z2.max(0).values   # (chunk,)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def mixqtl_perm(genotypes1_t, genotypes2_t, counts1_t, counts2_t, y_total_t,
                lib_size_t=None, covariates_t=None,
                trc_cutoff=20, asc_cutoff=5, weight_cap=100, asc_cap=5000,
                n_cutoff=15, nperm=10000, seed=None, perm_chunk=200,
                beta_approx=True, logger=None, verbose=True):
    """
    mixQTL with permutation-based empirical p-values (gene-level).

    Runs the nominal mixQTL model then generates `nperm` permutations of
    sample labels — jointly applied to y_total, y1, y2, lib_size — to build
    the null distribution of the best meta z² across all variants.  Returns
    the nominal results plus an empirical p-value for the lead variant.

    Parameters
    ----------
    genotypes1_t, genotypes2_t : Tensor (n_variants × n_samples)
        Haplotype dosages (0, 0.5, or 1).
    counts1_t, counts2_t : Tensor (n_samples,)
        Allele-specific read counts (haplotype 1 and 2).
    y_total_t : Tensor (n_samples,)
        Total read counts.
    lib_size_t : Tensor (n_samples,), optional
    covariates_t : Tensor (n_samples × n_cov), optional
        First column must be the intercept (ones).
    trc_cutoff, asc_cutoff, weight_cap, asc_cap, n_cutoff
        Forwarded to mixqtl().
    nperm : int
        Number of permutations (default 10 000).
    seed : int, optional
        Random seed for reproducibility.
    perm_chunk : int
        Permutations per ASC batch (trades peak memory vs speed; default 200).
    beta_approx : bool
        Fit a beta distribution to the permutation null for a smoother tail
        p-value estimate (analogous to map_cis pval_beta).

    Returns
    -------
    dict
        All keys from mixqtl() plus:
        lead_ix           : int    — index of the lead (lowest meta p-value) variant
        pval_perm         : float  — empirical p-value  = (#perms ≥ obs_z² + 1) / (nperm + 1)
        pval_beta         : float  — beta-approximation p-value (nan if beta_approx=False)
        beta_shape1       : float
        beta_shape2       : float
        true_df           : float  — effective DoF from beta fit
        pval_true_df      : float
        perm_z2_dist      : ndarray (nperm,) — null distribution of max z²
        n_variants_tested : int    — number of variants with a valid meta result
    """
    if logger is None:
        logger = SimpleLogger(verbose=verbose)

    device = genotypes1_t.device
    dtype  = genotypes1_t.dtype
    n_variants, n_samples = genotypes1_t.shape
    t0 = time.time()

    logger.write('mixQTL permutation mapping')
    logger.write(f'  * {n_samples} samples, {n_variants} variants')
    logger.write(f'  * device: {device}')
    if covariates_t is not None:
        logger.write(f'  * {covariates_t.shape[1] - 1} covariates')
    logger.write(f'  * nperm={nperm}, perm_chunk={perm_chunk}')
    if seed is not None:
        logger.write(f'  * seed: {seed}')

    # ── 1. Nominal mixQTL ────────────────────────────────────────────────────
    from mixqtl import mixqtl as _mixqtl
    nominal = _mixqtl(
        genotypes1_t, genotypes2_t, counts1_t, counts2_t, y_total_t,
        lib_size_t=lib_size_t, covariates_t=covariates_t,
        trc_cutoff=trc_cutoff, asc_cutoff=asc_cutoff,
        weight_cap=weight_cap, asc_cap=asc_cap, n_cutoff=n_cutoff,
        logger=logger, verbose=verbose,
    )
    meta_p      = nominal['meta_p']
    meta_b      = nominal['meta_b']
    meta_se     = nominal['meta_se']
    meta_method = nominal['meta_method']   # np.ndarray (n_variants,)

    # Lead variant
    valid_mask = ~torch.isnan(meta_p)
    n_valid = int(valid_mask.sum().item())
    if n_valid == 0:
        logger.write('  WARNING: no valid meta p-values')
        _nan = dict(lead_ix=np.nan, pval_perm=np.nan, pval_beta=np.nan,
                    beta_shape1=np.nan, beta_shape2=np.nan, true_df=np.nan,
                    pval_true_df=np.nan, perm_z2_dist=np.full(nperm, np.nan),
                    n_variants_tested=0)
        nominal.update(_nan)
        return nominal

    p_finite = meta_p.clone()
    p_finite[~valid_mask] = 1.0
    lead_ix  = int(p_finite.argmin().item())
    obs_z2   = float((meta_b[lead_ix] / meta_se[lead_ix]) ** 2)
    logger.write(f'  * lead variant: index {lead_ix}, '
                 f'meta_p={float(meta_p[lead_ix]):.3e}, z²={obs_z2:.3f}')
    logger.write(f'  * {n_valid} variants with valid meta results')

    # ── 2. Pre-compute permutation-invariant arrays ──────────────────────────
    logger.write('  * preparing TRC arrays ...')
    h1 = genotypes1_t.clone(); h2 = genotypes2_t.clone()
    h1[torch.isnan(h1)] = 0.5;  h2[torch.isnan(h2)] = 0.5
    geno_sum_t = h1 + h2

    trc_prep = _trc_prepare(geno_sum_t, y_total_t, lib_size_t, covariates_t,
                            trc_cutoff, device, dtype)
    trc_valid_ix = trc_prep['valid_ix']
    n_trc        = trc_prep['n_trc']

    logger.write('  * preparing ASC arrays ...')
    asc_prep = _asc_prepare(genotypes1_t, genotypes2_t, counts1_t, counts2_t,
                            asc_cutoff, weight_cap, asc_cap, device, dtype)
    if asc_prep is not None:
        asc_valid_ix = asc_prep['valid_ix']
        n_asc        = asc_prep['n_asc']
        logger.write(f'    * {n_asc} ASC samples, '
                     f'{len(asc_valid_ix)} valid ASC variants')
    else:
        asc_valid_ix = torch.tensor([], dtype=torch.long, device=device)
        n_asc = 0
        logger.write('    * ASC model skipped (insufficient samples)')

    # ── 3. Generate coupled permutation indices ────────────────────────────
    # Shared samples (passing both TRC and ASC) are permuted together so that
    # the TRC-ASC correlation under the null is preserved.  Model-exclusive
    # samples are permuted independently within their respective sets.
    rng = np.random.default_rng(seed)
    trc_mask_np = trc_prep['m_samples'].cpu().numpy()
    asc_mask_np = asc_prep['m_samples'].cpu().numpy() if asc_prep is not None else None
    perm_ix_trc, perm_ix_asc = _generate_coupled_permutations(
        trc_mask_np, asc_mask_np, nperm, rng)
    n_shared = int((trc_mask_np & asc_mask_np).sum()) if asc_mask_np is not None else 0
    logger.write(f'    * {n_shared}/{n_trc} TRC samples shared with ASC '
                 f'(coupled permutation)')

    # ── 4. Permutation loop ──────────────────────────────────────────────────
    logger.write('  * running permutations ...')
    perm_best_z2 = np.zeros(nperm, dtype=np.float64)

    for start in range(0, nperm, perm_chunk):
        end = min(start + perm_chunk, nperm)
        s   = slice(start, end)

        trc_b_p, trc_se_p = _trc_perm_chunk(
            trc_prep,
            torch.from_numpy(perm_ix_trc[s]).to(device))

        if asc_prep is not None:
            asc_b_p, asc_se_p = _asc_perm_chunk(
                asc_prep,
                torch.from_numpy(perm_ix_asc[s]).to(device))
        else:
            asc_b_p = asc_se_p = None

        chunk_z2 = _meta_z2_chunk(
            trc_b_p, trc_se_p, asc_b_p, asc_se_p,
            trc_valid_ix, asc_valid_ix,
            meta_method, n_variants, device, dtype)

        perm_best_z2[start:end] = chunk_z2.cpu().numpy()

    # ── 5. Empirical p-value ─────────────────────────────────────────────────
    pval_perm = float((np.sum(perm_best_z2 >= obs_z2) + 1) / (nperm + 1))
    logger.write(f'  * pval_perm = {pval_perm:.4g}')

    # ── 6. Beta approximation (optional) ────────────────────────────────────
    pval_beta = beta_shape1 = beta_shape2 = true_df = pval_true_df = np.nan
    if beta_approx:
        # Pick initial DoF based on the lead variant's meta_method.
        # fit_beta_parameters will optimise the effective DoF, so this is a
        # starting point; using the right model's DoF improves convergence.
        lead_method = meta_method[lead_ix]
        if lead_method == 'asc':
            dof_init = float(n_asc - 1)
        else:
            # 'trc' or 'meta' — use TRC DoF as starting point
            dof_init = float(n_trc - 2)
        r2_perm    = perm_best_z2 / (perm_best_z2 + dof_init)
        r2_nominal = obs_z2       / (obs_z2       + dof_init)
        try:
            pval_beta, beta_shape1, beta_shape2, true_df, pval_true_df = \
                calculate_beta_approx_pval(r2_perm, r2_nominal, dof_init)
            logger.write(f'  * pval_beta = {pval_beta:.4g}  '
                         f'(shape1={beta_shape1:.3f}, shape2={beta_shape2:.3f},'
                         f' true_df={true_df:.1f})')
        except Exception as exc:
            logger.write(f'  * beta approximation failed: {exc}')

    logger.write(f'  Time elapsed: {(time.time() - t0) / 60:.2f} min')
    logger.write('done.')

    nominal.update(dict(
        lead_ix           = lead_ix,
        pval_perm         = pval_perm,
        pval_beta         = float(pval_beta),
        beta_shape1       = float(beta_shape1),
        beta_shape2       = float(beta_shape2),
        true_df           = float(true_df),
        pval_true_df      = float(pval_true_df),
        perm_z2_dist      = perm_best_z2,
        n_variants_tested = n_valid,
    ))
    return nominal
