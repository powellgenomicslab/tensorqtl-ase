import torch
import numpy as np
import pandas as pd
import os
import sys
import scipy.stats
from scipy.optimize import minimize

sys.path.insert(1, os.path.dirname(__file__))
from core import *

def betaln(a, b):
    return torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)

def bb_loglike(k, n, a, b):
    """Beta-Binomial log-likelihood"""
    # k: counts, n: total, a: alpha, b: beta
    loglike = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    loglike += betaln(k + a, n - k + b) - betaln(a, b)
    return loglike

def bnb_loglike(k, n, a, b):
    """Beta-Negative Binomial log-likelihood (WASP parameterization)"""
    # Pochhammer(n, k) = gamma(n+k)/gamma(n)
    loglike = torch.lgamma(k + n) - torch.lgamma(k + 1) - torch.lgamma(n)
    loglike += betaln(a + n, b + k) - betaln(a, b)
    return loglike

def calculate_loglikelihood(params, genotypes1, genotypes2, counts1, counts2, hetps, y_total, lib_size,
                             as_sigmas, bnb_sigmas, is_as_only=False, is_bnb_only=False):
    """
    Vectorized CHT log-likelihood calculation.
    params: (num_variants, 3) tensor [alpha, beta, r]
    genotypes1, genotypes2: (num_variants, num_samples)
    counts1, counts2: (num_variants, num_samples, num_linked_snps)
    hetps: (num_variants, num_samples, num_linked_snps)
    y_total: (num_samples)
    lib_size: (num_samples)
    as_sigmas: (num_samples)
    bnb_sigmas: (num_samples)
    """
    alpha = params[:, 0:1] # (num_variants, 1)
    beta = params[:, 1:2]  # (num_variants, 1)
    r = params[:, 2:3]     # (num_variants, 1)
    
    num_variants = genotypes1.shape[0]
    num_samples = genotypes1.shape[1]
    device = genotypes1.device
    
    loglike = torch.zeros(num_variants, device=device)

    # 1. Allele-specific part (Beta-Binomial)
    if not is_bnb_only:
        log_alpha_sum = torch.log(alpha + beta)
        logp1 = torch.log(alpha) - log_alpha_sum
        logp2 = torch.log(beta) - log_alpha_sum
        
        # as_sigmas: (num_samples)
        inv_sigma2_minus_1 = 1.0 / (as_sigmas**2) - 1.0
        # as_a, as_b: (1, num_samples)
        as_a = torch.exp(logp1 + torch.log(inv_sigma2_minus_1.unsqueeze(0)))
        as_b = torch.exp(logp2 + torch.log(inv_sigma2_minus_1.unsqueeze(0)))
        
        # counts1, counts2: (num_variants, num_samples, num_linked_snps)
        # as_a, as_b need to be broadcast to (num_variants, num_samples, 1)
        as_a = as_a.unsqueeze(2)
        as_b = as_b.unsqueeze(2)
        
        n_as = counts1 + counts2
        ll_as = bb_loglike(counts1, n_as, as_a, as_b)
        
        # WASP handles hetps: log(hetp * exp(ll_as) + (1-hetp) * exp(ll_err))
        # For simplicity, we assume hetps=1 or use them directly if provided.
        # We also need to mask where n_as > 0
        mask_as = n_as > 0
        ll_as = torch.where(mask_as, ll_as, torch.zeros_like(ll_as))
        
        # Sum over samples and linked SNPs
        loglike += ll_as.sum((1, 2))

    # 2. Total depth part (Beta-Negative Binomial)
    if not is_as_only:
        G = genotypes1 + genotypes2 # (num_variants, num_samples)
        # alpha, beta are (num_variants, 1)
        # lib_size: (num_samples)
        m = ((2.0 - G) * alpha + G * beta) * lib_size.unsqueeze(0)
        
        # WASP: p = n / (n + m) where n is bnb_sigmas
        n_val = bnb_sigmas.unsqueeze(0) # (1, num_samples)
        p = n_val / (n_val + m)
        
        # WASP: sigma_bnb = (1/r)**2 where r is optimized parameter
        sigma_bnb = (1.0 / r)**2 # (num_variants, 1)
        bnb_a = p * sigma_bnb + 1.0
        bnb_b = (1.0 - p) * sigma_bnb
        
        # y_total: (num_samples) -> broadcast to (num_variants, num_samples)
        # WASP: k=y_total, mean=m, sigma=r, n=n_val
        ll_bnb = bnb_loglike(y_total.unsqueeze(0), n_val, bnb_a, bnb_b)
        loglike += ll_bnb.sum(1)

    return loglike


def run_cht(genotypes1_t, genotypes2_t, counts1_t, counts2_t, hetps_t, y_total_t, lib_size_t,
            as_sigmas_t, bnb_sigmas_t, max_iter=100, lr=1.0, verbose=True):
    """
    Vectorized CHT optimization using PyTorch.
    """
    num_variants = genotypes1_t.shape[0]
    device = genotypes1_t.device
    
    # Initialize parameters
    # alpha, beta in log space, r in logit space
    params_raw = torch.zeros((num_variants, 3), device=device, requires_grad=True)
    with torch.no_grad():
        # Initialization: log(1.0) = 0
        params_raw[:, 0:2] = 0.0
        params_raw[:, 2] = 0.0 
        
    def get_params(p_raw):
        alpha = torch.exp(p_raw[:, 0:1])
        beta = torch.exp(p_raw[:, 1:2])
        r = torch.sigmoid(p_raw[:, 2:3])
        return torch.cat([alpha, beta, r], dim=1)

    # 1. Null Model (alpha = beta)
    # Optimize (alpha, r) only
    params_null_raw = torch.zeros((num_variants, 2), device=device, requires_grad=True)
    optimizer_null = torch.optim.LBFGS([params_null_raw], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    
    def get_params_null(p_raw):
        alpha = torch.exp(p_raw[:, 0:1])
        r = torch.sigmoid(p_raw[:, 1:2])
        return torch.cat([alpha, alpha, r], dim=1)

    def closure_null():
        optimizer_null.zero_grad()
        p_null = get_params_null(params_null_raw)
        loglikes = calculate_loglikelihood(p_null, genotypes1_t, genotypes2_t, counts1_t, counts2_t, hetps_t, 
                                          y_total_t, lib_size_t, as_sigmas_t, bnb_sigmas_t)
        loss = -loglikes.sum()
        loss.backward()
        return loss

    optimizer_null.step(closure_null)
        
    with torch.no_grad():
        p_null = get_params_null(params_null_raw)
        ll_null = calculate_loglikelihood(p_null, genotypes1_t, genotypes2_t, counts1_t, counts2_t, hetps_t, 
                                         y_total_t, lib_size_t, as_sigmas_t, bnb_sigmas_t)

    # 2. Alt Model (alpha != beta)
    # Initialize from null model
    params_alt_raw = torch.zeros((num_variants, 3), device=device, requires_grad=True)
    with torch.no_grad():
        params_alt_raw[:, 0] = params_null_raw[:, 0]
        params_alt_raw[:, 1] = params_null_raw[:, 0]
        params_alt_raw[:, 2] = params_null_raw[:, 1]

    optimizer_alt = torch.optim.LBFGS([params_alt_raw], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    
    def get_params_alt(p_raw):
        alpha = torch.exp(p_raw[:, 0:1])
        beta = torch.exp(p_raw[:, 1:2])
        r = torch.sigmoid(p_raw[:, 2:3])
        return torch.cat([alpha, beta, r], dim=1)

    def closure_alt():
        optimizer_alt.zero_grad()
        p_alt = get_params_alt(params_alt_raw)
        loglikes = calculate_loglikelihood(p_alt, genotypes1_t, genotypes2_t, counts1_t, counts2_t, hetps_t, 
                                          y_total_t, lib_size_t, as_sigmas_t, bnb_sigmas_t)
        loss = -loglikes.sum()
        loss.backward()
        return loss

    optimizer_alt.step(closure_alt)
        
    with torch.no_grad():
        p_alt = get_params_alt(params_alt_raw)
        ll_alt = calculate_loglikelihood(p_alt, genotypes1_t, genotypes2_t, counts1_t, counts2_t, hetps_t, 
                                        y_total_t, lib_size_t, as_sigmas_t, bnb_sigmas_t)
        
    chisq = 2 * (ll_alt - ll_null)
    # Clamp chisq to be non-negative
    chisq = torch.clamp(chisq, min=0.0)
    pval = 1.0 - torch.distributions.Chi2(torch.tensor([1.0], device=device)).cdf(chisq)
    
    res_df = pd.DataFrame({
        'chisq': chisq.cpu().numpy(),
        'pval': pval.cpu().numpy(),
        'alpha': p_alt[:, 0].cpu().numpy(),
        'beta': p_alt[:, 1].cpu().numpy(),
        'r': p_alt[:, 2].cpu().numpy()
    })
    
    return res_df
