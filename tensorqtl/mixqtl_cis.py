import torch
import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from core import SimpleLogger, get_allele_stats
from mixqtl import mixqtl as _mixqtl


def cis_nominal(
    genotype_df,
    variant_df,
    phenotype_df,
    phenotype_pos_df,
    covariates_df=None,
    prefix='',
    window=500_000,
    trc_cutoff=20,
    asc_cutoff=5,
    weight_cap=100,
    asc_cap=5000,
    n_cutoff=15,
    maf_threshold=0.05,
    hap1_df=None,
    hap2_df=None,
    counts1_df=None,
    counts2_df=None,
    lib_size_s=None,
    device=None,
    logger=None,
):
    """
    cis-QTL mapping using the mixQTL model (TRC + ASC meta-analysis).

    Maps all variant-gene pairs within the cis window for every gene in
    phenotype_pos_df, combining total read count (TRC) and allele-specific
    count (ASC) models via inverse-variance meta-analysis.

    Parameters
    ----------
    genotype_df : pd.DataFrame
        Variants x samples.  Dosage values (0/1/2 or 0/0.5/1).
    variant_df : pd.DataFrame
        Index = variant_id.  Must contain 'chrom' and 'pos' columns.
    phenotype_df : pd.DataFrame
        Genes x samples.  Total read counts.
    phenotype_pos_df : pd.DataFrame
        Index = gene_id.  Must contain 'chrom' and 'start' columns.
        'start' is used as the TSS for every gene.
    covariates_df : pd.DataFrame, optional
        Samples x covariates.
    prefix : str
        Output file prefix.  If non-empty, results are written to
        ``{prefix}.cis_qtl_pairs.txt.gz``.
    window : int
        Cis window in bp: [TSS - window, TSS + window].
    trc_cutoff, asc_cutoff, weight_cap, asc_cap, n_cutoff : int/float
        Passed directly to mixqtl().
    maf_threshold : float
        Variants with MAF < maf_threshold are skipped.
    hap1_df, hap2_df : pd.DataFrame, optional
        Variants x samples.  Haplotype 1/2 dosages (0/0.5/1).
        If None, genotype_df / 2 is used for both (additive fallback).
    counts1_df, counts2_df : pd.DataFrame, optional
        Genes x samples.  ASC haplotype 1/2 counts.
        If None, zeros are passed (TRC-only mode).
    lib_size_s : pd.Series, optional
        Library sizes, indexed by sample_id.  If None, ones are used.
    device : torch.device, optional
        Compute device.  Auto-detected (CUDA > CPU) if None.
        Note: mixQTL uses float64, so MPS is not supported.
    logger : SimpleLogger, optional

    Returns
    -------
    pd.DataFrame with columns:
        phenotype_id, variant_id, tss_distance, af, ma_samples, ma_count,
        pval_nominal, slope, slope_se,
        trc_b, trc_se, trc_n, asc_b, asc_se, asc_n, meta_method
    """
    if logger is None:
        logger = SimpleLogger()

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.write('mixQTL cis-nominal mapping')

    # ------------------------------------------------------------------
    # Sample alignment: intersect across all provided DataFrames
    # ------------------------------------------------------------------
    sample_sets = [set(genotype_df.columns), set(phenotype_df.columns)]
    if hap1_df is not None:
        sample_sets.append(set(hap1_df.columns))
    if hap2_df is not None:
        sample_sets.append(set(hap2_df.columns))
    if counts1_df is not None:
        sample_sets.append(set(counts1_df.columns))
    if counts2_df is not None:
        sample_sets.append(set(counts2_df.columns))
    if lib_size_s is not None:
        sample_sets.append(set(lib_size_s.index))
    if covariates_df is not None:
        sample_sets.append(set(covariates_df.index))

    common_samples = sorted(set.intersection(*sample_sets))
    n_samples = len(common_samples)
    if n_samples == 0:
        raise ValueError(
            "No samples overlap across genotype_df, phenotype_df, and any provided "
            "hap/counts/lib_size/covariates DataFrames. Check that column/index names match.")
    if covariates_df is not None and covariates_df.isnull().any().any():
        raise ValueError("covariates_df contains NaN; impute or drop missing covariates before calling cis_nominal()")
    if lib_size_s is not None and (lib_size_s <= 0).any():
        raise ValueError("lib_size_s contains non-positive values; library sizes must be > 0")
    logger.write(f'  * {n_samples} samples (after intersection)')
    logger.write(f'  * {phenotype_df.shape[0]} phenotypes')
    logger.write(f'  * {variant_df.shape[0]} variants')
    logger.write(f'  * cis-window: +/-{window:,}')
    if maf_threshold > 0:
        logger.write(f'  * MAF threshold: {maf_threshold}')

    # Subset all DataFrames to aligned samples
    genotype_df = genotype_df[common_samples]
    phenotype_df = phenotype_df[common_samples]
    if hap1_df is not None:
        hap1_df = hap1_df[common_samples]
    if hap2_df is not None:
        hap2_df = hap2_df[common_samples]
    if counts1_df is not None:
        counts1_df = counts1_df[common_samples]
    if counts2_df is not None:
        counts2_df = counts2_df[common_samples]
    if lib_size_s is not None:
        lib_size_s = lib_size_s[common_samples]
    if covariates_df is not None:
        covariates_df = covariates_df.loc[common_samples]

    # ------------------------------------------------------------------
    # Precompute shared tensors
    # ------------------------------------------------------------------
    dtype = torch.float64

    if lib_size_s is not None:
        lib_size_t = torch.tensor(lib_size_s.values.astype(np.float64),
                                  dtype=dtype, device=device)
    else:
        lib_size_t = torch.ones(n_samples, dtype=dtype, device=device)

    if covariates_df is not None:
        # Build (samples x (1 + n_covariates)) with intercept in first column
        cov_vals = covariates_df.values.astype(np.float64)  # samples x covariates
        intercept = np.ones((n_samples, 1))
        covariates_t = torch.tensor(
            np.hstack([intercept, cov_vals]), dtype=dtype, device=device
        )
        logger.write(f'  * {cov_vals.shape[1]} covariates')
    else:
        covariates_t = None

    # ------------------------------------------------------------------
    # Chromosome loop
    # ------------------------------------------------------------------
    chroms_in_phenotypes = phenotype_pos_df['chrom'].unique()
    chroms_in_variants = set(variant_df['chrom'].unique())

    all_rows = []
    gene_count = 0
    total_genes = phenotype_pos_df.shape[0]

    start_time = time.time()

    for chrom in chroms_in_phenotypes:
        if chrom not in chroms_in_variants:
            logger.write(f'  Chromosome {chrom}: no variants, skipping')
            continue

        logger.write(f'  Chromosome {chrom}')

        # Variant positions on this chromosome
        chrom_var_mask = variant_df['chrom'] == chrom
        chrom_variant_df = variant_df[chrom_var_mask]
        chrom_var_pos = chrom_variant_df['pos'].values  # numpy array

        # Genes on this chromosome
        chrom_gene_mask = phenotype_pos_df['chrom'] == chrom
        chrom_genes = phenotype_pos_df[chrom_gene_mask]

        for gene_id, gene_row in chrom_genes.iterrows():
            gene_count += 1
            if gene_count % 10 == 0:
                logger.write(f'  * processed {gene_count}/{total_genes} genes')

            tss = int(gene_row['start'])
            lo = tss - window
            hi = tss + window

            # Find cis variants
            cis_mask = (chrom_var_pos >= lo) & (chrom_var_pos <= hi)
            if not cis_mask.any():
                continue

            cis_variant_ids = chrom_variant_df.index[cis_mask]
            cis_pos = chrom_var_pos[cis_mask]

            # Subset genotypes (variants x samples)
            geno_cis = genotype_df.loc[cis_variant_ids].values.astype(np.float64)

            # MAF filter (denominator = 2 * non-missing samples per variant)
            if maf_threshold > 0:
                n_obs = np.sum(~np.isnan(geno_cis), axis=1)
                af_arr = np.nansum(geno_cis, axis=1) / (2.0 * n_obs)
                maf_arr = np.where(af_arr > 0.5, 1.0 - af_arr, af_arr)
                maf_pass = maf_arr >= maf_threshold
                if not maf_pass.any():
                    continue
                geno_cis = geno_cis[maf_pass]
                cis_variant_ids = cis_variant_ids[maf_pass]
                cis_pos = cis_pos[maf_pass]

            n_cis = geno_cis.shape[0]

            # Haplotype genotypes
            if hap1_df is not None:
                h1 = hap1_df.loc[cis_variant_ids].values.astype(np.float64)
            else:
                h1 = geno_cis / 2.0

            if hap2_df is not None:
                h2 = hap2_df.loc[cis_variant_ids].values.astype(np.float64)
            else:
                h2 = geno_cis / 2.0

            # Allele-specific counts for this gene
            if counts1_df is not None and gene_id in counts1_df.index:
                c1 = counts1_df.loc[gene_id].values.astype(np.float64)
            else:
                c1 = np.zeros(n_samples, dtype=np.float64)

            if counts2_df is not None and gene_id in counts2_df.index:
                c2 = counts2_df.loc[gene_id].values.astype(np.float64)
            else:
                c2 = np.zeros(n_samples, dtype=np.float64)

            # Total read counts for this gene
            y_total = phenotype_df.loc[gene_id].values.astype(np.float64)

            # Convert to tensors
            g1_t = torch.tensor(h1, dtype=dtype, device=device)
            g2_t = torch.tensor(h2, dtype=dtype, device=device)
            c1_t = torch.tensor(c1, dtype=dtype, device=device)
            c2_t = torch.tensor(c2, dtype=dtype, device=device)
            yt_t = torch.tensor(y_total, dtype=dtype, device=device)

            # Run mixQTL (suppress per-gene logging)
            res = _mixqtl(
                g1_t, g2_t, c1_t, c2_t, yt_t,
                lib_size_t=lib_size_t,
                covariates_t=covariates_t,
                trc_cutoff=trc_cutoff,
                asc_cutoff=asc_cutoff,
                weight_cap=weight_cap,
                asc_cap=asc_cap,
                n_cutoff=n_cutoff,
                verbose=False,
            )

            meta_b = res['meta_b'].cpu().numpy()
            meta_se = res['meta_se'].cpu().numpy()
            meta_p = res['meta_p'].cpu().numpy()
            meta_method = res['meta_method']
            trc_b = res['trc_b'].cpu().numpy()
            trc_se = res['trc_se'].cpu().numpy()
            trc_n = res['trc_n']
            asc_b = res['asc_b'].cpu().numpy()
            asc_se = res['asc_se'].cpu().numpy()
            asc_n = res['asc_n']

            # Skip gene if all results are NaN
            if np.all(np.isnan(meta_b)):
                continue

            # Allele stats for MAF-filtered cis variants
            geno_t = torch.tensor(geno_cis, dtype=dtype, device=device)
            af_t, ma_samples_t, ma_count_t = get_allele_stats(geno_t)
            af_arr_out = af_t.cpu().numpy()
            ma_samples_arr = ma_samples_t.cpu().numpy()
            ma_count_arr = ma_count_t.cpu().numpy()

            # TSS distance (signed: negative = upstream)
            tss_distance = (cis_pos - tss).astype(np.int32)

            gene_df = pd.DataFrame({
                'phenotype_id': gene_id,
                'variant_id': cis_variant_ids,
                'tss_distance': tss_distance,
                'af': af_arr_out.astype(np.float32),
                'ma_samples': ma_samples_arr.astype(np.int32),
                'ma_count': ma_count_arr.astype(np.int32),
                'pval_nominal': meta_p,
                'slope': meta_b.astype(np.float32),
                'slope_se': meta_se.astype(np.float32),
                'trc_b': trc_b.astype(np.float32),
                'trc_se': trc_se.astype(np.float32),
                'trc_n': trc_n,
                'asc_b': asc_b.astype(np.float32),
                'asc_se': asc_se.astype(np.float32),
                'asc_n': asc_n,
                'meta_method': meta_method,
            })
            all_rows.append(gene_df)

    logger.write(f'  * processed {gene_count}/{total_genes} genes (complete)')
    logger.write(f'  Time elapsed: {(time.time() - start_time) / 60:.2f} min')

    if all_rows:
        result_df = pd.concat(all_rows, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=[
            'phenotype_id', 'variant_id', 'tss_distance', 'af', 'ma_samples',
            'ma_count', 'pval_nominal', 'slope', 'slope_se',
            'trc_b', 'trc_se', 'trc_n', 'asc_b', 'asc_se', 'asc_n', 'meta_method',
        ])

    if prefix:
        out_path = f'{prefix}.cis_qtl_pairs.txt.gz'
        result_df.to_csv(out_path, sep='\t', index=False)
        logger.write(f'  * written to {out_path}')

    logger.write('done.')
    return result_df
