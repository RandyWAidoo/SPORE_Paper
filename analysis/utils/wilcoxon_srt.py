import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.stats.multitest as smm
import typing as tp

def wilcoxon_srt(
    df, 
    alternative: tp.Literal['greater', 'less'] = "greater", 
    comparand="SPORE", metric="ARI", alpha=0.05, hb_correction=False, need_pivot=True, index="Dataset", test_set_column="Algorithm"    
):
    # If a pivot is needed, pivot to wide format: rows = index, columns = (values of the column of interest)
    pivot = (
        df.pivot(index=index, columns=test_set_column, values=metric) if need_pivot
        else df.copy()
    )

    # Identify methods other than `tested`
    methods = [m for m in pivot.columns if m != comparand]

    results = []
    for m in methods:
        paired = pivot[[comparand, m]].dropna()
        if len(paired) == 0:
            continue

        # Paired differences: `tested` - comparator
        diffs = paired[comparand] - paired[m]
        median_diff = diffs.median()

        res = wilcoxon(
            paired[comparand],
            paired[m],
            alternative=alternative,
            zero_method="wilcox"
        )

        results.append({
            "Method": m,
            "N_pairs": int(len(paired)),
            "Median_diff": float(median_diff),
            "Wilcoxon_stat": res.statistic,
            "p_raw": res.pvalue
        })

    res_df = pd.DataFrame(results)

    # Holm-Bonferroni correction
    if not res_df.empty:
        if hb_correction:
            reject, pvals_corrected, _, _ = smm.multipletests(
                res_df["p_raw"], method="holm", alpha=alpha,
            )
            res_df["p_corrected"] = pvals_corrected
            res_df["Significant"] = reject
        else:
            res_df["Significant"] = (res_df["p_raw"] < alpha)

    return res_df
