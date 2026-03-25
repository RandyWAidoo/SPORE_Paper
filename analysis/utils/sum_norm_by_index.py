import pandas as pd

def sum_norm_by_index(
    df: pd.DataFrame,
    metric: str,
    groupby_cols: list,
    aggregation_cols: list,
    lower_is_better: bool = False,
    eps: float = 1e-12,
):
    work = df.copy()
    work[metric] = work[metric].astype(float)

    # Denominator: sum over the chosen groups
    denom = work.groupby(groupby_cols)[metric].transform('sum').replace(0, eps)
    work['norm'] = work[metric] / denom

    # If lower is better, flip scores
    if lower_is_better:
        work['norm'] = 1 - work['norm']

    # Aggregate
    agg_sum = work.groupby(aggregation_cols)['norm'].sum().rename('sum_normalized')
    agg_mean = work.groupby(aggregation_cols)['norm'].mean().rename('mean_normalized')
    agg_scores = (
        pd.concat([agg_mean, agg_sum], axis=1)
        .sort_values('sum_normalized', ascending=False)
        .reset_index()
    )

    return work, agg_scores
