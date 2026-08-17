"""DataFrame validation helpers shared across dashboard features."""

import pandas as pd


def ensure_unique_monthly_rows(
    df: pd.DataFrame,
    key_cols: list[str],
    value_cols: list[str],
    context: str,
) -> pd.DataFrame:
    """
    Drop exact duplicate rows for a monthly merge key, but fail on conflicting duplicates.

    This prevents many-to-many joins from silently duplicating portfolio weights when
    the source table contains multiple rows for the same logical stock-month key.
    """
    if df.empty:
        return df

    cols = [c for c in key_cols + value_cols if c in df.columns]
    work = df.loc[:, cols].drop_duplicates().copy()
    dup_mask = work.duplicated(subset=key_cols, keep=False)
    if dup_mask.any():
        sample = work.loc[dup_mask, cols].sort_values(key_cols).head(12)
        raise ValueError(
            f"{context}: conflicting duplicate rows found for {key_cols}. "
            "This would create a many-to-many merge and distort valuations.\n"
            f"Sample:\n{sample.to_string(index=False)}"
        )

    return df.drop_duplicates(subset=key_cols, keep="last").copy()
