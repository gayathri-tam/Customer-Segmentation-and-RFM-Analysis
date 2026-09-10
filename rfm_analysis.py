"""RFM calculation and transparent quantile scoring."""
from __future__ import annotations

import pandas as pd


def _quintile_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Rank-based quintiles avoid qcut failures when many customers tie."""
    ranked = series.rank(method="first", pct=True)
    scores = (ranked * 5).apply(lambda x: min(5, max(1, int(__import__('math').ceil(x)))))
    return scores if higher_is_better else 6 - scores


def rfm_segment(row: pd.Series) -> str:
    """Business rule applied after data-derived 1–5 quantile scores."""
    r, f, m = row["RecencyScore"], row["FrequencyScore"], row["MonetaryScore"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    if r >= 3 and f >= 4:
        return "Loyal Customers"
    if r >= 4 and f >= 2:
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "New Customers"
    if r <= 2 and (f >= 3 or m >= 3):
        return "At Risk"
    if r == 1 and f <= 2:
        return "Lost Customers"
    return "Needs Attention"


def build_rfm(transactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    max_date = transactions["InvoiceDate"].max().normalize()
    reference_date = max_date + pd.Timedelta(days=1)
    rfm = (
        transactions.groupby("CustomerID", as_index=False)
        .agg(
            Recency=("InvoiceDateOnly", lambda x: int((reference_date - x.max()).days)),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
    )
    rfm["RecencyScore"] = _quintile_score(rfm["Recency"], higher_is_better=False)
    rfm["FrequencyScore"] = _quintile_score(rfm["Frequency"], higher_is_better=True)
    rfm["MonetaryScore"] = _quintile_score(rfm["Monetary"], higher_is_better=True)
    rfm["RFMScore"] = rfm[["RecencyScore", "FrequencyScore", "MonetaryScore"]].sum(axis=1)
    rfm["RFMSegment"] = rfm.apply(rfm_segment, axis=1)
    return rfm, reference_date
