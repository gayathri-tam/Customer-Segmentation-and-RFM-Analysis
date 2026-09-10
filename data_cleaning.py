"""Cleaning utilities for the UCI Online Retail transaction data."""
from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country",
]


def profile_quality(raw: pd.DataFrame) -> dict:
    """Return reproducible, pre-cleaning data-quality counts."""
    invoice = raw["InvoiceNo"].astype(str)
    return {
        "rows_before_cleaning": int(len(raw)),
        "missing_customer_id": int(raw["CustomerID"].isna().sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "cancelled_transaction_lines": int(invoice.str.startswith("C", na=False).sum()),
        "non_positive_quantity": int((raw["Quantity"] <= 0).sum()),
        "non_positive_unit_price": int((raw["UnitPrice"] <= 0).sum()),
        "invalid_invoice_dates": int(pd.to_datetime(raw["InvoiceDate"], errors="coerce").isna().sum()),
    }


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep attributable, positive, non-cancelled retail purchase lines."""
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")

    data = raw.copy()
    data["InvoiceNo"] = data["InvoiceNo"].astype(str).str.strip()
    data["StockCode"] = data["StockCode"].astype(str).str.strip()
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")
    data = data.drop_duplicates()
    data = data.dropna(subset=["CustomerID", "InvoiceDate", "Description"])
    data = data[~data["InvoiceNo"].str.startswith("C", na=False)]
    data = data[(data["Quantity"] > 0) & (data["UnitPrice"] > 0)].copy()
    data["CustomerID"] = data["CustomerID"].astype("int64").astype(str)
    data["Revenue"] = data["Quantity"] * data["UnitPrice"]
    data["InvoiceDateOnly"] = data["InvoiceDate"].dt.normalize()
    data["YearMonth"] = data["InvoiceDate"].dt.to_period("M").astype(str)
    return data.sort_values(["InvoiceDate", "InvoiceNo"]).reset_index(drop=True)
