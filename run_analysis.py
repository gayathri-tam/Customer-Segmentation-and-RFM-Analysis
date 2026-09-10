"""End-to-end, reproducible Customer Segmentation and RFM analysis.

Run from repository root: python src/run_analysis.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from customer_segmentation import cluster_diagnostics, choose_k, prepare_features, run_clustering
from data_cleaning import clean_transactions, profile_quality
from rfm_analysis import build_rfm

RAW_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


def download_source() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "online_retail.zip"
    xlsx = RAW_DIR / "Online Retail.xlsx"
    if not xlsx.exists():
        if not archive.exists():
            print("Downloading UCI Online Retail source data...")
            urllib.request.urlretrieve(RAW_URL, archive)
        import zipfile
        with zipfile.ZipFile(archive) as zf:
            source = next(name for name in zf.namelist() if name.lower().endswith(".xlsx"))
            with zf.open(source) as src, open(xlsx, "wb") as dst:
                dst.write(src.read())
    return xlsx


def save_plot(fig, filename: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    source = download_source()
    raw = pd.read_excel(source)
    quality = profile_quality(raw)
    sales = clean_transactions(raw)
    quality["rows_after_cleaning"] = int(len(sales))
    quality["records_removed"] = int(len(raw) - len(sales))

    rfm, reference_date = build_rfm(sales)
    _, scaled = prepare_features(rfm)
    diagnostics = cluster_diagnostics(scaled)
    final_k = choose_k(diagnostics)
    customers, cluster_summary, explained_variance, _ = run_clustering(rfm, final_k)

    # Dimensional model outputs for Power BI.
    fact = sales[["InvoiceNo", "StockCode", "CustomerID", "InvoiceDate", "InvoiceDateOnly", "Quantity", "UnitPrice", "Revenue", "Country"]].copy()
    fact.rename(columns={"InvoiceNo": "OrderID", "StockCode": "ProductID", "CustomerID": "CustomerKey", "InvoiceDateOnly": "Date"}, inplace=True)
    dim_customer = customers.merge(
        sales.groupby("CustomerID", as_index=False)["Country"].agg(lambda x: x.mode().iat[0]),
        left_on="CustomerID", right_on="CustomerID", how="left",
    ).rename(columns={"CustomerID": "CustomerKey", "Country": "PrimaryCountry"})
    dim_product = sales.groupby(["StockCode", "Description"], as_index=False).agg(
        UnitsSold=("Quantity", "sum"), ProductRevenue=("Revenue", "sum")
    ).rename(columns={"StockCode": "ProductID", "Description": "ProductDescription"})
    dates = pd.DataFrame({"Date": pd.date_range(fact["Date"].min(), fact["Date"].max(), freq="D")})
    dates["Year"] = dates["Date"].dt.year
    dates["MonthNumber"] = dates["Date"].dt.month
    dates["Month"] = dates["Date"].dt.month_name()
    dates["YearMonth"] = dates["Date"].dt.to_period("M").astype(str)
    geography = pd.DataFrame({"Country": sorted(fact["Country"].unique())})

    fact.to_csv(PROCESSED / "fact_sales.csv", index=False)
    dim_customer.to_csv(PROCESSED / "dim_customer.csv", index=False)
    dim_product.to_csv(PROCESSED / "dim_product.csv", index=False)
    dates.to_csv(PROCESSED / "dim_date.csv", index=False)
    geography.to_csv(PROCESSED / "dim_geography.csv", index=False)
    customers.to_csv(RESULTS / "rfm_summary.csv", index=False)
    cluster_summary.to_csv(RESULTS / "cluster_summary.csv", index=False)
    diagnostics.to_csv(RESULTS / "kmeans_diagnostics.csv", index=False)

    # Reproducible headline metrics and business cuts.
    orders = fact["OrderID"].nunique()
    metrics = {
        **quality,
        "source_columns": int(raw.shape[1]),
        "analysis_start_date": str(fact["InvoiceDate"].min()),
        "analysis_max_transaction_date": str(fact["InvoiceDate"].max()),
        "rfm_reference_date": str(reference_date.date()),
        "total_revenue_gbp": float(fact["Revenue"].sum()),
        "total_orders": int(orders),
        "total_customers": int(fact["CustomerKey"].nunique()),
        "total_products": int(fact["ProductID"].nunique()),
        "total_quantity": int(fact["Quantity"].sum()),
        "average_order_value_gbp": float(fact.groupby("OrderID")["Revenue"].sum().mean()),
        "average_orders_per_customer": float(orders / fact["CustomerKey"].nunique()),
        "average_revenue_per_customer_gbp": float(fact.groupby("CustomerKey")["Revenue"].sum().mean()),
        "final_k": int(final_k),
        "pca_explained_variance_pc1": float(explained_variance[0]),
        "pca_explained_variance_pc2": float(explained_variance[1]),
        "pca_explained_variance_total": float(explained_variance.sum()),
    }
    (RESULTS / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    monthly = fact.groupby(fact["InvoiceDate"].dt.to_period("M")).agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "nunique"), Customers=("CustomerKey", "nunique")).reset_index()
    monthly["InvoiceDate"] = monthly["InvoiceDate"].astype(str)
    monthly.to_csv(RESULTS / "monthly_performance.csv", index=False)
    top_products = sales.groupby(["StockCode", "Description"], as_index=False).agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"), Orders=("InvoiceNo", "nunique")).sort_values("Revenue", ascending=False)
    top_products.head(20).to_csv(RESULTS / "top_products.csv", index=False)
    country = fact.groupby("Country", as_index=False).agg(Revenue=("Revenue", "sum"), Customers=("CustomerKey", "nunique"), Orders=("OrderID", "nunique")).sort_values("Revenue", ascending=False)
    country.to_csv(RESULTS / "country_summary.csv", index=False)

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["InvoiceDate"], monthly["Revenue"], marker="o", color="#1f77b4")
    ax.set(title="Monthly Revenue", xlabel="Month", ylabel="Revenue (GBP)")
    ax.tick_params(axis="x", rotation=45)
    save_plot(fig, "monthly_revenue.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=customers, x="PCA1", y="PCA2", hue="Segment", palette="tab10", s=24, alpha=.75, ax=ax)
    ax.set(title=f"PCA View of K-Means Clusters (K={final_k})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Segment")
    save_plot(fig, "pca_visualization.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=customers, x="TSNE1", y="TSNE2", hue="Segment", palette="tab10", s=24, alpha=.75, ax=ax)
    ax.set(title=f"t-SNE View of K-Means Clusters (K={final_k})")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Segment")
    save_plot(fig, "tsne_visualization.png")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(diagnostics["k"], diagnostics["inertia"], marker="o", label="Inertia", color="#1f77b4")
    ax1.set(xlabel="Number of clusters (K)", ylabel="Inertia")
    ax2 = ax1.twinx()
    ax2.plot(diagnostics["k"], diagnostics["silhouette_score"], marker="s", label="Silhouette", color="#ff7f0e")
    ax2.set_ylabel("Silhouette score")
    ax1.set_title("K-Means Diagnostics")
    save_plot(fig, "kmeans_diagnostics.png")

    print(json.dumps(metrics, indent=2))
    print(f"Completed successfully. Selected K={final_k} based on the highest silhouette score tested.")


if __name__ == "__main__":
    main()
