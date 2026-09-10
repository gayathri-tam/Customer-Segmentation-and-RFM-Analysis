# Customer Segmentation & RFM Analysis — Power BI Business Intelligence Dashboard

## Overview

This project turns retail transaction lines into customer-level RFM scores, K-Means customer segments, visual diagnostics, and Power BI-ready star-schema tables. It is designed for a data analyst portfolio: every metric below is produced by the reproducible Python pipeline, not hard-coded in this document.

## Business Problem

The retailer has thousands of customers but needs a practical way to identify high-value buyers, customers becoming inactive, frequent purchasers, and groups that require different retention or premium-offer strategies. The analysis answers:

1. Who are the most valuable and most active customers?
2. Which groups are at risk or dormant?
3. How much revenue does each group contribute?
4. Which customers merit retention versus premium campaigns?

## Dataset

Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), Chen (2015), CC BY 4.0. It records purchases for a UK-based non-store retailer from 1 December 2010 through 9 December 2011. The raw source has **541,909 rows and 8 columns**.

| Column | Meaning |
|---|---|
| `InvoiceNo` | Invoice/transaction ID; an ID beginning with `C` is a cancellation |
| `StockCode`, `Description` | Product identifier and product description |
| `Quantity` | Units on the transaction line |
| `InvoiceDate` | Transaction timestamp |
| `UnitPrice` | Price per unit in GBP |
| `CustomerID` | Customer identifier |
| `Country` | Customer country |

## Data Cleaning

The pipeline drops exact duplicate lines, records without a customer ID or valid timestamp/product description, cancellation invoices, and non-positive quantity or unit-price lines. This focuses the RFM analysis on attributable, completed purchase behavior.

| Quality check before cleaning | Actual count |
|---|---:|
| Missing customer IDs | 135,080 |
| Exact duplicate rows | 5,268 |
| Cancelled invoice lines | 9,288 |
| Non-positive quantity lines | 10,624 |
| Non-positive unit-price lines | 2,517 |
| Invalid dates | 0 |
| Rows after cleaning | 392,692 |
| Rows removed overall | 149,217 |

Counts overlap, so their individual values do not sum to the overall removals.

## Exploratory Data Analysis

The cleaned data contains **£8,887,208.89** in line revenue across **18,532** orders, **4,338** customers, and **3,665** products. Average order value is **£479.56**; average orders per customer is **4.27** and average revenue per customer is **£2,048.69**.

The UK accounts for £7,285,024.64 in revenue and 3,920 customers. The top revenue product is `PAPER CRAFT , LITTLE BIRDIE` (£168,469.60), but it has one invoice, so it should be interpreted cautiously. `REGENCY CAKESTAND 3 TIER` is a more repeatable high-revenue product: £142,264.75 across 1,703 orders. Monthly revenue, country, and product extracts are saved in `results/`.

![Monthly revenue](results/monthly_revenue.png)

## RFM Analysis

RFM describes how recently a customer purchased (Recency), how many distinct invoices they placed (Frequency), and how much revenue they generated (Monetary). The maximum transaction timestamp is 2011-12-09 12:50; the fixed reference date is therefore **2011-12-10**.

```python
reference_date = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
rfm = transactions.groupby("CustomerID", as_index=False).agg(
    Recency=("InvoiceDateOnly", lambda x: int((reference_date - x.max()).days)),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
)
```

Each metric receives a rank-based 1–5 quintile score. Higher Frequency and Monetary rank higher; lower Recency ranks higher. This yields a comparable `RFMScore` from 3–15 while retaining each raw measure. `results/rfm_summary.csv` is the customer-level RFM table.

The rule-based RFM labels complement the unsupervised clusters. For example, 942 Champions generated £5,742,846.84 (64.6% of cleaned revenue), while 911 At Risk customers generated £1,162,209.82 (13.1%). The thresholds are transparent combinations of these data-derived quintile scores in `src/rfm_analysis.py`.

## K-Means Customer Segmentation

K-Means groups customers with similar RFM profiles. It uses Recency plus log-transformed `Frequency` and `Monetary`; `log1p` reduces their strong right skew, and `StandardScaler` prevents the largest-scale variable from dominating distance.

The pipeline evaluates K=2–8 using inertia (elbow context) and silhouette score. **K=3** was selected because it had the highest tested silhouette score (**0.416**; K=2 was 0.406). The full comparison is in `results/kmeans_diagnostics.csv`.

| Segment | Customers | Avg Recency | Avg Frequency | Avg Monetary | Revenue % |
|---|---:|---:|---:|---:|---:|
| High-Value Active | 1,320 | 30.60 days | 9.84 | £5,492.04 | 81.6% |
| High-Spend Watchlist | 2,037 | 55.25 days | 2.05 | £612.71 | 14.0% |
| Dormant Customers | 981 | 255.62 days | 1.39 | £397.17 | 4.4% |

The names are assigned after clustering from each cluster’s observed recency, frequency, and monetary profile; they do not influence K-Means.

## PCA and t-SNE

PCA reduces the three scaled clustering features to two linear components for a compact overview. PC1 and PC2 explain **72.1%** and **21.6%** of variance respectively (**93.6%** together).

![PCA clusters](results/pca_visualization.png)

t-SNE is a non-linear neighborhood-preserving visualization that can reveal local groupings. Unlike PCA, it is not a variance summary and should not be used as evidence that the clusters are statistically correct. It is visual context only; K selection is based on diagnostics.

![t-SNE clusters](results/tsne_visualization.png)

## Business Insights and Recommendations

| Segment | Behavior and value | Recommended action |
|---|---|---|
| High-Value Active | 30.6-day average recency, 9.84 orders, and 81.6% of revenue | Protect this base with VIP/loyalty benefits, early product access, and cross-sell offers. Avoid indiscriminate discounting. |
| High-Spend Watchlist | Largest customer group (47.0%) but only 14.0% of revenue, with about two orders | Use a second-purchase journey, replenishment prompts, and product recommendations to increase repeat rate. |
| Dormant Customers | 255.6-day average recency and only 4.4% of revenue | Use a cost-controlled win-back test. Suppress customers who do not re-engage after a defined campaign window. |

The priority is to retain High-Value Active customers, improve repeat purchase among the Watchlist, then test inexpensive reactivation for Dormant customers.

## Power BI Dashboard

Load the CSVs in `data/processed/` as a star schema:

```text
DimDate ─┐
DimCustomer ─┼──> FactSales <── DimProduct
DimGeography ─┘
```

`FactSales` is at transaction-line grain. The four dimensions have unique keys and one-to-many, single-direction relationships to the fact table. See [DAX measures](powerbi/DAX%20Measures.md) and the [four-page dashboard guide](powerbi/Dashboard%20Design.md). Build the `.pbix` in Power BI Desktop after importing the generated data; do not upload a PBIX that contains local credentials or source paths.

## Project Architecture

```text
UCI transaction data → cleaning → EDA / RFM → scaled K-Means → PCA / t-SNE
                  → Power BI fact + dimensions → dashboard and actions
```

## Repository Structure

```text
data/                 Source documentation; raw/ and processed/ are generated
notebooks/            Jupyter notebook companion
src/                  Reproducible cleaning, RFM, clustering, and runner code
results/              Generated summaries and visualizations
powerbi/              DAX and page specification for Power BI Desktop
```

## How to Run

```bash
python -m pip install -r requirements.txt
python src/run_analysis.py
```

The runner downloads the original UCI Excel source, creates `data/processed/fact_sales.csv` and the four dimension CSVs, plus all `results/` artifacts. In Google Colab, upload the repository or clone it, run the install cell, then open and run `notebooks/customer_segmentation_rfm.ipynb`.

## Technologies Used

Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter, Power BI, and DAX.

## Results

The central result is concentrated value: the 1,320 High-Value Active customers (30.4% of customers) generate 81.6% of cleaned revenue. This concentration makes a retention-first marketing strategy more defensible than broad, equal treatment of every customer.

## Resume Bullets

- Built a customer segmentation pipeline on 541,909 UCI retail transaction lines, cleaning 149,217 non-attributable, duplicate, cancelled, or invalid records and producing a Power BI-ready star schema.
- Applied RFM scoring and K-Means clustering to 4,338 customers; selected 3 clusters using silhouette analysis (0.416), identifying a 1,320-customer high-value segment responsible for 81.6% of £8.89M cleaned revenue.
- Created PCA and t-SNE cluster visualizations, DAX measures, and a four-page Power BI dashboard specification to support retention, repeat-purchase, and win-back strategies.

## Future Improvements

- Validate campaign uplift with A/B testing instead of assuming retention impact.
- Add margin, returns, acquisition channel, and product-category data.
- Refresh the model on rolling periods and monitor segment migration.
- Compare K-Means with Gaussian Mixtures or hierarchical clustering.

## Interview Story (2–3 minutes)

“I started with 541,909 UCI online retail transaction lines and cleaned them to 392,692 attributable, positive purchase lines. I created RFM measures at the customer level, then transformed the skewed Frequency and Monetary features, scaled them, and compared K-Means values from two to eight. K=3 had the best silhouette score at 0.416. The outcome showed that High-Value Active customers were only 30.4% of customers but generated 81.6% of revenue, so I recommended protecting that group with VIP retention while using repeat-purchase journeys for the large lower-frequency group and low-cost win-back tests for dormant customers. I published the cleaned fact table, customer segments, date, product, and geography dimensions for a Power BI dashboard with DAX measures and segment slicers.”

## GitHub Upload Guidance

Commit code, this README, generated CSV summaries, and PNGs if desired. Do **not** commit the downloaded raw data (`data/raw/`), large processed fact CSV (`data/processed/`), virtual environments, notebook checkpoints, or a `.pbix` containing local paths/credentials. The supplied `.gitignore` excludes these items.

## Project Completion Checklist

- [x] Public real dataset selected, downloaded, and cited
- [x] Actual data-quality profiling and cleaned transaction table
- [x] EDA metrics, product/country/month outputs, and visuals
- [x] RFM calculation, 1–5 scores, and business labels
- [x] K-Means diagnostics, selected K, cluster profiles, PCA, and t-SNE
- [x] Power BI-ready fact/dimension CSVs, DAX, and dashboard specification
- [ ] Build the PBIX in Power BI Desktop using the provided import/model/page instructions
