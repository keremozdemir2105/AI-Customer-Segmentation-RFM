import os
import datetime as dt

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

N_CLUSTERS = 4
RANDOM_STATE = 42
HF_DATASET = "pavan-naik/Online-Retail-Cleaned"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ─── 1. Load Data ─────────────────────────────────────────────────────────────

def load_data():
    print(f"Fetching dataset from Hugging Face...")
    dataset = load_dataset(HF_DATASET)
    df = pd.DataFrame(dataset["train"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalSum"] = df["Quantity"] * df["UnitPrice"]
    print(f"Loaded {df.shape[0]:,} transactions, {df['CustomerID'].nunique():,} customers.")
    return df

# ─── 2. RFM ───────────────────────────────────────────────────────────────────

def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalSum", "sum"),
    )
    print("RFM aggregation complete.")
    return rfm

# ─── 3. Clustering ────────────────────────────────────────────────────────────

def cluster_rfm(rfm, n_clusters=N_CLUSTERS):
    rfm_scaled = StandardScaler().fit_transform(np.log1p(rfm))
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=RANDOM_STATE, n_init=10)
    rfm = rfm.copy()
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    stats = rfm.groupby("Cluster").agg(
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
    ).round(1)
    print("\nCluster summary:")
    print(stats.to_string())
    return rfm, stats

# ─── 4. AI Personas ───────────────────────────────────────────────────────────

def generate_persona(cluster_id, stats):
    prompt = f"""
You are a Strategic Marketing Expert.
Analyze this customer segment:
- Average days since last purchase: {stats['Recency']:.0f}
- Average total orders: {stats['Frequency']:.1f}
- Average lifetime spend: ${stats['Monetary']:.2f}

Provide:
1. A creative Name for this persona.
2. A brief 2-sentence description of their shopping psychology.
3. One actionable marketing tip to increase their value.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_all_personas(cluster_stats):
    print("\nGenerating AI personas...\n")
    for cluster_id, stats in cluster_stats.iterrows():
        content = generate_persona(cluster_id, stats)
        if content:
            print(f"{'─'*50}\n  CLUSTER {cluster_id}\n{'─'*50}")
            print(content, "\n")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    rfm = build_rfm(df)
    rfm, cluster_stats = cluster_rfm(rfm)
    generate_all_personas(cluster_stats)
    rfm.to_csv("customer_segments.csv")
    print("Saved to customer_segments.csv")
