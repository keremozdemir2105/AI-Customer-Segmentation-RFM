import pandas as pd
import numpy as np
import datetime as dt
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from openai import OpenAI
from google.colab import userdata

client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

dataset = load_dataset("pavan-naik/Online-Retail-Cleaned")
df = pd.DataFrame(dataset['train'])

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalSum'] = df['Quantity'] * df['UnitPrice']
df.columns = df.columns.str.strip()

snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalSum': 'sum'
}).rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalSum': 'Monetary'
})

rfm_log = np.log1p(rfm)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_summary = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean'
}).round(1)

def generate_persona(row):
    prompt = f"""
    Analyze this customer segment:
    - Recency: {row.Recency:.0f} days
    - Frequency: {row.Frequency:.1f} orders
    - Monetary: ${row.Monetary:.2f}
    
    Provide:
    1. Persona Name.
    2. 2-sentence psychology description.
    3. 1 marketing tip.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return None

for row in cluster_summary.itertuples():
    persona = generate_persona(row)
    if persona:
        print(f"Cluster {int(row.Index)}:")
        print(persona, "\n")

rfm.to_csv('customer_segments.csv')
