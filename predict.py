import warnings
warnings.simplefilter("ignore")

import urllib3
urllib3.disable_warnings()

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import sys
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("data/training_data.csv").fillna("")
df.columns = [c.strip() for c in df.columns]

query_col = "Query"
rec_cols = ["Recommendation 1", "Recommendation 2", "Recommendation 3", "Recommendation 4", "Recommendation 5"]

df[query_col] = df[query_col].astype(str).str.strip()
df = df[df[query_col] != ""]
df = df[df[query_col].str.lower() != "-100(unknown)"]

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus = df[query_col].tolist()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def predict_recommendations(user_query, threshold=0.55):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_idx = torch.argmax(scores).item()

    row = df.iloc[top_idx]
    score = float(scores[top_idx])

    if score < threshold:
        return ["No recommendations found"]

    recs = []
    for col in rec_cols:
        val = str(row.get(col, "")).strip()
        if val and val.lower() != "nan":
            recs.append(val)

    return recs[:4]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python predict.py "your query"')
        sys.exit(1)

    user_query = sys.argv[1].strip()
    recs = predict_recommendations(user_query)

    print(f"Query: {user_query}")
    print("Recommendations:")
    for rec in recs:
        print(f"- {rec}")