#!/usr/bin/env python3
import os
import re
import sys
import json
import unicodedata
import requests
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




# =========================
# CONFIG
# =========================
TRAINING_FILE = "data/training_data.csv"   # update if needed
SEMANTIC_THRESHOLD = 0.78
MAX_RECS = 4
DEBUG = False


# =========================
# LOAD TRAINING DATA
# =========================
import streamlit as st  #
@st.cache_data(ttl=300)  # refresh every 5 min
def load_training_data():
    return pd.read_csv(TRAINING_FILE, on_bad_lines="skip")

df = load_training_data()

df["Query"] = df["Query"].astype(str).str.strip()
df = df[df["Query"].notna() & (df["Query"] != "") & (df["Query"].str.lower() != "nan")]
df = df.drop_duplicates(subset=["Query"], keep="last").reset_index(drop=True)

rec_cols = [c for c in df.columns if "recommendation" in str(c).lower()]


# =========================
# NORMALIZATION
# =========================
ALIASES = {
    "tshirt": "t shirt",
    "tee shirt": "t shirt",
    "tee shirts": "t shirt",
    "earphone": "headphones",
    "earphones": "headphones",
    "earbuds": "headphones",
}

def normalize_query(text: str) -> str:
    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[-_/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        x = str(x).strip()
        key = x.lower()
        if x and key not in seen:
            seen.add(key)
            out.append(x)
    return out


def extract_recommendations_from_row(row):
    recs = [
        str(row[col]).strip()
        for col in rec_cols
        if pd.notna(row[col]) and str(row[col]).strip()
    ]
    return unique_keep_order(recs)[:MAX_RECS]


# =========================
# TRAINING DATA MATCHING
# =========================
df["Query_normalized"] = df["Query"].apply(normalize_query)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(2, 4),
    lowercase=False,
    preprocessor=normalize_query
)

word_vectorizer = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    lowercase=False,
    preprocessor=normalize_query
)

X_char = char_vectorizer.fit_transform(df["Query"])
X_word = word_vectorizer.fit_transform(df["Query"])


def get_semantic_match(query):
    q_char = char_vectorizer.transform([query])
    q_word = word_vectorizer.transform([query])

    sims_char = cosine_similarity(q_char, X_char)[0]
    sims_word = cosine_similarity(q_word, X_word)[0]
    sims = 0.55 * sims_char + 0.45 * sims_word

    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    return best_idx, best_score


# =========================
# PERPLEXITY FALLBACK
# =========================
def is_valid_recommendation(text):
    if not text:
        return False
    if len(text) > 40:
        return False
    if len(text.split()) > 5:
        return False
    if text.endswith("."):
        return False
    if any(ch in text for ch in ["{", "}", "[", "]", ":"]):
        return False
    return True

def detect_query_language_hint(query: str) -> str:
    import re
    q = str(query)

    if re.search(r"[\u4e00-\u9fff]", q):  # Chinese characters
        return "Chinese"
    if re.search(r"[äöüßÄÖÜ]", q):         # German umlauts
        return "German"
    if re.search(r"[àáâäãåèéêëìíîïòóôöøùúûüÿçñ]", q):  # French accented chars
        return "French"
    if re.search(r"[àáâäãåèéêëìíîïòóôöøùúûüÿ]", q):   # Dutch accented chars
        return "Dutch"
    return "English or same as user query"


def get_perplexity_suggestions(query):
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        return [], "no_api_key"

    language_hint = detect_query_language_hint(query)

    prompt = f"""
You are a JoyBuy shopping assistant.

User query: "{query}"
Preferred output language: {language_hint}

Task:
Infer what the shopper is likely looking for and generate up to 4 useful narrower shopping suggestions.

Rules:
- Return suggestions in the same language as the user query when possible
- Suggestions must be shopping/product/category labels only
- Each suggestion must be 1 to 4 words
- No explanation
- No numbering
- No full sentences
- Prefer mainstream e-commerce subtypes or shopping groupings
- Prefer product types over luxury brands unless the query explicitly names a brand
- Do not switch to a neighboring product type; for example lipstick should stay lipstick, not lip balm
- If the query is clearly nonsense or too ambiguous, return an empty list

Examples:
- "cap" -> ["Baseball Caps", "Beanies", "Snapback Caps", "Men's Hats"]
- "headphone" -> ["Wireless Headphones", "Gaming Headsets", "Bluetooth Headphones", "Earbuds"]
- "soup" -> ["Tomato Soup", "Chicken Soup", "Instant Soup", "Vegetable Soup"]
- "watches" -> ["Smart Watches", "Men's Watches", "Women's Watches", "Sports Watches"]
- "lip stick" -> ["Matte Lipstick", "Liquid Lipstick", "Long-Wear Lipstick", "Cream Lipstick"]
- "苹果" -> ["苹果手机", "苹果充电器", "苹果配件", "苹果耳机"]
- "kopfhörer" -> ["Bluetooth Kopfhörer", "Gaming Kopfhörer", "In-Ear Kopfhörer", "Kabellose Kopfhörer"]

Return JSON only:
{{
  "recommendations": ["item1", "item2", "item3", "item4"],
  "category": "short category label or empty string",
  "reason": "short reason"
}}
"""

    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "sonar-pro",
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"].strip()

        if DEBUG:
            print("RAW PERPLEXITY:", text)

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return [], "bad_json"

        payload = json.loads(match.group(0))
        recs = payload.get("recommendations", [])

        recs = [str(x).strip() for x in recs if str(x).strip()]
        recs = unique_keep_order(recs)
        recs = [r for r in recs if is_valid_recommendation(r)]
        recs = recs[:MAX_RECS]

        if not recs:
            return [], "empty"

        return recs, "ok"

    except Exception as e:
        print(f"Perplexity error: {e}")
        return [], "error"

# =========================
# MAIN PREDICT
# =========================
def predict(query):
    raw_query = query.strip()
    norm_query = normalize_query(raw_query)

    if len(norm_query) < 2:
        return [], 0.0, "assortment_lacking"

    # 1) exact training data
    exact = df[df["Query_normalized"] == norm_query]
    if not exact.empty:
        row = exact.iloc[0]
        recs = extract_recommendations_from_row(row)
        if recs:
            return recs, 1.0, "exact"

    # 2) semantic training data
    best_idx, best_score = get_semantic_match(raw_query)
    if best_score >= SEMANTIC_THRESHOLD:
        row = df.iloc[best_idx]
        recs = extract_recommendations_from_row(row)
        if recs:
            return recs, best_score, "semantic"

    # 3) Perplexity fallback
    pplx_recs, pplx_status = get_perplexity_suggestions(raw_query)
    if pplx_recs:
        # We use a soft score here because the result is generative
        return pplx_recs, max(best_score, 0.60), "perplexity"

    # 4) final fallback
    return [], best_score, "assortment_lacking"


# =========================
# CLI
# =========================
if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "lego"
    recs, score, mode = predict(query)

    print(f"Query: {query}")
    print(f"Mode: {mode}")
    print(f"Confidence: {score:.2f}")

    if recs:
        print("Recommendations:")
        for r in recs:
            print(f"- {r}")
    else:
        print("Lack assortment")