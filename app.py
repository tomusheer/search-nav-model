import streamlit as st
from predict_knn import predict

st.set_page_config(page_title="JoyBuy Search Navigation Demo", page_icon="🛍️", layout="centered")

MODE_LABELS = {
    "exact": "In-house ML recommendation",
    "semantic": "In-house ML recommendation",
    "perplexity": "Gen AI recommendation",
    "related": "Related suggestions",
    "no_match": "No strong match found",
    "assortment_lacking": "No strong match found",
}

def set_query(value):
    st.session_state["query"] = value

if "query" not in st.session_state:
    st.session_state["query"] = ""

st.markdown("""
<style>
.reco-list {
    margin-top: 0.25rem;
    padding-left: 1.2rem;
}
.reco-list li {
    margin-bottom: 0.2rem;
    line-height: 1.3;
}
.small-label {
    color: #666;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

st.title("JoyBuy Search Navigation Demo")
st.caption("Try a customer search query to see recommended Search Tiles.")

sample_queries = [
    "lego", "coffee", "rice", "iphone",
    "air fryer", "pokemon", "nintendo switch", "washing machine"
]

cols = st.columns(4)
for i, q in enumerate(sample_queries):
    cols[i % 4].button(q, on_click=set_query, args=(q,))

query = st.text_input("Search query", key="query")

if st.button("Generate recommendations", type="primary"):
    if query.strip():
        recs, score, mode = predict(query)
        mode_label = MODE_LABELS.get(mode, mode)

        st.markdown(f"**Source:** {mode_label}")

        if score is not None:
            st.markdown(f"**Confidence:** {score:.2f}")

        if recs:
            st.markdown('<div class="small-label">Recommended options</div>', unsafe_allow_html=True)
            items = "".join([f"<li>{r}</li>" for r in recs])
            st.markdown(f'<ul class="reco-list">{items}</ul>', unsafe_allow_html=True)
        else:
            st.info("We couldn’t find a strong match for that query. Try a slightly different search term.")
    else:
        st.info("Please enter a search query.")