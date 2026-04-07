import streamlit as st
from predict_knn import predict

st.set_page_config(page_title="JoyBuy Search Navigation Demo", layout="centered")

MODE_LABELS = {
    "exact": "In-house ML recommendation",
    "semantic": "In-house ML recommendation",
    "perplexity": "Gen AI recommendation",
    "related": "Related suggestions",
    "no_match": "No strong match found",
}

st.markdown("""
<style>
.reco-list {
    margin-top: 0.25rem;
    padding-left: 1.2rem;
}
.reco-list li {
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.small-label {
    color: #666;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

st.title("JoyBuy Search Navigation Demo")
st.caption("Try a search query to see recommended Search Tiles for your Keyword.")

sample_queries = [
    "lego", "coffee", "rice", "iphone",
    "air fryer", "pokemon", "nintendo switch", "washing machine"
]

cols = st.columns(4)
for i, q in enumerate(sample_queries):
    if cols[i % 4].button(q):
        st.session_state["query"] = q

query = st.text_input("Search query", value=st.session_state.get("query", ""))

if st.button("Generate recommendations"):
    recs, score, mode = predict(query)
    st.write({"query": query, "mode": mode, "score": score, "recs": recs})

    mode_label = MODE_LABELS.get(mode, mode)

    st.markdown(f"**Source:** {mode_label}")
    st.markdown(f"**Confidence:** {score:.2f}")

    if recs:
        st.markdown('<div class="small-label">Recommended options</div>', unsafe_allow_html=True)
        items = "".join([f"<li>{r}</li>" for r in recs])
        st.markdown(f'<ul class="reco-list">{items}</ul>', unsafe_allow_html=True)
    else:
        st.warning("Couldn't find that exact item right now.")