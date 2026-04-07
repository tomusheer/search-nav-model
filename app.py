import streamlit as st
from predict_knn import predict

st.set_page_config(page_title="Search Debug App", layout="centered")

st.title("Search Debug App")

if "query" not in st.session_state:
    st.session_state["query"] = ""

query = st.text_input("Query", key="query")

if st.button("Run prediction"):
    recs, score, mode = predict(query)

    st.subheader("Debug")
    st.json({
        "query": query,
        "mode": mode,
        "score": float(score) if score is not None else None,
        "recs": recs
    })

    st.subheader("Result")
    st.write(f"Source: {mode}")
    st.write(f"Confidence: {score:.2f}" if score is not None else "Confidence: N/A")

    if recs:
        for i, rec in enumerate(recs, 1):
            st.write(f"{i}. {rec}")
    else:
        st.warning("No recommendations returned")