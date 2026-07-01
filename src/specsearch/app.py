import re

import streamlit as st

from specsearch.pipeline import run

st.set_page_config(page_title="SpecSearch", page_icon="📋", layout="wide")
st.title("📋 SpecSearch")
st.caption("Hybrid RAG search over construction-spec PDFs — BM25 + vectors, RRF-fused, reranked.")

query = st.text_input(
    "Question",
    placeholder="What is the required concrete compressive strength at 28 days?",
)
col_btn, col_explain = st.columns([1, 3])
search = col_btn.button("Search", type="primary", use_container_width=True)
explain = col_explain.checkbox("Show stage details")

if search and query.strip():
    with st.spinner("Retrieving → fusing → reranking → generating …"):
        try:
            answer = run(query.strip(), explain=explain)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

    # --- Answer ---
    st.subheader("Answer")
    # Highlight citations as inline code so they stand out visually
    highlighted = re.sub(r"(\[[^\]]+p\.\d+\])", r"`\1`", answer.text)
    st.markdown(highlighted)

    if answer.citations:
        st.caption("**Citations:** " + " · ".join(f"`{c}`" for c in answer.citations))

    st.divider()

    # --- Source chunks ---
    if answer.rerank_scores:
        st.subheader(f"Sources — top {len(answer.rerank_scores)}")
        for score, chunk in answer.rerank_scores:
            label = f"[{score:+.3f}]  {chunk.source_file}  p.{chunk.page}  —  {chunk.section[:70]}"
            with st.expander(label):
                meta = f"**File:** {chunk.source_file}  |  **Page:** {chunk.page}"
                st.caption(meta + f"  |  **Section:** {chunk.section}")
                st.text(chunk.text)

    # --- Timings (explain mode) ---
    if explain and answer.timings:
        st.subheader("Stage Timings")
        cols = st.columns(len(answer.timings))
        for col, (stage, secs) in zip(cols, answer.timings.items()):
            col.metric(stage.replace("_", " "), f"{secs:.3f}s")

        if answer.candidates:
            st.caption(
                f"{len(answer.candidates)} candidates after RRF → "
                f"{len(answer.rerank_scores)} returned after rerank"
            )

elif search and not query.strip():
    st.warning("Enter a question first.")
