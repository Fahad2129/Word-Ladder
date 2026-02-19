"""
Word Ladder Search — Streamlit GUI
Run with: streamlit run app.py
"""

import streamlit as st
import os, time
from word_ladder_search import (
    load_embeddings, WordLadderEnv, run_search, ALGORITHMS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Word Ladder Search",
    page_icon="🔤",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}
.stApp {
    background: #0d0d14;
    color: #e8e6f0;
}
h1, h2, h3 {
    font-family: 'Sora', sans-serif;
    font-weight: 800;
    letter-spacing: -0.03em;
}
.metric-card {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #6b6b8a;
    margin-top: 4px;
}
.path-box {
    background: #16161f;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 20px 24px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    line-height: 2;
    color: #c4b5fd;
}
.word-chip {
    display: inline-block;
    background: #1e1e30;
    border: 1px solid #4c1d95;
    border-radius: 6px;
    padding: 2px 10px;
    margin: 3px;
    color: #ddd6fe;
}
.arrow {
    color: #6b6b8a;
    margin: 0 4px;
}
.success-badge {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.fail-badge {
    display: inline-block;
    background: #450a0a;
    color: #fca5a5;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.stSelectbox > div > div {
    background: #16161f !important;
    border-color: #2a2a3a !important;
    color: #e8e6f0 !important;
}
.stTextInput > div > div > input {
    background: #16161f !important;
    border-color: #2a2a3a !important;
    color: #e8e6f0 !important;
    font-family: 'Space Mono', monospace !important;
}
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Sora', sans-serif;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}
.stSlider > div > div > div > div {
    background: #7c3aed !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD EMBEDDINGS (cached)
# ─────────────────────────────────────────────
GLOVE_FILE = os.path.join(os.path.dirname(__file__), "glove_100d_20000.txt")

@st.cache_resource(show_spinner="Loading GloVe embeddings…")
def get_env(k: int):
    w2i, mat, i2w = load_embeddings(GLOVE_FILE)
    env = WordLadderEnv(w2i, mat, i2w)
    env.K = k
    return env


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 🔤 Word Ladder Search")
st.markdown(
    "<p style='color:#6b6b8a;font-size:0.95rem;margin-top:-12px;'>"
    "Semantic graph search using GloVe 100-d embeddings &nbsp;·&nbsp; "
    "Fahad Azfar · AI Assignment #1</p>",
    unsafe_allow_html=True
)
st.divider()


# ─────────────────────────────────────────────
# SIDEBAR / CONTROLS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Search Settings")

    start_word = st.text_input("Start word", value="network",
                               placeholder="e.g. network").strip().lower()
    goal_word  = st.text_input("Goal word",  value="hate",
                               placeholder="e.g. hate").strip().lower()

    algo = st.selectbox("Algorithm", list(ALGORITHMS.keys()),
                        index=list(ALGORITHMS.keys()).index("A*"))

    k = st.slider("Neighbours k", min_value=5, max_value=50,
                  value=20, step=1,
                  help="Number of nearest neighbours per node (5 ≤ k ≤ 50)")

    if algo == "DFS":
        depth_limit = st.slider("DFS depth limit", 5, 30, 15)
    else:
        depth_limit = 15

    run_btn = st.button("▶  Run Search", use_container_width=True)

    st.divider()
    st.markdown("""
**Algorithm guide**
- **BFS** – shortest hop count
- **DFS** – memory-efficient, may find long paths
- **UCS** – optimal by semantic cost
- **Greedy** – fast, uses only heuristic
- **A*** – combines cost + heuristic
""")


# ─────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────
env = get_env(k)

if run_btn:
    # Validate inputs
    err = False
    if not start_word:
        st.error("Please enter a start word.")
        err = True
    elif not env.is_valid(start_word):
        st.error(f"❌ '{start_word}' is not in the vocabulary ({len(env.idx_to_word):,} words).")
        err = True

    if not goal_word:
        st.error("Please enter a goal word.")
        err = True
    elif env.is_valid(goal_word) is False and not err:
        st.error(f"❌ '{goal_word}' is not in the vocabulary.")
        err = True

    if not err:
        with st.spinner(f"Running {algo}…"):
            # Patch DFS depth limit
            if algo == "DFS":
                from word_ladder_search import dfs as _dfs
                result = _dfs(env, start_word, goal_word, depth_limit=depth_limit)
            else:
                result = run_search(env, algo, start_word, goal_word)

        # ── Metrics row ─────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{'✓' if result['found'] else '✗'}</div>
                <div class="metric-label">Path Found</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            steps = len(result["path"]) - 1 if result["found"] else "—"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{steps}</div>
                <div class="metric-label">Steps</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['nodes_expanded']:,}</div>
                <div class="metric-label">Nodes Expanded</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['time_taken']:.3f}s</div>
                <div class="metric-label">Time Taken</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Path display ─────────────────────────
        if result["found"]:
            st.markdown(f'<span class="success-badge">✓ PATH FOUND</span>', unsafe_allow_html=True)
            chips = " <span class='arrow'>→</span> ".join(
                f"<span class='word-chip'>{w}</span>" for w in result["path"]
            )
            st.markdown(f'<div class="path-box">{chips}</div>', unsafe_allow_html=True)

            # Similarity breakdown
            if len(result["path"]) > 1:
                st.markdown("**Similarity breakdown along path:**")
                path = result["path"]
                rows = []
                total_cost = 0.0
                for i in range(len(path) - 1):
                    sim  = env.cosine_similarity(path[i], path[i+1])
                    cost = 1.0 - sim
                    total_cost += cost
                    rows.append({
                        "Step": f"{i+1}",
                        "From": path[i],
                        "To": path[i+1],
                        "Cosine Similarity": f"{sim:.4f}",
                        "Edge Cost": f"{cost:.4f}",
                    })
                import pandas as pd
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown(f"**Total path cost (UCS metric):** `{total_cost:.4f}`")
        else:
            st.markdown(f'<span class="fail-badge">✗ NO PATH FOUND</span>', unsafe_allow_html=True)
            st.info(
                f"No path found from **{start_word}** → **{goal_word}** "
                f"within the current constraints (k={k}, depth_limit={depth_limit if algo=='DFS' else 'N/A'}).\n\n"
                "Try increasing k, relaxing the depth limit, or choosing semantically closer words."
            )

else:
    # ── Welcome / Instructions ────────────────
    st.markdown("""
    <div style="background:#16161f;border:1px solid #2a2a3a;border-radius:14px;
                padding:32px 36px;max-width:700px;">
        <h3 style="margin-top:0;color:#c4b5fd;">How to use</h3>
        <ol style="color:#a1a1c0;line-height:2;font-size:0.95rem;">
            <li>Enter a <strong style='color:#ddd6fe;'>start word</strong> and <strong style='color:#ddd6fe;'>goal word</strong> in the sidebar.</li>
            <li>Choose a <strong style='color:#ddd6fe;'>search algorithm</strong>.</li>
            <li>Optionally tune the number of neighbours <em>k</em>.</li>
            <li>Click <strong style='color:#ddd6fe;'>▶ Run Search</strong>.</li>
        </ol>
        <hr style="border-color:#2a2a3a;margin:20px 0;">
        <p style="color:#6b6b8a;font-size:0.85rem;margin:0;">
            Built on GloVe 100-d embeddings · 20 000 word vocabulary<br>
            Heuristic: h(n) = 1 − cosine_similarity(n, goal)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sample pairs
    st.markdown("<br>**Sample word pairs to try:**", unsafe_allow_html=True)
    sample_data = {
        "Start": ["network",  "leather", "replace", "knave",    "whistler"],
        "Goal":  ["hate",     "soar",    "shoves",  "brutes",   "panah"],
        "Difficulty": ["Easy", "Medium", "Medium",  "Hard",     "Very Hard"],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True,
                 hide_index=True, column_config={"Difficulty": st.column_config.TextColumn()})
