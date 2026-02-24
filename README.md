# 🔤 Word Ladder Search
Fahad Azfar 31659 AI Assignment #1 — IBA Karachi, Spring 2026

Search for semantic paths between words using glove word embeddings and classic AI search algorithms.

---

## What it does

Given two words, the app finds a path between them by jumping through semantically similar words in embedding space. You can compare how different search algorithms (BFS, DFS, UCS, Greedy, A*) perform on the same word pair.

---

## How to run

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/word-ladder.git
cd word-ladder
```

**2. Install dependencies**
```bash
pip install streamlit numpy
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit GUI |
| `word_ladder_search.py` | Search algorithms (BFS, DFS, UCS, Greedy, A*) |
| `glove_100d_20000.txt` | glove 100-d word embeddings (20,000 words) |

---

## Sample word pairs to try

| Start | Goal | Difficulty |
|-------|------|------------|
| network | hate | Easy |
| leather | soar | Medium |
| replace | shoves | Medium |
| knave | brutes | Hard |
| whistler | panah | Very Hard |

> Note: Common words like "king" or "dog" are not in this vocabulary. Use the pairs above or check the glove file for valid words.

---

## Algorithms

- **BFS** — shortest path by number of hops
- **DFS** — depth-limited to 15, memory efficient
- **UCS** — optimal by semantic edge cost
- **Greedy** — fast but can find suboptimal paths
- **A*** — best overall, combines cost and heuristic

---

## Built with

- Python
- NumPy
- Streamlit
- glove Embeddings (Stanford NLP)

