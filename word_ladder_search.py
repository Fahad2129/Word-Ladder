"""
Word Ladder Search in Semantic Embedding Space
Assignment #1 - Introduction to Artificial Intelligence
IBA Karachi, Spring 2026
"""

import numpy as np
import heapq
import time
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────
# 1. EMBEDDING LOADER
# ─────────────────────────────────────────────

def load_embeddings(filepath: str) -> tuple[dict, np.ndarray, list]:
    """
    Load GloVe embeddings from a text file.
    Returns:
        word_to_idx : word -> index mapping
        matrix      : (N, 100) float32 array of unit-normalized vectors
        idx_to_word : index -> word list
    """
    words = []
    vectors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) != 101:          # word + 100 dims
                continue
            words.append(parts[0])
            vectors.append(np.array(parts[1:], dtype=np.float32))

    matrix = np.array(vectors, dtype=np.float32)
    # L2-normalise every row so cosine sim = dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    word_to_idx = {w: i for i, w in enumerate(words)}
    return word_to_idx, matrix, words


# ─────────────────────────────────────────────
# 2. SEARCH ENVIRONMENT
# ─────────────────────────────────────────────

class WordLadderEnv:
    """
    State         : a word (str)
    Action        : move to one of the k nearest neighbours in embedding space
    Path cost     : 1 - cosine_similarity(current, neighbour)   ∈ [0, 2]
    Heuristic     : 1 - cosine_similarity(current, goal)        ∈ [0, 2]

    Why k=20?
    ---------
    k=5  gives too sparse a graph; many pairs have no solution.
    k=50 inflates the branching factor and makes BFS/DFS impractical.
    k=20 balances connectivity with manageable expansion counts and was
    verified experimentally to solve most semantically related pairs.
    """

    K = 20        # number of neighbours (5 ≤ k ≤ 50)

    def __init__(self, word_to_idx: dict, matrix: np.ndarray, idx_to_word: list):
        self.word_to_idx = word_to_idx
        self.matrix = matrix
        self.idx_to_word = idx_to_word
        self._neighbour_cache: dict[str, list] = {}

    # ── similarity helpers ────────────────────

    def _vec(self, word: str) -> np.ndarray:
        return self.matrix[self.word_to_idx[word]]

    def cosine_similarity(self, a: str, b: str) -> float:
        """Dot product of pre-normalised vectors = cosine similarity."""
        return float(np.dot(self._vec(a), self._vec(b)))

    def edge_cost(self, a: str, b: str) -> float:
        """Cost of moving from a to b: lower similarity → higher cost."""
        return 1.0 - self.cosine_similarity(a, b)

    def heuristic(self, current: str, goal: str) -> float:
        """
        h(n) = 1 - cosine_similarity(current, goal)
        Intuition: if current is already semantically close to the goal
        (high similarity), the heuristic is small (nearly zero), meaning
        we are almost there.  Distant words yield h ≈ 1-2.

        Admissibility:
            The actual minimum path cost from n to goal is at least
            edge_cost(n, goal) = 1 - sim(n, goal) = h(n) only when goal
            is a direct neighbour of n.  In general h can overestimate
            when the direct cosine distance exceeds the true graph path
            cost, so cosine-based h is NOT guaranteed admissible.
            It is therefore used as an *informed* but non-admissible
            heuristic that still works well empirically.
        """
        return 1.0 - self.cosine_similarity(current, goal)

    # ── neighbours ───────────────────────────

    def neighbours(self, word: str) -> list[tuple[str, float]]:
        """
        Return the K nearest words (excluding self) with their edge costs.
        Results are cached to avoid repeated matrix operations.
        """
        if word in self._neighbour_cache:
            return self._neighbour_cache[word]

        vec = self._vec(word)
        sims = self.matrix @ vec          # shape (N,)
        # exclude self by setting its similarity to -inf
        sims[self.word_to_idx[word]] = -np.inf
        top_k_idx = np.argpartition(sims, -self.K)[-self.K:]
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]

        result = [
            (self.idx_to_word[i], 1.0 - float(sims[i]))
            for i in top_k_idx
        ]
        self._neighbour_cache[word] = result
        return result

    def is_valid(self, word: str) -> bool:
        return word in self.word_to_idx


# ─────────────────────────────────────────────
# 3. PATH RECONSTRUCTION
# ─────────────────────────────────────────────

def reconstruct_path(parent: dict, goal: str) -> list[str]:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ─────────────────────────────────────────────
# 4. SEARCH ALGORITHMS
# ─────────────────────────────────────────────

SearchResult = dict   # keys: path, nodes_expanded, time_taken, found


def bfs(env: WordLadderEnv, start: str, goal: str) -> SearchResult:
    """
    Breadth-First Search
    Explores level by level; guarantees shortest path in number of edges.
    """
    t0 = time.perf_counter()
    if start == goal:
        return {"path": [start], "nodes_expanded": 0,
                "time_taken": 0.0, "found": True}

    frontier = deque([start])
    parent = {start: None}
    nodes_expanded = 0

    while frontier:
        node = frontier.popleft()
        nodes_expanded += 1

        for nbr, _ in env.neighbours(node):
            if nbr not in parent:
                parent[nbr] = node
                if nbr == goal:
                    path = reconstruct_path(parent, goal)
                    return {"path": path,
                            "nodes_expanded": nodes_expanded,
                            "time_taken": time.perf_counter() - t0,
                            "found": True}
                frontier.append(nbr)

    return {"path": [], "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - t0, "found": False}


def dfs(env: WordLadderEnv, start: str, goal: str,
        depth_limit: int = 15) -> SearchResult:
    """
    Depth-First Search with depth limit to guarantee termination.
    depth_limit=15 keeps runtime manageable while allowing meaningful paths.
    """
    t0 = time.perf_counter()
    if start == goal:
        return {"path": [start], "nodes_expanded": 0,
                "time_taken": 0.0, "found": True}

    # Stack entries: (node, parent_node, depth)
    stack = [(start, None, 0)]
    parent = {}
    visited = set()
    nodes_expanded = 0

    while stack:
        node, par, depth = stack.pop()

        if node in visited:
            continue
        visited.add(node)
        parent[node] = par
        nodes_expanded += 1

        if node == goal:
            path = reconstruct_path(parent, goal)
            return {"path": path,
                    "nodes_expanded": nodes_expanded,
                    "time_taken": time.perf_counter() - t0,
                    "found": True}

        if depth < depth_limit:
            for nbr, _ in env.neighbours(node):
                if nbr not in visited:
                    stack.append((nbr, node, depth + 1))

    return {"path": [], "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - t0, "found": False}


def ucs(env: WordLadderEnv, start: str, goal: str) -> SearchResult:
    """
    Uniform Cost Search
    Expands the node with the lowest cumulative path cost (g-value).
    Uses edge_cost = 1 - cosine_similarity as the step cost.
    """
    t0 = time.perf_counter()
    if start == goal:
        return {"path": [start], "nodes_expanded": 0,
                "time_taken": 0.0, "found": True}

    # heap: (g_cost, node)
    heap = [(0.0, start)]
    g_cost = {start: 0.0}
    parent = {start: None}
    explored = set()
    nodes_expanded = 0

    while heap:
        g, node = heapq.heappop(heap)

        if node in explored:
            continue
        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            path = reconstruct_path(parent, goal)
            return {"path": path,
                    "nodes_expanded": nodes_expanded,
                    "time_taken": time.perf_counter() - t0,
                    "found": True}

        for nbr, cost in env.neighbours(node):
            new_g = g + cost
            if nbr not in g_cost or new_g < g_cost[nbr]:
                g_cost[nbr] = new_g
                parent[nbr] = node
                heapq.heappush(heap, (new_g, nbr))

    return {"path": [], "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - t0, "found": False}


def greedy(env: WordLadderEnv, start: str, goal: str) -> SearchResult:
    """
    Greedy Best-First Search
    Always expands the node with the smallest heuristic h(n).
    Fast but can find suboptimal paths and may get trapped in local minima.
    """
    t0 = time.perf_counter()
    if start == goal:
        return {"path": [start], "nodes_expanded": 0,
                "time_taken": 0.0, "found": True}

    heap = [(env.heuristic(start, goal), start)]
    parent = {start: None}
    explored = set()
    nodes_expanded = 0

    while heap:
        _, node = heapq.heappop(heap)

        if node in explored:
            continue
        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            path = reconstruct_path(parent, goal)
            return {"path": path,
                    "nodes_expanded": nodes_expanded,
                    "time_taken": time.perf_counter() - t0,
                    "found": True}

        for nbr, _ in env.neighbours(node):
            if nbr not in explored:
                h = env.heuristic(nbr, goal)
                if nbr not in parent:
                    parent[nbr] = node
                heapq.heappush(heap, (h, nbr))

    return {"path": [], "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - t0, "found": False}


def astar(env: WordLadderEnv, start: str, goal: str) -> SearchResult:
    """
    A* Search
    Expands the node with the smallest f(n) = g(n) + h(n).
    Combines actual cost with heuristic estimate.
    """
    t0 = time.perf_counter()
    if start == goal:
        return {"path": [start], "nodes_expanded": 0,
                "time_taken": 0.0, "found": True}

    h0 = env.heuristic(start, goal)
    heap = [(h0, 0.0, start)]        # (f, g, node)
    g_cost = {start: 0.0}
    parent = {start: None}
    explored = set()
    nodes_expanded = 0

    while heap:
        f, g, node = heapq.heappop(heap)

        if node in explored:
            continue
        explored.add(node)
        nodes_expanded += 1

        if node == goal:
            path = reconstruct_path(parent, goal)
            return {"path": path,
                    "nodes_expanded": nodes_expanded,
                    "time_taken": time.perf_counter() - t0,
                    "found": True}

        for nbr, cost in env.neighbours(node):
            new_g = g + cost
            if nbr not in g_cost or new_g < g_cost[nbr]:
                g_cost[nbr] = new_g
                parent[nbr] = node
                h = env.heuristic(nbr, goal)
                heapq.heappush(heap, (new_g + h, new_g, nbr))

    return {"path": [], "nodes_expanded": nodes_expanded,
            "time_taken": time.perf_counter() - t0, "found": False}


# ─────────────────────────────────────────────
# 5. DISPATCH
# ─────────────────────────────────────────────

ALGORITHMS = {
    "BFS": bfs,
    "DFS": dfs,
    "UCS": ucs,
    "Greedy": greedy,
    "A*": astar,
}


def run_search(env: WordLadderEnv, algo: str,
               start: str, goal: str) -> SearchResult:
    return ALGORITHMS[algo](env, start, goal)


# ─────────────────────────────────────────────
# 6. COMMAND-LINE DEMO (optional)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os

    glove_path = sys.argv[1] if len(sys.argv) > 1 else "glove_100d_20000.txt"
    print(f"Loading embeddings from {glove_path} …")
    w2i, mat, i2w = load_embeddings(glove_path)
    env = WordLadderEnv(w2i, mat, i2w)
    print(f"Loaded {len(i2w):,} words.\n")

    test_pairs = [
        ("network",  "hate"),      # easy / semantically connected
        ("leather",  "soar"),      # medium
        ("replace",  "shoves"),    # medium
        ("knave",    "brutes"),    # medium-hard
        ("whistler", "panah"),     # semantically distant (hard)
    ]

    header = f"{'Start':<10} {'Goal':<10} {'Algo':<8} {'Found':<6} " \
             f"{'Steps':<7} {'Expanded':<10} {'Time(s)':<10}"
    print(header)
    print("─" * len(header))

    for start, goal in test_pairs:
        for algo in ALGORITHMS:
            r = run_search(env, algo, start, goal)
            steps = len(r["path"]) - 1 if r["found"] else "-"
            print(f"{start:<10} {goal:<10} {algo:<8} {str(r['found']):<6} "
                  f"{str(steps):<7} {r['nodes_expanded']:<10} "
                  f"{r['time_taken']:.4f}")
        print()
