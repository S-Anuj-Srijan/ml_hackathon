"""
PathSnap Task C: road-graph connectivity, HMM/Viterbi route decoding,
and route-continuity evaluation.

Dataset convention:
- arc_id is the zero-based row index in <trip>.arcs
- each .arcs row is (from_node_id, to_node_id)
- candidate probabilities are supplied by Person B, not learned here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RoadGraph:
    """Directed road graph where each arc row index is its arc_id."""
    arcs: np.ndarray
    nodes: np.ndarray

    @classmethod
    def from_files(cls, nodes_path: str | Path, arcs_path: str | Path) -> "RoadGraph":
        nodes = np.loadtxt(nodes_path, dtype=float, ndmin=2)
        arcs = np.loadtxt(arcs_path, dtype=int, ndmin=2)
        if arcs.shape[1] != 2 or nodes.shape[1] != 2:
            raise ValueError("Expected .nodes with 2 columns and .arcs with 2 columns.")
        if arcs.size and (arcs.min() < 0 or arcs.max() >= len(nodes)):
            raise ValueError("An arc references a node outside the .nodes array.")
        return cls(arcs=arcs, nodes=nodes)

    @property
    def n_nodes(self) -> int:
        return int(len(self.nodes))

    @property
    def n_arcs(self) -> int:
        return int(len(self.arcs))

    def __post_init__(self):
        # Arc adjacency: i -> j iff arc i ends where arc j begins.
        outgoing: dict[int, list[int]] = {}
        for arc_id, (u, _v) in enumerate(self.arcs):
            outgoing.setdefault(int(u), []).append(arc_id)
        object.__setattr__(self, "_outgoing_by_start", outgoing)
        object.__setattr__(self, "_arc_end", self.arcs[:, 1].astype(int))
        object.__setattr__(self, "_arc_start", self.arcs[:, 0].astype(int))
        object.__setattr__(self, "_arc_adjacency", [
            tuple(outgoing.get(int(v), ())) for v in self.arcs[:, 1]
        ])
        object.__setattr__(self, "_hop_cache", {})

    def next_arcs(self, arc_id: int) -> tuple[int, ...]:
        self._check_arc(arc_id)
        return self._arc_adjacency[int(arc_id)]

    def is_directly_connected(self, arc_a: int, arc_b: int) -> bool:
        self._check_arc(arc_a)
        self._check_arc(arc_b)
        return int(self._arc_end[arc_a]) == int(self._arc_start[arc_b])

    def _check_arc(self, arc_id: int) -> None:
        if not 0 <= int(arc_id) < self.n_arcs:
            raise IndexError(f"arc_id {arc_id} outside [0, {self.n_arcs})")

    def arc_hops(self, arc_a: int, arc_b: int, max_hops: int = 8) -> int | None:
        """Fewest directed arc transitions from arc_a to arc_b; None if not found."""
        self._check_arc(arc_a)
        self._check_arc(arc_b)
        key = (int(arc_a), int(arc_b), int(max_hops))
        if key in self._hop_cache:
            return self._hop_cache[key]
        if arc_a == arc_b:
            self._hop_cache[key] = 0
            return 0
        queue = deque([(int(arc_a), 0)])
        visited = {int(arc_a)}
        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for nxt in self.next_arcs(current):
                if nxt == arc_b:
                    self._hop_cache[key] = depth + 1
                    return depth + 1
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))
        self._hop_cache[key] = None
        return None

    def connectivity_summary(self) -> dict:
        """Directed graph diagnostics. Weak components are computed without NetworkX."""
        undirected = [[] for _ in range(self.n_nodes)]
        for u, v in self.arcs:
            undirected[int(u)].append(int(v))
            undirected[int(v)].append(int(u))
        seen = set()
        component_sizes = []
        for start in range(self.n_nodes):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for v in undirected[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            component_sizes.append(size)
        indegree = np.bincount(self.arcs[:, 1], minlength=self.n_nodes)
        outdegree = np.bincount(self.arcs[:, 0], minlength=self.n_nodes)
        return {
            "nodes": self.n_nodes,
            "arcs": self.n_arcs,
            "isolated_nodes": int(np.sum((indegree + outdegree) == 0)),
            "weak_component_count": len(component_sizes),
            "largest_weak_component_nodes": max(component_sizes, default=0),
            "component_sizes_desc": sorted(component_sizes, reverse=True),
        }


def _normalise_candidates(
    candidates: Sequence[Sequence[tuple[int, float]]],
    n_arcs: int,
) -> list[list[tuple[int, float]]]:
    """Validate and normalise each observation's (arc_id, probability) candidates."""
    result = []
    for t, row in enumerate(candidates):
        cleaned = []
        for arc_id, probability in row:
            arc_id = int(arc_id)
            probability = float(probability)
            if not 0 <= arc_id < n_arcs:
                raise ValueError(f"Observation {t}: invalid arc_id {arc_id}")
            if not np.isfinite(probability) or probability < 0:
                continue
            cleaned.append((arc_id, probability))
        if not cleaned:
            # Empty candidate sets are retained and handled as sequence breaks.
            result.append([])
            continue
        total = sum(p for _, p in cleaned)
        if total <= 0:
            uniform = 1.0 / len(cleaned)
            result.append([(a, uniform) for a, _ in cleaned])
        else:
            result.append([(a, p / total) for a, p in cleaned])
    return result


def viterbi_decode(
    graph: RoadGraph,
    candidates: Sequence[Sequence[tuple[int, float]]],
    timestamps: Sequence[float] | None = None,
    *,
    max_transition_hops: int = 4,
    transition_penalty_per_extra_hop: float = 1.5,
    disconnected_penalty: float = 12.0,
    max_gap_seconds: float | None = 10.0,
    emission_epsilon: float = 1e-12,
) -> dict:
    """
    Decode the most likely arc per observation using a log-space HMM/Viterbi DP.

    candidates[t] = [(arc_id, classifier_probability), ...] from Person B.
    Directly adjacent arcs receive the strongest transition score. A short
    connected path (up to max_transition_hops) is allowed for skipped GPS
    samples. Unreachable transitions receive a finite penalty rather than
    crashing the decoder. Empty candidates and large timestamp gaps split
    the sequence into independent decoding chunks.

    Returns predicted_arc_ids (None for observations with no candidates),
    chunk scores, and diagnostic counts.
    """
    cand = _normalise_candidates(candidates, graph.n_arcs)
    n = len(cand)
    if timestamps is not None and len(timestamps) != n:
        raise ValueError("timestamps and candidates must have equal lengths.")

    # Identify independent contiguous chunks.
    chunks: list[tuple[int, int]] = []
    start = None
    for i in range(n):
        gap_boundary = False
        if i > 0 and timestamps is not None and max_gap_seconds is not None:
            dt = float(timestamps[i]) - float(timestamps[i - 1])
            gap_boundary = (dt <= 0 or dt > max_gap_seconds)
        if gap_boundary and start is not None:
            chunks.append((start, i))
            start = None
        if not cand[i]:
            if start is not None:
                chunks.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        chunks.append((start, n))

    predictions: list[int | None] = [None] * n
    total_log_score = 0.0
    disconnected_transitions = 0
    relaxed_transitions = 0

    for lo, hi in chunks:
        # Dynamic-programming score and backpointer arrays, one row per time.
        dp: list[np.ndarray] = []
        back: list[np.ndarray] = []
        first = np.log(np.maximum(
            np.array([p for _, p in cand[lo]], dtype=float), emission_epsilon
        ))
        dp.append(first)
        back.append(np.full(len(first), -1, dtype=int))

        for t in range(lo + 1, hi):
            prev_ids = [a for a, _ in cand[t - 1]]
            curr_ids = [a for a, _ in cand[t]]
            curr_probs = np.array([p for _, p in cand[t]], dtype=float)
            emission = np.log(np.maximum(curr_probs, emission_epsilon))
            scores = np.full((len(prev_ids), len(curr_ids)), -np.inf)

            for i, prev_arc in enumerate(prev_ids):
                for j, curr_arc in enumerate(curr_ids):
                    hops = graph.arc_hops(
                        prev_arc, curr_arc, max_hops=max_transition_hops
                    )
                    if hops is not None:
                        scores[i, j] = -transition_penalty_per_extra_hop * max(0, hops - 1)
                    else:
                        scores[i, j] = -disconnected_penalty
                        disconnected_transitions += 1
                        relaxed_transitions += 1

            combined = dp[-1][:, None] + scores
            best_prev = np.argmax(combined, axis=0)
            best_scores = combined[best_prev, np.arange(len(curr_ids))]
            dp.append(best_scores + emission)
            back.append(best_prev)

        last_state = int(np.argmax(dp[-1]))
        total_log_score += float(dp[-1][last_state])
        state_path = [last_state]
        for k in range(len(back) - 1, 0, -1):
            state_path.append(int(back[k][state_path[-1]]))
        state_path.reverse()
        for offset, state_idx in enumerate(state_path):
            predictions[lo + offset] = int(cand[lo + offset][state_idx][0])

    return {
        "predicted_arc_ids": predictions,
        "log_score": total_log_score,
        "decoded_chunks": len(chunks),
        "empty_candidate_observations": sum(not row for row in cand),
        "disconnected_candidate_transitions_considered": disconnected_transitions,
        "relaxed_transitions_used": relaxed_transitions,
    }


def route_continuity(
    graph: RoadGraph,
    arc_ids: Sequence[int | None],
    *,
    max_hops: int = 1,
) -> dict:
    """
    Evaluate transitions between consecutive non-null predictions.
    Direct continuity means one arc ends at the next arc's start node.
    Reachable continuity permits up to max_hops transitions in the road graph.
    """
    pairs = [
        (int(a), int(b))
        for a, b in zip(arc_ids[:-1], arc_ids[1:])
        if a is not None and b is not None
    ]
    if not pairs:
        return {
            "evaluated_transitions": 0,
            "directly_connected": 0,
            "direct_continuity_rate": float("nan"),
            "reachable_within_max_hops": 0,
            "reachable_continuity_rate": float("nan"),
            "max_hops": max_hops,
        }
    direct = sum(graph.is_directly_connected(a, b) for a, b in pairs)
    reachable = sum(
        graph.arc_hops(a, b, max_hops=max_hops) is not None for a, b in pairs
    )
    return {
        "evaluated_transitions": len(pairs),
        "directly_connected": int(direct),
        "direct_continuity_rate": direct / len(pairs),
        "reachable_within_max_hops": int(reachable),
        "reachable_continuity_rate": reachable / len(pairs),
        "max_hops": max_hops,
    }


def classifier_only(candidates: Sequence[Sequence[tuple[int, float]]]) -> list[int | None]:
    """Top-1 baseline: highest Person-B probability independently per observation."""
    output = []
    for row in candidates:
        if not row:
            output.append(None)
        else:
            output.append(int(max(row, key=lambda item: float(item[1]))[0]))
    return output



def evaluate_ablation(
    graph: RoadGraph,
    candidates: Sequence[Sequence[tuple[int, float]]],
    ground_truth_arc_ids: Sequence[int | None],
    timestamps: Sequence[float] | None = None,
    **decoder_kwargs,
) -> dict:
    """
    Compare classifier Top-1 against classifier + HMM/Viterbi.

    ground_truth_arc_ids must be observation-aligned labels. Do not pass the
    variable-length .route sequence unless the team has established and
    documented a valid alignment to GPS observations.
    """
    from sklearn.metrics import accuracy_score, f1_score

    if len(candidates) != len(ground_truth_arc_ids):
        raise ValueError("Candidates and observation-aligned labels must match.")
    baseline = classifier_only(candidates)
    decoded_result = viterbi_decode(
        graph, candidates, timestamps=timestamps, **decoder_kwargs
    )
    decoded = decoded_result["predicted_arc_ids"]

    valid = [
        i for i, truth in enumerate(ground_truth_arc_ids)
        if truth is not None and baseline[i] is not None and decoded[i] is not None
    ]
    y_true = [int(ground_truth_arc_ids[i]) for i in valid]
    y_base = [int(baseline[i]) for i in valid]
    y_dec = [int(decoded[i]) for i in valid]

    def scores(y_pred):
        if not y_true:
            return {"accuracy": float("nan"), "macro_f1": float("nan")}
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    misses = sum(
        truth is not None and all(int(arc) != int(truth) for arc, _ in row)
        for truth, row in zip(ground_truth_arc_ids, candidates)
    )
    n_truth = sum(truth is not None for truth in ground_truth_arc_ids)
    return {
        "n_observations": len(candidates),
        "n_scored_observations": len(valid),
        "candidate_miss_count": int(misses),
        "candidate_miss_rate": misses / n_truth if n_truth else float("nan"),
        "classifier_only": {
            **scores(y_base),
            "route_continuity": route_continuity(graph, baseline),
        },
        "classifier_plus_hmm": {
            **scores(y_dec),
            "route_continuity": route_continuity(graph, decoded),
        },
        "decoder_diagnostics": decoded_result,
    }
