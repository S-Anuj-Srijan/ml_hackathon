"""Directed road-network HMM, log-space Viterbi, and explicit sequence breaks."""
from functools import lru_cache
import heapq
import numpy as np

class ViterbiDecoder:
    def __init__(self, beta_m=40., gap_s=10., max_distance_m=2000.):
        if min(beta_m, gap_s, max_distance_m) <= 0:
            raise ValueError('Decoder parameters must be positive')
        self.beta_m, self.gap_s, self.max_distance_m = beta_m, gap_s, max_distance_m

    def decode(self, network, track_xy, timestamps, candidates, probabilities):
        result = np.full(len(timestamps), -1, dtype=int)
        independent = result.copy()
        graph = [[] for _ in network.nodes]
        for (u, v), length in zip(network.arcs, network.length):
            graph[u].append((int(v), float(length)))
        @lru_cache(maxsize=4096)
        def distances(source):
            found, queue = {source: 0.}, [(0., source)]
            while queue:
                cost, u = heapq.heappop(queue)
                if cost != found[u]: continue
                for v, length in graph[u]:
                    nxt = cost + length
                    if nxt <= self.max_distance_m and nxt < found.get(v, np.inf):
                        found[v] = nxt
                        heapq.heappush(queue, (nxt, v))
            return found
        candidates = candidates.reset_index(drop=True)
        grouped = {int(i): g.index.to_numpy() for i, g in candidates.groupby('obs_id', sort=True)}
        frames, backs, scores, previous, breaks = [], [], None, None, []
        def finish():
            if not frames: return
            state = int(np.argmax(scores))
            for j in range(len(frames)-1, -1, -1):
                obs, ids = frames[j]
                result[obs] = ids[state]
                if j: state = int(backs[j][state])
        for obs in range(len(timestamps)):
            indices = grouped.get(obs)
            if indices is None:
                finish(); frames, backs, scores, previous = [], [], None, None
                breaks.append({'obs_id': obs, 'reason': 'no_candidates'})
                continue
            frame = candidates.loc[indices]
            ids, offsets = frame.arc_id.to_numpy(dtype=int), frame.offset_m.to_numpy()
            p = np.clip(np.asarray(probabilities)[indices], 1e-12, 1)
            emission = np.log(p / p.sum())
            independent[obs] = ids[np.argmax(p)]
            reset = previous is None or timestamps[obs]-timestamps[obs-1] > self.gap_s
            if not reset:
                prev_ids, prev_offsets = previous
                transition = np.full((len(prev_ids), len(ids)), -np.inf)
                displacement = np.linalg.norm(track_xy[obs]-track_xy[obs-1])
                for a, (arc, offset) in enumerate(zip(prev_ids, prev_offsets)):
                    d = distances(int(network.arcs[arc, 1]))
                    road = network.length[arc]-offset + np.array([d.get(int(v), np.inf) for v in network.arcs[ids, 0]]) + offsets
                    same = (ids == arc) & (offsets >= offset-5)
                    road[same] = np.maximum(0., offsets[same]-offset)
                    valid = road <= self.max_distance_m
                    transition[a, valid] = -np.abs(road[valid]-displacement)/self.beta_m
                    if valid.any():
                        peak = transition[a, valid].max()
                        transition[a, valid] -= peak + np.log(np.exp(transition[a, valid]-peak).sum())
                choices = scores[:, None]+transition
                back = np.argmax(choices, axis=0)
                next_scores = choices[back, np.arange(len(ids))]+emission
                reset = not np.isfinite(next_scores).any()
                if reset: breaks.append({'obs_id': obs, 'reason': 'disconnected'})
            elif previous is not None:
                breaks.append({'obs_id': obs, 'reason': 'time_gap'})
            if reset:
                finish(); frames, backs = [], []
                scores, back = emission, np.zeros(len(ids), dtype=int)
            else: scores = next_scores
            scores = scores-scores.max()
            frames.append((obs, ids)); backs.append(back)
            previous = ids, offsets
        finish()
        return dict(point_arcs=result, classifier_arcs=independent, breaks=breaks)


def continuity(arcs, network):
    """Strict adjacent-observation continuity; missing points count as failures."""
    arcs = np.asarray(arcs)
    if len(arcs) < 2: return 1.
    a, b = arcs[:-1], arcs[1:]
    valid = (a >= 0) & (b >= 0)
    connected = np.zeros(len(a), dtype=bool)
    connected[valid] = (a[valid] == b[valid]) | (network.arcs[a[valid], 1] == network.arcs[b[valid], 0])
    return float(connected.mean())
