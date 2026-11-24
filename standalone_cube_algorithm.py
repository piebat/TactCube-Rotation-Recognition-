"""
Standalone copy of the cube rotation recognition algorithm (3-finger grip:
pinch on a pair of opposite faces + index finger on a face of the same
ring), kept byte-for-byte in sync with cube_algorithm.py by hand.

This file exists ONLY so that validate_algorithm.py can exercise CubeTracker
without importing anything else from the project -- no Flask, no MQTT, no
session/combination/graph-export machinery, not even generate_graph.py. It
has zero imports beyond the standard library and numpy, so it keeps working
unchanged even if the rest of the server is refactored or removed. It is
deliberately NOT imported by app.py, mqtt_client.py or session_recorder.py:
those use cube_algorithm.py directly, so production behaviour is never
affected by this copy.

Geometry derived from the chirality datum provided: the vertex where faces
1,0,4 meet (in that order) is CLOCKWISE seen from outside.

Reference frame (fixed with respect to the HAND, not the cube): axis U is the
thumb-middle finger line (the pinch), axis V is the index finger direction,
axis W is the perpendicular direction (no finger touches it). Kept distinct
from the cube's x,y,z axes (FACE_VEC below), which are fixed to the cube's
body and not to the hand.

State observable from the sensors: (P, f)
  P = currently pinched axis ('A'=0-3, 'B'=1-2, 'C'=4-5)
  f = face touched by the index finger (one of the 4 faces not in P)
12 observable states in total.

Internal state tracked by the algorithm: one of the cube's 24 proper
orientations (integer 3x3 rotation matrix). It is needed because from the
observable state (P,f) alone, move prediction is ambiguous (there is a
"hidden" bit — which of the two pinched faces is at +x — that the sensors
cannot measure but which affects the outcome of spins/tips). By tracking a
specific orientation consistently over time (initialised arbitrarily; the
starting position does not matter) the ambiguity disappears: it has been
verified that every state has exactly 5 distinct elementary 90-degree moves:
  spin+  / spin-   : rotation about the pinch axis (U)        -> P unchanged, f changes
  tip_U+ / tip_U-   : rotation about the free axis (W)         -> both P and f change
  tip_H              : rotation about the index axis (V)       -> f unchanged, P changes
                        (the two directions +90/-90 produce the same observable outcome)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Cube geometry (derived from the chirality 0-1-4 = clockwise seen from outside)
# ---------------------------------------------------------------------------
FACE_VEC = {
    0: (1, 0, 0), 3: (-1, 0, 0),   # axis A
    1: (0, 1, 0), 2: (0, -1, 0),   # axis B
    4: (0, 0, 1), 5: (0, 0, -1),   # axis C
}
VEC_FACE = {v: k for k, v in FACE_VEC.items()}
AXIS_OF = {0: 'A', 3: 'A', 1: 'B', 2: 'B', 4: 'C', 5: 'C'}
PAIR = {0: 3, 3: 0, 1: 2, 2: 1, 4: 5, 5: 4}


def _rot_u(s):
    return np.array([[1, 0, 0], [0, 0, -s], [0, s, 0]])


def _rot_v(s):
    return np.array([[0, 0, s], [0, 1, 0], [-s, 0, 0]])


def _rot_w(s):
    return np.array([[0, -s, 0], [s, 0, 0], [0, 0, 1]])


# the 6 elementary rotations (about the hand axes u,v,w); v+ and v- collapse
# into the same observable outcome (verified), so they are grouped under a
# single 'tip_H' label.
_RAW_ROTATIONS = {
    'spin+': _rot_u(1), 'spin-': _rot_u(-1),
    'tip_H': _rot_v(1),          # v- would give the same observable result
    'tip_U+': _rot_w(1), 'tip_U-': _rot_w(-1),
}

MOVE_DESCRIPTIONS = {
    'spin+': "spin about the pinch axis (+ direction)",
    'spin-': "spin about the pinch axis (- direction)",
    'tip_H': "tip about the index axis (the index face does not change)",
    'tip_U+': "tip about the free axis (+ direction)",
    'tip_U-': "tip about the free axis (- direction)",
}


def _face_at(M, pos):
    body_vec = tuple(int(round(x)) for x in (M.T @ np.array(pos)))
    return VEC_FACE[body_vec]


def observe(M):
    """Given an orientation (3x3 matrix), returns the observable state (P, f)."""
    fx_plus = _face_at(M, (1, 0, 0))
    fx_minus = _face_at(M, (-1, 0, 0))
    assert PAIR[fx_plus] == fx_minus
    return AXIS_OF[fx_plus], _face_at(M, (0, 1, 0))


def _generate_all_orientations():
    seen = {}
    start = np.eye(3, dtype=int)
    seen[tuple(map(tuple, start))] = start
    changed = True
    while changed:
        changed = False
        for M in list(seen.values()):
            for R in _RAW_ROTATIONS.values():
                M2 = R @ M
                key = tuple(map(tuple, M2))
                if key not in seen:
                    seen[key] = M2
                    changed = True
    return list(seen.values())


# ---------------------------------------------------------------------------
# Precomputation (happens once only, when the module is imported)
# ---------------------------------------------------------------------------
_ORIENTATIONS = _generate_all_orientations()          # 24 matrices
_ORIENT_KEY = lambda M: tuple(map(tuple, M))
_ID_OF = {_ORIENT_KEY(M): i for i, M in enumerate(_ORIENTATIONS)}
_OBSERVE_OF = [observe(M) for M in _ORIENTATIONS]      # id -> (P,f)

# for each id, its 5 neighbours: move_label -> (neighbor_id, neighbor_state)
_NEIGHBORS = []
for M in _ORIENTATIONS:
    nb = {}
    for label, R in _RAW_ROTATIONS.items():
        M2 = R @ M
        nid = _ID_OF[_ORIENT_KEY(M2)]
        nb[label] = (nid, _OBSERVE_OF[nid])
    _NEIGHBORS.append(nb)

# maps observable state -> list of compatible ids (1 or 2 elements)
_STATES_FOR_OBS = {}
for i, s in enumerate(_OBSERVE_OF):
    _STATES_FOR_OBS.setdefault(s, []).append(i)

ALL_OBSERVABLE_STATES = sorted(_STATES_FOR_OBS.keys())


def _bfs_shortest(start_id):
    """BFS from one id to all the others: returns dict id -> (distance, first_step)."""
    dist = {start_id: 0}
    first_step = {start_id: None}
    q = deque([start_id])
    while q:
        cur = q.popleft()
        for label, (nid, _) in _NEIGHBORS[cur].items():
            if nid not in dist:
                dist[nid] = dist[cur] + 1
                first_step[nid] = label if cur == start_id else first_step[cur]
                q.append(nid)
    return dist, first_step


_ALL_PAIRS_DIST = [{} for _ in range(24)]
_ALL_PAIRS_FIRSTSTEP = [{} for _ in range(24)]
for i in range(24):
    d, fs = _bfs_shortest(i)
    _ALL_PAIRS_DIST[i] = d
    _ALL_PAIRS_FIRSTSTEP[i] = fs


def bitmask_to_faces(bitmask: int) -> set[int]:
    """bit i (LSB=0) set -> face i touched."""
    return {i for i in range(6) if (bitmask >> i) & 1}


def faces_to_state(touched: set[int]):
    """Validates 3 touched faces (opposite pair + 1) and returns (P, f), or None if invalid."""
    if len(touched) != 3:
        return None
    pairs_touched = [f for f in touched if PAIR[f] in touched]
    if len(pairs_touched) != 2:
        return None
    p_faces = set(pairs_touched)
    others = touched - p_faces
    if len(others) != 1:
        return None
    f = next(iter(others))
    p_face = next(iter(p_faces))
    return AXIS_OF[p_face], f


@dataclass
class Event:
    ok: bool
    reason: str = ""
    state: tuple | None = None
    move: str | None = None
    move_desc: str | None = None
    distance: int = 0
    ambiguous: bool = False
    alt_moves: list = field(default_factory=list)
    raw_faces: set = field(default_factory=set)


class CubeTracker:
    """Keeps the internal state (one of the 24 orientations) and classifies
    the transitions on every new valid reading."""

    def __init__(self):
        self._id = None  # id of the current internal orientation (0..23)

    @property
    def state(self):
        return None if self._id is None else _OBSERVE_OF[self._id]

    def reset(self):
        self._id = None

    def calibrate(self, expected_move: str, previous_state: tuple) -> bool:
        """Pins down the hidden bit using a move known from outside the sensors.

        The two orientations compatible with the same observable state (P,f)
        are mirror images: from the wrong one a rotation reads with inverted
        polarity (tip_U+ as tip_U-, spin+ as spin-; tip_H is self-symmetric
        and unaffected). The sensors cannot tell the twins apart, but the move
        the operator was *asked* to perform can: only one twin of
        ``previous_state`` reaches the currently observed state via
        ``expected_move``.

        Returns True if the bit could be pinned down, False if the move is
        uninformative (tip_H) or inconsistent with what was observed -- in
        which case the caller should leave the tracker alone.
        """
        s = self.state
        if s is None or previous_state is None:
            return False
        consistent = [pid for pid in _STATES_FOR_OBS.get(previous_state, [])
                      if _NEIGHBORS[pid][expected_move][1] == s]
        if len(consistent) != 1:
            return False
        self._id = _NEIGHBORS[consistent[0]][expected_move][0]
        return True

    def update(self, touched_faces: set[int]) -> Event:
        s_new = faces_to_state(touched_faces)
        if s_new is None:
            return Event(ok=False, reason="invalid reading (not pinch+1)", raw_faces=touched_faces)

        if self._id is None:
            # first reading: no reference -> the hidden orientation is chosen
            # arbitrarily (it does not matter, like the starting position)
            self._id = _STATES_FOR_OBS[s_new][0]
            return Event(ok=True, reason="initial state", state=s_new, raw_faces=touched_faces)

        # 1) try a direct match among the 5 elementary neighbours
        for label, (nid, nstate) in _NEIGHBORS[self._id].items():
            if nstate == s_new:
                self._id = nid
                return Event(ok=True, state=s_new, move=label,
                             move_desc=MOVE_DESCRIPTIONS[label], distance=1,
                             raw_faces=touched_faces)

        if s_new == _OBSERVE_OF[self._id]:
            return Event(ok=True, state=s_new, move=None, move_desc="no movement",
                         distance=0, raw_faces=touched_faces)

        # 2) more than one elementary rotation between two readings: precomputed BFS
        candidates = _STATES_FOR_OBS[s_new]
        best_dist = None
        best_targets = []
        for cid in candidates:
            d = _ALL_PAIRS_DIST[self._id].get(cid)
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_targets = [cid]
            elif d == best_dist:
                best_targets.append(cid)

        # all the initial moves compatible with a shortest path towards ANY
        # of the best_targets (not just the first one chosen arbitrarily):
        # this also captures the "same target, different first step"
        # ambiguity that comparing best_targets alone would miss.
        alt_moves = []
        for label, (nid, _) in _NEIGHBORS[self._id].items():
            for tgt in best_targets:
                d2 = _ALL_PAIRS_DIST[nid].get(tgt)
                if d2 is not None and 1 + d2 == best_dist:
                    alt_moves.append(label)
                    break
        alt_moves = sorted(set(alt_moves))

        ambiguous = len(best_targets) > 1 or len(alt_moves) > 1
        chosen = best_targets[0]
        first_move = alt_moves[0] if alt_moves else _ALL_PAIRS_FIRSTSTEP[self._id].get(chosen)
        self._id = chosen
        alt_desc = ", ".join(alt_moves) if len(alt_moves) > 1 else MOVE_DESCRIPTIONS.get(first_move, '?')
        return Event(ok=True, state=s_new, move=first_move,
                     move_desc=f"multiple jump ({best_dist} moves); first move: {alt_desc}"
                               + (" -- AMBIGUOUS, they cannot be told apart from touch data alone"
                                  if ambiguous else ""),
                     distance=best_dist or 0, ambiguous=ambiguous, alt_moves=alt_moves,
                     raw_faces=touched_faces)


# ---------------------------------------------------------------------------
# Self-derived enumeration of all 120 (orientation, move) combinations.
#
# Kept here instead of in generate_graph.py so validate_algorithm.py never
# has to import that module (or anything else the web app also uses).
# ---------------------------------------------------------------------------
def all_combinations() -> list[dict]:
    """Every (source orientation id, move) pair, 24*5 = 120 total, each with
    its resulting target id -- derived purely from this module's own
    _NEIGHBORS table."""
    combos = []
    for source in range(24):
        for move, (target, _) in _NEIGHBORS[source].items():
            combos.append({"source": source, "move": move, "target": target})
    return combos
