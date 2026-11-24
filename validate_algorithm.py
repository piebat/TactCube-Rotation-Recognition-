"""Standalone, reproducible validation of the cube rotation recognition
algorithm against all 120 possible (starting orientation, elementary move)
combinations.

This does NOT use cube_algorithm.py or any other project file: it imports
only standalone_cube_algorithm.py, a self-contained copy of CubeTracker kept
in this project for exactly this purpose (see that module's docstring). No
Flask, no MQTT, no session/calibration/positioning/graph-export logic --
nothing this script does can be affected by, or affect, the rest of the
server.

Run:  python3 validate_algorithm.py
Exit code is 0 iff all 120 combinations pass.

See also publish_test_protocol.py, which exercises the same 120
combinations against a LIVE app.py over MQTT (realistic sensor readings,
calibration and positioning included) using the PRODUCTION cube_algorithm.py
-- the two together cover both the algorithm in isolation and its behaviour
as actually driven by the running server.
"""
import sys

import standalone_cube_algorithm as ca

AXIS_FACES = {'A': (0, 3), 'B': (1, 2), 'C': (4, 5)}


def faces_for_state(axis: str, face: int) -> set[int]:
    return set(AXIS_FACES[axis]) | {face}


def validate() -> list[dict]:
    """Returns one result dict per combination, in a fixed deterministic
    order (sorted by source id, then move name)."""
    edges = sorted(ca.all_combinations(), key=lambda e: (e["source"], e["move"]))

    results = []
    for e in edges:
        source, target, move = e["source"], e["target"], e["move"]
        tracker = ca.CubeTracker()
        tracker._id = source  # force the known starting orientation

        start_state = tracker.state
        faces = faces_for_state(*ca._OBSERVE_OF[target])
        ev = tracker.update(faces)

        expected_state = ca._OBSERVE_OF[target]
        # "unrecognized": the tracker didn't identify any move at all for
        # this transition (invalid reading or no move reported) -- distinct
        # from a MISMATCH, where a move WAS recognized but the wrong one.
        unrecognized = (not ev.ok) or (ev.move is None)
        ok = (ev.ok and not ev.ambiguous and ev.move == move
              and ev.state == expected_state and tracker._id == target)

        results.append({
            "source": source, "target": target, "move": move,
            "start_state": start_state, "expected_state": expected_state,
            "got_state": ev.state, "got_move": ev.move,
            "ambiguous": ev.ambiguous, "unrecognized": unrecognized, "ok": ok,
        })
    return results


def main() -> int:
    results = validate()
    failures = [r for r in results if not r["ok"]]
    unrecognized = [r for r in results if r["unrecognized"]]
    ambiguous = [r for r in results if r["ambiguous"]]

    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"[{status}] id {r['source']:2d} {r['start_state']} "
              f"--{r['move']:6s}--> id {r['target']:2d} {r['expected_state']}  "
              f"got move={r['got_move']} state={r['got_state']} "
              f"ambiguous={r['ambiguous']}")

    print(f"\n{len(results) - len(failures)}/{len(results)} combinations passed")

    if unrecognized:
        print(f"\n{len(unrecognized)} UNRECOGNIZED MOVES "
              "(the tracker reported no move at all for this transition):")
        for r in unrecognized:
            print(f"  id {r['source']} --{r['move']}--> expected {r['expected_state']}, "
                  f"got move={r['got_move']} state={r['got_state']}")
    else:
        print("\nNo unrecognized moves: every transition was identified as some move.")

    if ambiguous:
        print(f"\n{len(ambiguous)} AMBIGUOUS transitions "
              "(more than one elementary move could explain the reading):")
        for r in ambiguous:
            print(f"  id {r['source']} --{r['move']}--> expected {r['expected_state']}, "
                  f"got move={r['got_move']} state={r['got_state']}")
    else:
        print("No ambiguous transitions: every reading resolved to a single elementary move.")

    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for r in failures:
            print(f"  id {r['source']} --{r['move']}--> expected {r['expected_state']}, "
                  f"got move={r['got_move']} state={r['got_state']} ambiguous={r['ambiguous']}")
        return 1

    print("\nAll 120 (orientation, move) combinations verified OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
