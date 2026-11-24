# TactCube Rotation Recognition

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Validating the cube algorithm

`validate_algorithm.py` is a standalone, self-contained check of the cube
rotation recognition algorithm. It exercises all 120 possible combinations
of (starting orientation, elementary move) and checks that the tracker
always recognizes the correct move and lands on the correct resulting
state.

It imports only `standalone_cube_algorithm.py` — a byte-for-byte copy of the
`CubeTracker` class used in the application described in the article "TactCube Rotation Recognition: A Real-Time Algorithm for Tangible Conversation with Ambient Intelligence", kept here for exactly this purpose. No MQTT logic is involved, so it is usable locally without any server or broker setup.

Run it with:

```bash
source venv/bin/activate
python3 validate_algorithm.py
```

The script prints a PASS/FAIL line per combination, followed by a summary of
any unrecognized moves, ambiguous transitions, and failures. It exits with
code `0` if and only if all 120 combinations pass, which makes it suitable
for use in CI.
