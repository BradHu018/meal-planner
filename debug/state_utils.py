import json
from pathlib import Path


SNAPSHOT_DIR = Path("debug/snapshots")


def save_state(state, filename):
    SNAPSHOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = SNAPSHOT_DIR / filename

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            state,
            file,
            indent=2
        )

    print(
        f"\nSaved state snapshot to: {path}"
    )


def load_state(filename):
    path = SNAPSHOT_DIR / filename

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)