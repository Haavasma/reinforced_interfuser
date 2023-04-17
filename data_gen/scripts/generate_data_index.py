from pathlib import Path

DATA_DIR = Path("expert_data/train")
DATASET_INDEX_PATH = DATA_DIR / "dataset_index.txt"


def main():
    if DATASET_INDEX_PATH.exists():
        DATASET_INDEX_PATH.unlink()

    for route_dir in DATA_DIR.iterdir():
        if not route_dir.is_dir():
            continue

        num_frames = len(list((route_dir / "rgb_front").iterdir()))
        relative_route_dir = route_dir.relative_to(DATA_DIR)
        with open(DATASET_INDEX_PATH, "a") as f:
            f.write(f"{relative_route_dir}/ {num_frames}\n")

    return


if __name__ == "__main__":

    main()
