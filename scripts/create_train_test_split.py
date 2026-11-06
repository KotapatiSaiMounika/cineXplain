
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/movielens")
OUTPUT_DIR = DATA_DIR / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RATING_FILE = DATA_DIR / "rating.csv"
CHUNK_SIZE = 500_000

print("Starting MovieLens train/test split...")

# PASS 1: Find the latest rating row for each user
latest_rows = {}
row_offset = 0

for chunk in pd.read_csv(
    RATING_FILE,
    chunksize=CHUNK_SIZE,
    dtype={"timestamp": str}
):
    chunk["row_id"] = range(
        row_offset,
        row_offset + len(chunk)
    )

    # Find each user's latest timestamp in this chunk
    latest = chunk.loc[
        chunk.groupby("userId")["timestamp"].idxmax()
    ]

    for user_id, timestamp, row_id in zip(
        latest["userId"],
        latest["timestamp"],
        latest["row_id"]
    ):
        previous = latest_rows.get(user_id)

        if previous is None or (timestamp, row_id) > previous:
            latest_rows[user_id] = (timestamp, row_id)

    row_offset += len(chunk)

print("Latest rating identified for every user.")

# Store selected test row IDs
test_row_ids = {
    row_id for timestamp, row_id in latest_rows.values()
}

# PASS 2: Write training and testing files in chunks
train_path = OUTPUT_DIR / "train_ratings.csv"
test_path = OUTPUT_DIR / "test_ratings.csv"

train_count = 0
test_count = 0
row_offset = 0

for chunk in pd.read_csv(
    RATING_FILE,
    chunksize=CHUNK_SIZE,
    dtype={"timestamp": str}
):
    row_ids = range(
        row_offset,
        row_offset + len(chunk)
    )

    mask = pd.Series(
        [row_id in test_row_ids for row_id in row_ids],
        index=chunk.index
    )

    train_chunk = chunk.loc[~mask]
    test_chunk = chunk.loc[mask]

    train_chunk.to_csv(
        train_path,
        mode="a" if train_count else "w",
        index=False,
        header=(train_count == 0)
    )

    test_chunk.to_csv(
        test_path,
        mode="a" if test_count else "w",
        index=False,
        header=(test_count == 0)
    )

    train_count += len(train_chunk)
    test_count += len(test_chunk)

    row_offset += len(chunk)

print("\nTRAIN/TEST SPLIT COMPLETED")
print("=" * 40)
print("Total ratings:", train_count + test_count)
print("Training ratings:", train_count)
print("Testing ratings:", test_count)
print("Testing users:", test_count)
print("Files saved in:", OUTPUT_DIR)
