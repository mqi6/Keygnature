# count_sessions.py

from collections import Counter
from preprocess.dataset import SessionDataset

def main():
    dataset = SessionDataset(
        data_dir="data/raw",
        segment_duration=10,
        mouse_sample_rate=100,
        max_mouse_len=1000,
        max_key_len=500,
        debug=False
    )
    # Collect all player_ids
    all_ids = [sample["player_id"] for sample in dataset]
    counts = Counter(all_ids)
    print("Session counts per player:")
    for player, cnt in counts.items():
        print(f"  {player}: {cnt}")

if __name__ == "__main__":
    main()
