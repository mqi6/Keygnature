import os, re, argparse
import numpy as np
import pandas as pd

# Map of special key names to ASCII codes
SPECIAL_KEYS = {
    'tab': 9,
    'enter': 13,
    'space': 32,
    'backspace': 8,
    'shift': 16,
    'ctrl': 17,
    'alt': 18,
    # add more mappings as needed
}


def parse_filename(fn):
    """
    From "12345_7.csv" or "12345_7_Ahri.csv" extract:
      user_id = "12345", session_id = "7" or "7_Ahri"
    """
    m = re.match(r'([^_]+)_(.+)\.csv$', fn)
    if not m:
        raise ValueError(f"Filename {fn!r} doesn't match pattern")
    return m.group(1), m.group(2)


def extract_events(df):
    """
    Pair up keyboard press->release events, record [start_ms, end_ms, ascii_code].
    Skip non-keyboard or unmapped keys.
    """
    presses = {}
    events = []
    for _, row in df.sort_values('time_ms').iterrows():
        if row['event_type'].lower() != 'keyboard':
            continue
        key = str(row['key_button'])
        action = row['action'].lower()
        t = row['time_ms']
        if action == 'press':
            presses[key] = t
        elif action == 'release' and key in presses:
            t0 = presses.pop(key)
            # Determine ASCII code
            if len(key) == 1:
                code = ord(key)
            else:
                code = SPECIAL_KEYS.get(key.lower())
                if code is None:
                    continue  # skip unmapped special keys
            events.append([t0, t, code])
    return np.array(events, dtype=np.float32)


def segment_events(events, sequence_length):
    """
    Break events array into non-overlapping segments of length sequence_length.
    Segments shorter than sequence_length are dropped.
    Returns list of (segment_index, segment_array).
    """
    num_segments = events.shape[0] // sequence_length
    segments = []
    for i in range(num_segments):
        start = i * sequence_length
        end = start + sequence_length
        segments.append((i, events[start:end]))
    return segments


def convert_folder(input_folder, output_npy, sequence_length):
    dataset = {}

    for fn in os.listdir(input_folder):
        if not fn.lower().endswith('.csv'):
            continue
        user, sess = parse_filename(fn)
        file_path = os.path.join(input_folder, fn)
        try:
            df = pd.read_csv(
                file_path,
                engine='python',
                on_bad_lines='skip'
            )
        except Exception as e:
            print(f"Skipping {fn}: parse error: {e}")
            continue

        ev = extract_events(df)
        if ev.size == 0:
            continue  # no usable keystrokes

        segments = segment_events(ev, sequence_length)
        for idx, seg in segments:
            session_key = f"{sess}_{idx}"
            dataset.setdefault(user, {})[session_key] = seg

    os.makedirs(os.path.dirname(output_npy) or '.', exist_ok=True)
    np.save(output_npy, dataset, allow_pickle=True)
    total = sum(len(s) for s in dataset.values())
    print(f"Converted {total} segments from {len(dataset)} users → {output_npy}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert folder of userid_session CSVs → segmented .npy for training with aligned ASCII codes")
    parser.add_argument('input_folder', help="Path to folder containing CSV session files")
    parser.add_argument('output_npy',    help="Path to write output .npy dataset")
    parser.add_argument('-l', '--length', type=int, default=128,
                        help="Number of events per segment (default: 128)")
    args = parser.parse_args()
    convert_folder(args.input_folder, args.output_npy, args.length)
