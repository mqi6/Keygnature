# convert_csv_folder_to_npy.py
'''
python convert_csv_folder_to_npy.py ~/Desktop/raw/ ~/Desktop/game_dev_set.npy
'''
import os, re, argparse
import numpy as np
import pandas as pd

def parse_filename(fn):
    """
    From "12345_7.csv" or "12345_7_Ahri.csv" extract:
      user_id = "12345", session_id = "7" or "7_Ahri"
    """
    m = re.match(r'([^_]+)_(.+)\.csv$', fn)
    if not m:
        raise ValueError(f"Filename {fn!r} doesn't match pattern")
    return m.group(1), m.group(2)

def map_event_code(etype, key, action, code_map):
    """
    Assign a unique integer code for each (event_type, key_button, action) tuple.
    """
    k = (etype, key, action)
    if k not in code_map:
        code_map[k] = len(code_map) + 1
    return code_map[k]

def extract_events(df, code_map):
    """
    Walk through sorted rows, pair up press→release for each (etype, key_button),
    and record [start_ms, end_ms, code].
    Skip any unmatched or 'move' events.
    """
    presses = {}
    events = []
    for _, row in df.sort_values('time_ms').iterrows():
        t = row['time_ms']
        etype = row['event_type']
        key   = row['key_button']
        action= row['action']
        if action == 'press':
            presses[(etype, key)] = t
        elif action == 'release' and (etype, key) in presses:
            t0 = presses.pop((etype, key))
            code = map_event_code(etype, key, action, code_map)
            events.append([t0, t, code])
    return np.array(events, dtype=np.float32)

def convert_folder(input_folder, output_npy):
    dataset  = {}
    code_map = {}
    for fn in os.listdir(input_folder):
        if not fn.lower().endswith('.csv'):
            continue
        user, sess = parse_filename(fn)
        df = pd.read_csv(os.path.join(input_folder, fn))
        ev = extract_events(df, code_map)
        if ev.size == 0:
            continue  # skip entirely empty sessions
        dataset.setdefault(user, {})[sess] = ev

    # ensure the output folder exists
    os.makedirs(os.path.dirname(output_npy) or '.', exist_ok=True)
    np.save(output_npy, dataset, allow_pickle=True)
    # (Optional) save the mapping so you can decode codes later:
    np.save(output_npy.replace('.npy','_code_map.npy'), code_map, allow_pickle=True)
    print(f"Converted {sum(len(v) for v in dataset.values())} sessions from "
          f"{len(dataset)} users → {output_npy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert folder of userid_session CSVs → single .npy for training")
    parser.add_argument('input_folder', help="Path to CSV files")
    parser.add_argument('output_npy',    help="Path to write .npy dataset")
    args = parser.parse_args()
    convert_folder(args.input_folder, args.output_npy)
