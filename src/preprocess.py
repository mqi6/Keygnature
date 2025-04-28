from pathlib import Path
import zipfile
import shutil
import pandas as pd
import pickle
import time

# ─── PATHS ───────────────────────────────────────────────────────────────────────
RAW_ZIP       = Path(__file__).resolve().parent.parent / "data" / "raw" / "Keystrokes.zip"
EXTRACT_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw_extracted"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

if __name__ == "__main__":
    print("=== Preprocessing Started ===")
    t0 = time.time()

    # 1) Unzip raw archive
    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Place your ZIP at {RAW_ZIP!s}")
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)
    print(f"[{time.time()-t0:.2f}s] Unzipping {RAW_ZIP.name} → {EXTRACT_DIR}")
    with zipfile.ZipFile(RAW_ZIP, 'r') as z:
        z.extractall(EXTRACT_DIR)
    print(f"[{time.time()-t0:.2f}s] Unzip complete")

    # 2) Locate keystroke txt files
    txt_files = list(EXTRACT_DIR.rglob("*_keystrokes.txt"))
    #txt_files = txt_files[:2000]   # ← only keep the first 2000 files
    print(f"[{time.time()-t0:.2f}s] Found {len(txt_files)} keystroke files")

    # 3) Read & concatenate
    dfs = []
    for idx, path in enumerate(txt_files, 1):
        t1 = time.time()
        print(f"  → [{idx}/{len(txt_files)}] Reading {path.name} ... ", end="", flush=True)
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                usecols=["PARTICIPANT_ID","TEST_SECTION_ID","PRESS_TIME","RELEASE_TIME","KEYCODE"],
                encoding="latin-1",
                on_bad_lines="skip",
                low_memory=False
            )
            # convert KEYCODE to int, coercing bad values to 0
            df["KEYCODE"] = (
                pd.to_numeric(df["KEYCODE"], errors="coerce")
                  .fillna(0)
                  .astype(int)
            )
            dfs.append(df)
            print(f"{len(df):,} rows ({time.time()-t1:.2f}s)")
        except Exception as e:
            print(f"Error: {e} ({time.time()-t1:.2f}s)")

    if not dfs:
        raise FileNotFoundError(f"No `*_keystrokes.txt` files loaded under {EXTRACT_DIR!s}")

    print(f"[{time.time()-t0:.2f}s] Concatenating {len(dfs)} dataframes ...", end="", flush=True)
    all_df = pd.concat(dfs, ignore_index=True)
    print(f" {len(all_df):,} rows ({time.time()-t0:.2f}s)")

    # 4) Rename to pipeline schema
    all_df.rename(columns={
        "PARTICIPANT_ID": "user_id",
        "TEST_SECTION_ID":"session_id",
        "PRESS_TIME":     "press_time",
        "RELEASE_TIME":   "release_time",
        "KEYCODE":        "key_code"
    }, inplace=True)

    # 5) Group by user → list of session arrays
    print(f"[{time.time()-t0:.2f}s] Grouping by user & session ...")
    data_dict = {}
    for user, grp in all_df.groupby("user_id"):
        sessions = []
        for sess, sg in grp.groupby("session_id"):
            arr = sg[["press_time","release_time","key_code"]].to_numpy()
            sessions.append(arr)
        if len(sessions) == 15:
            data_dict[user] = sessions
    users = list(data_dict.values())
    U = len(users)
    print(f"[{time.time()-t0:.1f}s] {U} users with 15 sessions")
    
    # 6) Dynamic split (60/20/20)
    if U>=3:
        tr_end = int(0.6*U)
        val_end= int(0.8*U)
        training_data   = users[:tr_end]
        validation_data = users[tr_end:val_end]
        testing_data    = users[val_end:]
    else:
        # small‐dataset fallback
        training_data   = users
        validation_data = users
        testing_data    = users
    print(f"[{time.time()-t0:.1f}s] Split → "
          f"{len(training_data)} train, "
          f"{len(validation_data)} val, "
          f"{len(testing_data)} test")

    # 7) Write pickles
    for name, data in [
        ("training_data.pickle",   training_data),
        ("validation_data.pickle", validation_data),
        ("testing_data.pickle",    testing_data),
    ]:
        out = PROCESSED_DIR / name
        with open(out, "wb") as f:
            pickle.dump(data, f)
        print(f"[{time.time()-t0:.1f}s] Wrote {len(data)} → {out.name}")

    print(f"=== PREPROCESS COMPLETE in {time.time()-t0:.1f}s ===")
