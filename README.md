# LoL Input Recorder & Behavioral Authentication

## Introduction

This project captures mouse and keyboard events during League of Legends (LoL) gameplay sessions. It employs a dual-stream Transformer model to perform behavioral authentication, distinguishing account owners from smurfs based on keystroke and mouse movement patterns.

### Features
- **Input Recording:** A GUI-based Python script (`record_debug.py`) to automatically monitor LoL sessions and record input events.
- **Data Analysis & Modeling:** Python scripts for data preprocessing, feature engineering, Transformer model training using triplet loss, and session verification.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Record Input Sessions

Run the recording GUI:
```bash
python record.py
```
- Enter your Username and select the Save Path, then click **Save**.
- The script automatically detects the LoL game and starts/stops recording, saving sessions as `username_session.csv`.
- Debugging logs are available in `record_debug.log`.

### 2. Preprocess Data

Prepare data for training:
```bash
python -m preprocess.dataset --data_dir data/raw --debug
```
- Verify data parsing, feature extraction, and tensor shapes.

### 3. Train the Model

Train the dual-stream Transformer:
```bash
python -m training.train --config configs/config.yaml
```
- Checkpoints are saved in `experiments/checkpoints/`.

### 4. Verify Sessions

Authenticate recorded sessions:
```bash
python -m evaluation.verify \
  --checkpoint experiments/checkpoints/model_epoch_50.pth \
  --data_dir data/raw \
  --owner_id <player_id> \
  --test_file data/raw/<test_session>.csv
```
- Outputs session similarity scores and verification results.

## Project Structure
```
your_project/
├── record_debug.py          # Recording script with GUI
├── requirements.txt
├── .gitignore
├── session_counters.json    # Session tracking per user
├── data/
│   └── raw/                 # Recorded session CSV files
├── preprocess/
│   ├── parse_logs.py
│   ├── feature_engineering.py
│   └── dataset.py
├── models/
│   ├── positional_encoding.py
│   └── dual_transformer.py
├── training/
│   └── train.py
├── evaluation/
│   └── verify.py
└── experiments/
    ├── logs/
    └── checkpoints/
```

## License & Author

Created by Your Name.