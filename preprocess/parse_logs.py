import csv

def parse_log(file_path):
    """
    Parse the CSV log file and return two lists: keyboard_events and mouse_events.
    Each event is a dictionary with:
      - timestamp: float
      - For mouse events: event (e.g., "move", "left") and optional action, x, y.
      - For keyboard events: key (e.g., "y") and action ("press"/"release").
    """
    keyboard_events = []
    mouse_events = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Expected row format: timestamp, device, event/key, action, x, y, ...
        for row in reader:
            if not row or len(row) < 4:
                continue
            timestamp = float(row[0])
            device = row[1].strip().lower()
            if device == "mouse":
                event = row[2].strip().lower()
                action = row[3].strip().lower() if row[3].strip() != "" else None
                try:
                    x = float(row[4]) if row[4].strip() != "" else None
                except Exception:
                    x = None
                try:
                    y = float(row[5]) if row[5].strip() != "" else None
                except Exception:
                    y = None
                mouse_events.append({
                    "timestamp": timestamp,
                    "event": event,
                    "action": action,
                    "x": x,
                    "y": y
                })
            elif device == "keyboard":
                key = row[2].strip()
                action = row[3].strip().lower() if row[3].strip() != "" else None
                keyboard_events.append({
                    "timestamp": timestamp,
                    "key": key,
                    "action": action
                })
    return {"mouse": mouse_events, "keyboard": keyboard_events}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_logs.py <csv_file>")
        exit(1)
    file_path = sys.argv[1]
    data = parse_log(file_path)
    print("Parsed {} mouse events and {} keyboard events.".format(len(data["mouse"]), len(data["keyboard"])))
