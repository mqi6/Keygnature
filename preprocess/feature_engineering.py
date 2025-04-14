from parse_logs import parse_log

def compute_mouse_velocity(mouse_events):
    """
    Compute velocity (vx, vy) for mouse 'move' events.
    For each consecutive mouse move event, calculate the difference
    in positions divided by the time difference.
    """
    if not mouse_events:
        return mouse_events
    prev_event = None
    for event in mouse_events:
        if event["event"] == "move" and prev_event is not None and prev_event["event"] == "move":
            dt = event["timestamp"] - prev_event["timestamp"]
            if dt > 0:
                dx = event["x"] - prev_event["x"]
                dy = event["y"] - prev_event["y"]
                event["vx"] = dx / dt
                event["vy"] = dy / dt
            else:
                event["vx"] = 0.0
                event["vy"] = 0.0
        else:
            event["vx"] = 0.0
            event["vy"] = 0.0
        prev_event = event
    return mouse_events

def compute_keyboard_hold_times(keyboard_events):
    """
    Compute hold times for keyboard events.
    Matches each 'press' with its corresponding 'release' and
    adds a 'hold_time' field for release events.
    """
    key_press_times = {}
    new_keyboard_events = []
    for event in keyboard_events:
        if event["action"] == "press":
            key = event["key"]
            key_press_times.setdefault(key, []).append(event["timestamp"])
            new_keyboard_events.append(event)
        elif event["action"] == "release":
            key = event["key"]
            if key in key_press_times and key_press_times[key]:
                press_time = key_press_times[key].pop(0)
                event["hold_time"] = event["timestamp"] - press_time
            else:
                event["hold_time"] = 0.0
            new_keyboard_events.append(event)
        else:
            new_keyboard_events.append(event)
    return new_keyboard_events

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python feature_engineering.py <csv_file>")
        exit(1)
    file_path = sys.argv[1]
    data = parse_log(file_path)
    mouse_events = compute_mouse_velocity(data["mouse"])
    keyboard_events = compute_keyboard_hold_times(data["keyboard"])
    print("First mouse event:", mouse_events[0] if mouse_events else "No mouse events.")
    print("First keyboard event:", keyboard_events[0] if keyboard_events else "No keyboard events.")
