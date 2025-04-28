# Updated recording script with enhanced debugging logs

import tkinter as tk
from tkinter import messagebox
import os
import json
import threading
import time
import logging
import platform
import sys

# Setup logging to file and console
LOG_FILE = "record_debug.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info(f"Script started on {platform.system()} {platform.release()}, Python {platform.python_version()}")

# Try imports
try:
    import psutil
    logging.info("Imported psutil successfully")
except ImportError as e:
    psutil = None
    logging.warning(f"psutil import failed: {e}")

try:
    from pynput import keyboard, mouse
    logging.info("Imported pynput successfully")
except ImportError as e:
    keyboard = None
    mouse = None
    logging.warning(f"pynput import failed: {e}. Please install via 'pip install pynput'")

# Persistent session counter storage
SESSION_FILE = "session_counters.json"
session_counters = {}
if os.path.isfile(SESSION_FILE):
    try:
        with open(SESSION_FILE, "r") as f:
            session_counters = json.load(f)
        logging.info(f"Loaded session counters from {SESSION_FILE}")
    except Exception as e:
        logging.error(f"Failed to load session counters: {e}")

# Global state
username = ""
save_dir = ""
recording = False
stop_event = threading.Event()
start_time = None
event_list = []
keyboard_listener = None
mouse_listener = None

def log_event(msg):
    logging.debug(msg)

def is_game_running():
    """Check if League of Legends process is running."""
    if psutil is None:
        logging.error("psutil not available; cannot detect game process.")
        return False
    for proc in psutil.process_iter(['name']):
        try:
            name = proc.info['name']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name and "League of Legends" in name:
            return True
    return False

def start_recording():
    global recording, start_time, event_list, keyboard_listener, mouse_listener
    recording = True
    start_time = time.perf_counter()
    event_list.clear()  
    logging.info("Recording started.")

    def on_press(key):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        try:
            name = key.char
        except AttributeError:
            name = str(key)
        event_list.append([t, "keyboard", name, "press", "", "", "", ""])
    
    def on_release(key):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        try:
            name = key.char
        except AttributeError:
            name = str(key)
        event_list.append([t, "keyboard", name, "release", "", "", "", ""])
        if key == keyboard.Key.esc:
            logging.info("ESC pressed, stopping recording.")
            return False
    
    def on_move(x, y):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        event_list.append([t, "mouse", "move", "", x, y, "", ""])
    
    def on_click(x, y, button, pressed):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        action = "press" if pressed else "release"
        event_list.append([t, "mouse", str(button), action, x, y, "", ""])
    
    def on_scroll(x, y, dx, dy):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        event_list.append([t, "mouse", "scroll", "", x, y, dx, dy])
    
    try:
        if keyboard:
            keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            keyboard_listener.start()
        if mouse:
            mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
            mouse_listener.start()
    except Exception as e:
        logging.error(f"Failed to start listeners: {e}")

def stop_recording():
    global recording, keyboard_listener, mouse_listener
    if not recording:
        return
    recording = False
    duration = int((time.perf_counter() - start_time) * 1000)
    logging.info(f"Recording stopped. Duration: {duration}ms, Events: {len(event_list)}")
    
    if keyboard_listener:
        keyboard_listener.stop()
        keyboard_listener = None
    if mouse_listener:
        mouse_listener.stop()
        mouse_listener = None

    if duration >= 1 * 60 * 1000:
        session_counters[username] = session_counters.get(username, 0) + 1
        current_session = session_counters[username]
        try:
            with open(SESSION_FILE, "w") as f:
                json.dump(session_counters, f)
            logging.info(f"Updated session counters: {session_counters}")
        except Exception as e:
            logging.error(f"Failed to save session counters: {e}")
        
        filename = f"{username}_{current_session}.csv"
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("time_ms,event_type,key_button,action,x,y,dx,dy\n")
                for rec in event_list:
                    f.write(",".join(map(str, rec)) + "\n")
            logging.info(f"Saved recording to {filepath}")
        except Exception as e:
            logging.error(f"Error writing file {filepath}: {e}")

    event_list.clear()

def monitor_game():
    was_running = False
    while not stop_event.is_set():
        try:
            running = is_game_running()
        except Exception as e:
            logging.error(f"is_game_running error: {e}")
            running = False
        if running and not was_running:
            start_recording()
            was_running = True
        elif not running and was_running:
            stop_recording()
            was_running = False
        time.sleep(0.1)
    if was_running:
        stop_recording()

# GUI setup
root = tk.Tk()
root.title("LoL Input Recorder")
root.geometry("400x160")

tk.Label(root, text="Username:").pack(pady=(10,0))
username_entry = tk.Entry(root, width=40)
username_entry.pack()
tk.Label(root, text="Save path:").pack(pady=(5,0))
path_entry = tk.Entry(root, width=40)
path_entry.pack()
default_path = os.path.join(os.path.expanduser("~"), "Desktop")
path_entry.insert(0, default_path)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

def on_save():
    global username, save_dir
    u = username_entry.get().strip()
    p = path_entry.get().strip()
    if not u or not p:
        messagebox.showerror("Error", "Username and save path required.")
        return
    p = os.path.expanduser(p)
    try:
        os.makedirs(p, exist_ok=True)
    except Exception as e:
        messagebox.showerror("Error", f"Could not create path: {e}")
        return
    username = u
    save_dir = p
    session_counters.setdefault(username, 0)
    username_entry.config(state='disabled')
    path_entry.config(state='disabled')
    save_button.config(state='disabled')
    threading.Thread(target=monitor_game, daemon=True).start()
    logging.info(f"Monitoring started for user '{username}', saving to '{save_dir}'")
    messagebox.showinfo("Monitoring", "Started monitoring LoL. Close window or press Exit to stop.")

def on_exit():
    stop_event.set()
    if keyboard_listener:
        keyboard_listener.stop()
    if mouse_listener:
        mouse_listener.stop()
    logging.info("Exiting application.")
    root.destroy()

save_button = tk.Button(button_frame, text="Save", command=on_save)
exit_button = tk.Button(button_frame, text="Exit", command=on_exit)
save_button.pack(side="left", padx=10)
exit_button.pack(side="left", padx=10)

root.mainloop()

