import tkinter as tk
from tkinter import messagebox
import os
import json
import threading
import time
try:
    import psutil
except ImportError:
    psutil = None
try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None

# Persistent session counter storage
SESSION_FILE = "session_counters.json"
session_counters = {}
if os.path.isfile(SESSION_FILE):
    try:
        with open(SESSION_FILE, "r") as f:
            session_counters = json.load(f)
    except Exception as e:
        session_counters = {}

# Global state variables
username = ""
save_dir = ""
recording = False
stop_event = threading.Event()
start_time = None
event_list = []

# Listener references
keyboard_listener = None
mouse_listener = None

def is_game_running():
    """Check if League of Legends game process is running."""
    if psutil is None:
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
    """Begin recording keyboard and mouse events."""
    global recording, start_time, event_list, keyboard_listener, mouse_listener
    recording = True
    start_time = time.perf_counter()
    event_list = []
    # Callback functions for listeners
    def on_press(key):
        if not recording:
            return False  # stop if recording ended
        t = int((time.perf_counter() - start_time) * 1000)
        try:
            name = key.char
        except AttributeError:
            name = str(key)
        if name is None:
            name = str(key)
        if str(name).startswith('Key.'):
            name = str(name)[4:]
        event_list.append([t, "keyboard", name, "press", "", "", "", ""])
    def on_release(key):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        try:
            name = key.char
        except AttributeError:
            name = str(key)
        if name is None:
            name = str(key)
        if str(name).startswith('Key.'):
            name = str(name)[4:]
        event_list.append([t, "keyboard", name, "release", "", "", "", ""])
    def on_move(x, y):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        event_list.append([t, "mouse", "move", "", str(x), str(y), "", ""])
    def on_click(x, y, button, pressed):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        btn = str(button)
        if btn.startswith('Button.'):
            btn = btn[7:]
        action = "press" if pressed else "release"
        event_list.append([t, "mouse", btn, action, str(x), str(y), "", ""])
    def on_scroll(x, y, dx, dy):
        if not recording:
            return False
        t = int((time.perf_counter() - start_time) * 1000)
        event_list.append([t, "mouse", "scroll", "", str(x), str(y), str(dx), str(dy)])
    # Start listeners
    if keyboard is not None:
        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.start()
    if mouse is not None:
        mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        mouse_listener.start()

def stop_recording():
    """Stop recording events and save data if >=10min."""
    global recording, keyboard_listener, mouse_listener
    if not recording:
        return
    recording = False
    duration_ms = int((time.perf_counter() - start_time) * 1000)
    # Stop listeners if running
    if keyboard_listener:
        try:
            keyboard_listener.stop()
        except Exception:
            pass
        keyboard_listener = None
    if mouse_listener:
        try:
            mouse_listener.stop()
        except Exception:
            pass
        mouse_listener = None
    if duration_ms >= 1 * 60 * 1000:
        # Update session counter for this user
        session_counters[username] = session_counters.get(username, 0) + 1
        current_session = session_counters[username]
        # Save updated counters
        try:
            with open(SESSION_FILE, "w") as f:
                json.dump(session_counters, f)
        except Exception as e:
            print(f"Warning: Failed to update session file: {e}")
        # Save CSV data to file
        filename = f"{username}_{current_session}.csv"
        filepath = os.path.join(save_dir, filename)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("time_ms,event_type,key_button,action,x,y,dx,dy\n")
                for rec in event_list:
                    f.write(",".join(str(x) for x in rec) + "\n")
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")
    # Clear the event list (memory cleanup)
    event_list.clear()

def monitor_game():
    """Thread to monitor the LoL process and handle recording triggers."""
    was_running = False
    while not stop_event.is_set():
        running = False
        try:
            running = is_game_running()
        except Exception:
            running = False
        if running and not was_running:
            start_recording()
            was_running = True
        elif not running and was_running:
            stop_recording()
            was_running = False
        # Sleep a bit to avoid busy-waiting (check 10 times per second)
        for _ in range(10):
            if stop_event.is_set():
                break
            time.sleep(0.1)
    # If exiting and a session was still running, stop it
    if was_running:
        stop_recording()

# Set up the Tkinter GUI
root = tk.Tk()
root.title("LoL Input Recorder")
root.geometry("400x160")

tk.Label(root, text="Username:").pack(pady=(10,0))
username_entry = tk.Entry(root, width=40)
username_entry.pack()
tk.Label(root, text="Save path:").pack(pady=(5,0))
path_entry = tk.Entry(root, width=40)
path_entry.pack()
# Default to Desktop directory
default_path = os.path.join(os.path.expanduser("~"), "Desktop")
path_entry.insert(0, default_path)

button_frame = tk.Frame(root)
button_frame.pack(pady=10)
def on_save():
    global username, save_dir
    u = username_entry.get().strip()
    p = path_entry.get().strip()
    if u == "":
        messagebox.showerror("Input Error", "Please enter a username.")
        return
    if p == "":
        messagebox.showerror("Input Error", "Please enter a save path.")
        return
    p = os.path.expanduser(p)
    if not os.path.isdir(p):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Path Error", f"Could not create directory:\n{e}")
            return
    username = u
    save_dir = p
    # Initialize session counter for user if not already in data
    session_counters.setdefault(username, session_counters.get(username, 0))
    # Disable input fields after saving
    username_entry.config(state='disabled')
    path_entry.config(state='disabled')
    save_button.config(state='disabled')
    # Start the background monitoring thread
    t = threading.Thread(target=monitor_game, daemon=True)
    t.start()
    messagebox.showinfo("Monitoring", "Now monitoring for LoL game. Close this window or click Exit to stop.")
def on_exit():
    # Signal the monitoring thread to stop and close the application
    stop_event.set()
    if keyboard_listener:
        try:
            keyboard_listener.stop()
        except Exception:
            pass
    if mouse_listener:
        try:
            mouse_listener.stop()
        except Exception:
            pass
    root.destroy()
save_button = tk.Button(button_frame, text="Save", command=on_save)
exit_button = tk.Button(button_frame, text="Exit", command=on_exit)
save_button.pack(side="left", padx=10)
exit_button.pack(side="left", padx=10)

# Start the GUI event loop
root.mainloop()
