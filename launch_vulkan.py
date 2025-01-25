import tkinter as tk
from tkinter import filedialog
import subprocess

# Function to choose the .gguf file
def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select a .gguf File",
        filetypes=[("GGUF Files", "*.gguf"), ("All Files", "*.*")]
    )
    return file_path

# Get the selected .gguf file
gguf_file = choose_file()

if gguf_file:
    print(f"Selected file: {gguf_file}")
    # Launch the backend with the chosen .gguf file
    subprocess.Popen([
        "./llama-server.exe",
        "-m", gguf_file,
        "-c", "4096", "-b", "512", "-ub", "512", "--mlock", "--no-mmap", "-ngl", "99", "--no-webui", "--repeat-penalty", "1.1"
        "--flash-attn"
    ])
    # Launch the Streamlit UI
    subprocess.Popen(["streamlit", "run", "Main.py"
    ])
else:
    print("No file selected.")
