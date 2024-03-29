import subprocess
import sys
from pathlib import Path
import os

python = str(Path(sys.executable).name)

FILE_PATH = os.getcwd() + "\\run_websocket_server.py"
print(FILE_PATH)

call_alice = [python, FILE_PATH, "--port", "8777", "--id", "alice"]
call_bob = [python, FILE_PATH, "--port", "8778", "--id", "bob"]
call_charlie = [python, FILE_PATH, "--port", "8779", "--id", "charlie"]

print("Starting server for Alice")
subprocess.Popen(call_alice)

print("Starting server for Bob")
subprocess.Popen(call_bob)

print("Starting server for Charlie")
subprocess.Popen(call_charlie)
