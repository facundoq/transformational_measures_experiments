import os
import logging
import subprocess
import sys
def get_venv_path():
    if len(sys.argv) == 1:
        venv_path = "/home/facundo/dev/env/vm"
        logging.info(f"No virtual environment path specified, defaulting to {venv_path}.")
    elif len(sys.argv) == 2:
        venv_path = sys.argv[1]
        logging.info(f"Using virtual environment {venv_path}.")
    else:
        sys.exit("Wrong number of arguments")
    return venv_path

def run_python (venv_path, python_command):
    python_executable = "python3"
    venv_activate_path = os.path.join(venv_path, "bin", "activate")
    command = f"source {venv_activate_path} && {python_executable} {python_command}"
    logging.info(f"Running {command}")
    print(f"Running {command}")
    subprocess.call(f'/bin/bash -c "{command}"', shell=True)