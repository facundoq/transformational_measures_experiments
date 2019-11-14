import os
import logging
import subprocess
import sys

def run_python (venv_path, python_command):
    python_executable = "python3"
    venv_activate_path = os.path.join(venv_path, "bin", "activate")
    command = f'{venv_activate_path} && {python_executable} {python_command}'
    logging.info(f"Running {command}")
    print(f"*** Running {command}")
    #subprocess.call(f'/bin/bash -c {command}', shell=True)
    subprocess.call(command, shell=True)



