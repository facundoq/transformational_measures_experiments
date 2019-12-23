import os
import logging
import subprocess
import sys
from datetime import datetime
def run_python (venv_path, python_command,terminate_if_error=True,print_elapsed=True):
    python_executable = "python3"
    venv_activate_path = os.path.join(venv_path, "bin", "activate")
    command = f'{venv_activate_path} && {python_executable} {python_command}'

    strf_format = "%Y/%m/%d %H:%M:%S"
    dt = datetime.now()
    dt_string = dt.strftime(strf_format)
    message=f"[{dt_string}] *** Running {command} "
    logging.info(message)
    print(message)
    #subprocess.call(f'/bin/bash -c {command}', shell=True)
    return_code = subprocess.call(command, shell=True)
    if return_code != 0 and terminate_if_error:
        dt = datetime.now()
        dt_string = dt.strftime(strf_format)
        print(f"[{dt_string}] *** Terminating, error running: {command}.")
        sys.exit(return_code)






