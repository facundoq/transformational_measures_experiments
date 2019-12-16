#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK

import config
import subprocess
import pathlib
base_folder = config.plots_base_folder()
# base_folder = pathlib.Path("/home/facundoq/Dropbox/paper/plots/")

image_files = base_folder.rglob("*.png")
print(image_files)

binary = 'convert'

print(image_files)
for image_file in image_files:
    # print(image_file)
    # if not "NM" in str(image_file):
    #     continue
    image_filepath=f'"{image_file}"'
    image_cmd = [binary, image_filepath, "+matte",image_filepath]
    image_cmd_str = " ".join(image_cmd)
    print(image_cmd_str)
    subprocess.call(image_cmd, shell=False)

