# GNU GENERAL PUBLIC LICENSE Version 3

# Copyright (C) 2024 - P. Cayet, N. Ibanez and L. Rondier

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
from tqdm import tqdm
import glob
import argparse

def convert_sd_files(sd_dirname: str, rttm_dirname: str):
    """Function to convert .sd annotations into Pyannote-compatible
    .rttm annotations.
    Args
    ----
    sd_dirname: str
        Path to the parent directory of the .sd annotations
    rttm_dirname: str
        Path to the desired parent directory of the .rttm annotations to be generated.
    """
    converted_dir = os.path.join(rttm_dirname, "converted")
    os.makedirs(converted_dir, exist_ok=True)

    files = glob.glob(sd_dirname + "/**/*.sd", recursive=True)

    all_f = True
    print("Starting sd -> rttm file conversion...")
    for file in tqdm(files, desc="Converting files"):
        with open(str(file), "r") as f:
            lines = f.readlines()
            filename = ".".join(file.split("/")[-1].split(".")[:-1])

            
            new_lines = []

            for line in lines:
                elts = line.split(" ")
                start = elts[2]
                end = elts[3]
                delta = str(float(end) - float(start))
                speaker = elts[4]
                is_f = (elts[5] == "F")
                if all_f and not is_f:
                    all_f = False

                new_line = "SPEAKER " + filename + " 1 " + start + " " + delta + " <NA> <NA> " + speaker + " <NA> <NA>"
                new_lines.append(new_line)
            
            filename = os.path.join(converted_dir, filename)
            with open(filename + ".rttm", "w") as f:
                f.write("\n".join(new_lines))
    print("sd -> rttm file conversion completed ✅")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converting sd files into rttm")
    parser.add_argument("--sd_dirname", type=str, help="Directory containing sd files")
    parser.add_argument("--rttm_dirname", type=str, help="Directory where to save the converted rttm files")

    args = parser.parse_args()

    sd_dirname = args.sd_dirname
    rttm_dirname = args.rttm_dirname

    convert_sd_files(sd_dirname,rttm_dirname)