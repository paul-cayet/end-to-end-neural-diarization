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
from typing import Dict, List
import yaml

def generate_unique_logpath(logdir: str, raw_run_name: str) -> str:
    """Generate a unique directory name
    Args
    ----
    logdir: str
        the prefix directory
    raw_run_name: str
        the base name
    Returns
    -------
    log_path: str
        a non-existent path like logdir/raw_run_name_xxxx
                where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def _verify_args(args: Dict, keys_to_extract: List[str]) -> Dict:
    new_args = dict()
    for key in keys_to_extract:
        if key not in args:
            raise KeyError(f"key {key} is missing from the arguments {args}")
        new_args[key] = args[key]

    return new_args

