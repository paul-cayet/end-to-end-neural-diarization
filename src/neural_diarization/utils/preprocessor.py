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

import json
from torchaudio import AudioMetaData


class Torchaudiopreprocessor:
    """Preprocessor to be used when loading a Pyannote Dataset 
    to use precomputed audio metadata.
    """
    def __init__(self, path: str):
        """
        Args
        ----
        path: str
            Path to the torchaudio precomputed audio metadata. 
            Json file is generated in `precompute_torchaudio_info.py`
        """
        with open(path) as f:
            self.data = json.load(f)

    def __call__(self, file):
        id = file["uri"]
        if id not in self.data:
            raise KeyError
        return AudioMetaData(**self.data[id])
