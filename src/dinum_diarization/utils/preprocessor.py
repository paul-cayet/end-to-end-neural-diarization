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
