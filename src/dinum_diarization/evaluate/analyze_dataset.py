import os
import glob
import argparse
import yaml
import torchaudio
import torch
from tqdm import tqdm
from pathlib import Path
from pyannote.core import Annotation, Segment
from pyannote.database.util import load_rttm


def sort_annotation(
        annot: Annotation
):
    """ Sort the annotation by start time
    Args
    ----
    annot: Annotation
        Annotation to sort

    Returns
    -------
    Annotation
        Sorted annotation
    """
    segments = sorted([(segment, label) for segment, label in annot.itertracks()], key=lambda x: x[0].start)
    annot = Annotation()
    for segment in segments:
        annot[segment[0]] = segment[1]
    return annot

def get_overlap(
        annot: Annotation
):
    """ Get the overlap percentage of the annotation
    May not handle well overlap of more than 2 speakers
    Args
    ----
    annot: Annotation
        Annotation to analyze

    Returns
    -------
    float
        Overlap duration
    """
    annot = sort_annotation(annot)
    duration = 0
    prev_segment = None
    for segment in annot.itersegments():
        # ignore the first one : 
        if prev_segment is None:
            prev_segment = segment
        elif segment.start < prev_segment.end:
            duration += min(segment.end, prev_segment.end) - segment.start
            prev_segment = segment
    return duration

def get_speaker_changes(
        annot: Annotation
):
    """ Get the number of speaker changes (per minute) in the annotation
    Args
    ----
    annot: Annotation
        Annotation to analyze

    Returns
    -------
    float
        Speaker changes per minute
    """
    annot = sort_annotation(annot)
    max_time = max([segment.end for segment in annot.itersegments()])
    
    segments = [Segment(i, i+60) for i in range(0, int(max_time), 60)]
    speaker_changes = 0

    for segment in segments:
        sub_annot = annot.crop(segment, mode="intersection")
        speaker_changes += len(sub_annot.labels()) - 1

    return speaker_changes

def analyze(
        data: str,
        rttm: str,
        audio: str
        ):
    """ Analyze the dataset using the annotations
    Analysis : speech%, overlapped%, #speakers min/max/avg, SC -> average speaker changes times per minute

    Args
    ----
    data: str
        Dataset to analyze
    rttm: str
        Path to the rttm file(s)
    audio: str
        Path to the audio file(s)

    Returns
    -------
    None
    """
    
    # Get all the rttm files from the path
    rttm_files = glob.glob(os.path.join(rttm, "**", "*.rttm"), recursive=True)
    print(f"Found {len(rttm_files)} file{'' if len(rttm_files) == 1 else 's'}")

    # Get all the audio files from the path
    audio_files = glob.glob(os.path.join(audio, "**", "*.wav"), recursive=True)
    print(f"Found {len(audio_files)} file{'' if len(audio_files) == 1 else 's'}")

    count = 0

    speech_duration = 0
    overlap_duration = 0
    total_duration = 0

    speaker_changes = 0

    min_speakers = 1000
    max_speakers = 0
    avg_speakers = 0


    for file in tqdm(rttm_files[:], desc="Loading rttm files"):
        rttm_dict = load_rttm(file)
        for key, annot in rttm_dict.items():
        
            speaker_changes += get_speaker_changes(annot)
                        
            timeline = annot.get_timeline().segmentation()
            speech_duration += timeline.duration()
            overlap_duration += get_overlap(annot)

            min_speakers = min(min_speakers, len(annot.labels()))
            max_speakers = max(max_speakers, len(annot.labels()))
            avg_speakers += len(annot.labels())

            # Fine in audio_files the one that corresponds to the key
            audio_file = [f for f in audio_files if key in f][0]
            audio_info = torchaudio.info(audio_file)
            audio_duration = audio_info.num_frames / audio_info.sample_rate
            torch.cuda.empty_cache()

            total_duration += audio_duration

    print(f"Overlap duration : {overlap_duration / total_duration}")
    print(f"Speech % : {speech_duration / total_duration}")
    print(f"Total duration : {total_duration / 3600} hours")

    print(f"Speaker changes : {speaker_changes / (total_duration / 60)} per minute")

    print(f"Min speakers : {min_speakers}")
    print(f"Max speakers : {max_speakers}")
    print(f"Avg speakers : {avg_speakers / len(rttm_files)}")

    results = {
        "overlap_duration": overlap_duration / total_duration,
        "speech_percentage": speech_duration / total_duration,
        "total_duration": total_duration / 3600,
        "speaker_changes": speaker_changes / (total_duration / 60),
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "avg_speakers": avg_speakers / len(rttm_files)
    }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute some analysis on any dataset using the annotations")
    parser.add_argument("--data", type=str, help="Dataset to analyze")
    parser.add_argument("--rttm", type=str, help="Path to the rttm file(s)")
    parser.add_argument("--audio", type=str, help="Path to the audio file(s)")

    args = parser.parse_args()
    assert args.data is not None, "Dataset is required"
    assert args.rttm is not None, "Path to the rttm file is required"
    assert args.audio is not None, "Path to the audio file is required"
    assert os.path.exists(args.rttm), "Path does not exist"
    assert os.path.exists(args.audio), "Path does not exist"

    print(f"Analyzing dataset {args.data} with rttm file {args.rttm}, audio file {args.audio}")

    res = analyze(args.data, args.rttm, args.audio)

    analysis_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "analysis")
    analysis_folder = os.path.join(analysis_folder, args.data)
    analysis_folder = os.path.abspath(analysis_folder)
    os.makedirs(analysis_folder, exist_ok=True)

    with open(os.path.join(analysis_folder, "analysis.yaml"), "w") as f:
        yaml.dump(res, f)