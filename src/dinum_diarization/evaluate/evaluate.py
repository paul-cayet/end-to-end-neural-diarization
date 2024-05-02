from utils import split_by_duration, compute_metrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from pyannote.database.util import load_rttm

import pandas as pd
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import glob


def evaluate(
        dataset_name: str,
        labels_folder: str,
        chunks: int = None
        ):
    """Evaluate the diarization results of a dataset
    Args
    ----
    dataset_name: str
        The name of the dataset
    labels_folder: str
        The directory containing the labels
    chunks: int
        Number of chunks to split the audio files into
        
    Returns
    -------
    None
    """
    # Directory for the predictions
    pred_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "predictions")
    pred_folder = os.path.join(pred_folder, dataset_name)
    pred_folder = os.path.abspath(pred_folder)
    os.makedirs(pred_folder, exist_ok=True)
    assert os.path.exists(pred_folder), "Predictions directory not found"

    print(f"Getting files from {pred_folder}")
    predictions_rttm = glob.glob(os.path.join(pred_folder, "**", "*.rttm"), recursive=True)
    print(f"Found {len(predictions_rttm)} file{'' if len(predictions_rttm) == 1 else 's'}")
    predictions_dic = {}
    for rttm in predictions_rttm:
        predictions_dic = {**predictions_dic, **load_rttm(rttm)}

    print(f'Getting files from {labels_folder}')
    labels_rttm = glob.glob(os.path.join(labels_folder, "**", "*.rttm"), recursive=True)
    print(f"Found {len(labels_rttm)} file{'' if len(labels_rttm) == 1 else 's'}")
    labels_dic = {}
    for rttm in labels_rttm:
        labels_dic = {**labels_dic, **load_rttm(rttm)}

    plt_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "plots")
    plt_folder = os.path.join(plt_folder, dataset_name)
    plt_folder = os.path.abspath(plt_folder)
    os.makedirs(plt_folder, exist_ok=True)

    eval_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "evaluations")
    eval_folder = os.path.abspath(eval_folder)
    os.makedirs(eval_folder, exist_ok=True)

    df = None

    if os.path.exists(os.path.join(eval_folder, dataset_name + ".csv")):
        df = pd.read_csv(os.path.join(eval_folder, dataset_name + ".csv"))
    else:
        df = pd.DataFrame(
            columns=[
                'file_name', 
                'der', 
                'num_speakers_pred', 
                'num_speakers_gt', 
                'purity', 
                'coverage'
            ]
        )

    progress = tqdm(predictions_dic.items(), desc='Analyzing files')

    for (prediction_name, prediction) in progress:
        if prediction_name in labels_dic.keys():
            print(f"Comparing prediction and groundtruth for {prediction_name}")

            progress.set_postfix(file=prediction_name)
            groundtruth = labels_dic[prediction_name]

            if chunks:
                separated_preds, separated_gts = split_by_duration(prediction, 10*60, chunks), split_by_duration(groundtruth, 10*60, chunks)

                for i, (separated_pred, separated_gt) in enumerate(zip(separated_preds, separated_gts)):
                    der, num_speakers_pred, num_speakers_gt, purity, coverage = compute_metrics(separated_pred, separated_gt)

                    if prediction_name + f'_part{i}' in df['file_name'].values:
                        df = df[df['file_name'] != prediction_name + f'_part{i}']
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    'file_name': [prediction_name + f'_part{i}'],
                                    'der': [der],
                                    'num_speakers_pred': [num_speakers_pred],
                                    'num_speakers_gt': [num_speakers_gt],
                                    'purity': [purity],
                                    'coverage': [coverage]
                                }
                            )
                        ]
                    )

            else:
                der, num_speakers_pred, num_speakers_gt, purity, coverage = compute_metrics(prediction, groundtruth)

                if prediction_name in df['file_name'].values:
                    df = df[df['file_name'] != prediction_name]

                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                'file_name': [prediction_name],
                                'der': [der],
                                'num_speakers_pred': [num_speakers_pred],
                                'num_speakers_gt': [num_speakers_gt],
                                'purity': [purity],
                                'coverage': [coverage]
                            }
                        )
                    ]
                )
        else:
            print(f"No label found for {prediction_name}")

    # Summary
    desc_df = df.describe()
    print(desc_df)

    desc_df.to_csv(os.path.join(eval_folder, dataset_name +"_summary.csv"))
    df.to_csv(os.path.join(eval_folder, dataset_name +".csv"), index=False)

    print(f"Results saved in {eval_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which dataset to predict")
    parser.add_argument("--data", type=str, help="The dataset to predict")
    parser.add_argument("--labels", type=str, help="The directory containing the labels")
    parser.add_argument("--chunks", type=int, default=None, help="Number of chunks to split the audio files into")
    args = parser.parse_args()

    assert args.data, "No dataset provided (--data)"
    assert os.path.exists(args.labels), "Labels directory not found"
    
    print(f"Dataset: {args.data}")
    print(f"Labels directory: {args.labels}")

    evaluate(args.data, args.labels, args.chunks)