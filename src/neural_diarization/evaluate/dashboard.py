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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
from pathlib import Path


def plot_der_vs_num_speakers(
        df: pd.DataFrame,
        column_name: str, 
        title: str
        ):
    """ Plot the DER vs the number of speakers
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics
    column_name: str
        Name of the column to use for the x-axis
    title: str
        Title of the plot
        
    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    # Aggregate the data:
    agg_df = df.groupby(column_name).agg({
        "der": ["mean", "median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        column_name: "count",
    })

    agg_df.columns = ["der_mean", "der_median", "der_std", "der_q1", "der_q3", "count"]

    agg_df = agg_df.sort_index()

    flat_colors = sns.color_palette("muted")

    fig = plt.figure()

    # Line plot for mean DER
    plt.plot(agg_df.index, agg_df["der_mean"], label="Mean DER", alpha=0.3, color=flat_colors[0])
    plt.plot(agg_df.index, agg_df["der_median"], label="Median DER", alpha=0.8, color=flat_colors[1])
    plt.plot(agg_df.index, agg_df["der_q1"], label="Q1 DER", linestyle="--", alpha=0.8, color=flat_colors[2])
    plt.plot(agg_df.index, agg_df["der_q3"], label="Q3 DER", linestyle="--", alpha=0.8, color=flat_colors[2])

    plt.xlabel("Number of speakers")
    plt.ylabel("DER")
    plt.title(title)

    # Crop from 0 to 2:
    plt.ylim(0, 3)

    plt.legend()

    return fig

def plot_distribution(
        df: pd.DataFrame,
        column_name: str,
        title: str
        ):
    """Plot the distribution of a column
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics
    column_name: str
        Name of the column to use for the x-axis
    title: str
        Title of the plot
    
    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True, color="skyblue", bins=30)

    plt.xlabel(f"{column_name}", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(title, fontsize=16)

    plt.tight_layout()
    return fig

def plot_der_vs_num_speakers_gt(
        df: pd.DataFrame,
        ):
    """Plot the DER vs the number of speakers (ground truth)
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics

    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    return plot_der_vs_num_speakers(df, "num_speakers_gt", "DER vs Number of speakers (ground truth)")

def plot_der_vs_num_speakers_pred(
        df: pd.DataFrame,
        ):
    """Plot the DER vs the number of speakers (predicted)
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics
    
    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    return plot_der_vs_num_speakers(df, "num_speakers_pred", "DER vs Number of speakers (predicted)")

def plot_der_distribution(
        df: pd.DataFrame, 
        ):
    """Plot the distribution of the DER
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics

    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    filtered_df = df[df["der"] < df["der"].quantile(0.99)]
    return plot_distribution(filtered_df, "der", "DER Distribution")

def plot_num_speakers_gt_distribution(
        df: pd.DataFrame,
        ):
    """Plot the distribution of the number of speakers (ground truth)
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics

    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    return plot_distribution(df, "num_speakers_gt", "Number of speakers (ground truth) Distribution")

def plot_num_speakers_pred_distribution(
        df: pd.DataFrame,
        ):
    """Plot the distribution of the number of speakers (predicted)
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics

    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """
    return plot_distribution(df, "num_speakers_pred", "Number of speakers (predicted) Distribution")


def plot_der_vs_timeofday(
        df: pd.DataFrame
        ):
    """Plot the DER vs the time of day
    Args
    ----
    df: pd.DataFrame
        DataFrame containing the evaluation metrics
        
    Returns
    -------
    fig: plt.Figure
        Figure of the plot
    """

    # Add a column timeofday
    df["timeofday"] = df["file_name"].apply(lambda x: int(x[9:11]) + int(x[-1])*(1/6))

    # Aggregate the data:
    agg_df = df.groupby("timeofday").agg({
        "der": ["mean", "median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        "timeofday": "count",
    })

    agg_df.columns = ["der_mean", "der_median", "der_std", "der_q1", "der_q3", "count"]

    agg_df = agg_df.sort_index()

    flat_colors = sns.color_palette("muted")

    fig = plt.figure(figsize=(10, 6))

    # Line plot for mean DER
    plt.plot(agg_df.index, agg_df["der_mean"], label="Mean DER", alpha=0.3, color = flat_colors[0])
    plt.plot(agg_df.index, agg_df["der_median"], label="Median DER", alpha=0.8, color = flat_colors[1])
    plt.plot(agg_df.index, agg_df["der_q1"], label="Q1 DER", linestyle="--", alpha=0.8, color = flat_colors[2])
    plt.plot(agg_df.index, agg_df["der_q3"], label="Q3 DER", linestyle="--", alpha=0.8, color = flat_colors[2])


    plt.xlabel("Time of day")
    plt.ylabel("DER")
    plt.title("DER vs Time of day")

    plt.ylim(0, 3)

    plt.legend()

    return fig


def dashboard(
        dataset: str,
        path: str
        ):
    """Plot the evaluation metrics
    Args
    ----
    dataset: str
        Name of the dataset
    path: str
        Path to the evaluation file

    Returns
    -------
    None
    """
    # Evaluation file is a csv file
    df = pd.read_csv(path)

    plots_fn = {
        "DER_vs_num_speakers_gt": plot_der_vs_num_speakers_gt,
        "DER_vs_num_speakers_pred": plot_der_vs_num_speakers_pred,
        "DER_distribution": plot_der_distribution,
        "num_speakers_gt_distribution": plot_num_speakers_gt_distribution,
        "num_speakers_pred_distribution": plot_num_speakers_pred_distribution,
        # "DER_vs_timeofday": plot_der_vs_timeofday,
    }

    plt_folder = os.path.join(Path(__file__).resolve().parents[3], "results", "plots")
    plt_folder = os.path.join(plt_folder, dataset)
    plt_folder = os.path.abspath(plt_folder)
    os.makedirs(plt_folder, exist_ok=True)

    # DER_vs_timeofday is specific to INA dataset : 
    # if INA not in the path, remove the plot
    # if "ina" not in dataset.lower():
    #     plots_fn.pop("DER_vs_timeofday")
    #     print(f"INA not in the path, removing DER_vs_timeofday plot")

    for key, plot_fn in plots_fn.items():
        fig = plot_fn(df)
        fig.savefig(os.path.join(plt_folder, f"{key}.png"), dpi=300)
        print(f"Saved plot in {os.path.join(plt_folder, f'{key}.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot evaluation metrics')
    parser.add_argument("--data", type=str, help="Dataset to plot")
    parser.add_argument('--file', type=str, default='', help='Path to the evaluation file (csv DataFrame)')
    args = parser.parse_args()

    assert args.data is not None, "Dataset is required"
    assert args.file != '', "Path to the evaluation file is required"
    assert os.path.exists(args.file), "Path does not exist"
    print(f"Path to evaluation file: {args.file}")

    dashboard(args.data, args.file)