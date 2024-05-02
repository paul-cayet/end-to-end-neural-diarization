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
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, DiarizationPurity, DiarizationCoverage
import math
from typing import Tuple
from itertools import cycle


def plot_annotation(
        annotation: Annotation,
        ax: plt.Axes,
        y_offset: float = 0.1,
        figsize: Tuple[int, int] = (10, 3)
        ):
    """Plot the annotation on the given axis.
    Args
    ----
    annotation: Annotation
        The annotation to plot
    ax: plt.Axes
        The axis to plot the annotation on
    y_offset: float
        The offset between the different speakers
    figsize: Tuple[int, int]
        The size of the figure
    
    Returns
    -------
    None
    """

    if not annotation:
        ax.set_xlim(0, 0)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel('Time')
        return

    # colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])*
    colors = cycle([
        '#1abc9c',  # Turquoise
        '#2ecc71',  # Emerald
        '#3498db',  # Peter River
        '#9b59b6',  # Amethyst
        '#34495e',  # Wet Asphalt
        '#f1c40f',  # Sun Flower
        '#e67e22',  # Carrot
    ])
    
    # Map speakers to colors and y-positions
    speaker_info = {}
    
    current_y_position = 1
    
    for segment, _, label in annotation.itertracks(yield_label=True):
        if label not in speaker_info:
            speaker_info[label] = {
                'color': next(colors),
                'y_position': current_y_position
            }
            # Update the y-position next speaker
            current_y_position += y_offset
        
        # Use the assigned color and y-position for this speaker
        color = speaker_info[label]['color']
        y_position = speaker_info[label]['y_position']
        ax.plot([segment.start, segment.end], [y_position, y_position], label=label, color=color, marker='|', markersize=15, linestyle='-', linewidth=2)
    
    ax.set_ylim(0.5, current_y_position)
    starts_lst = [segment.start for segment, _ in annotation.itertracks()]
    if not starts_lst:
        starts_lst = [0]
    ends_lst = [segment.end for segment, _ in annotation.itertracks()]
    if not ends_lst:
        ends_lst = [3600]
    ax.set_xlim(min(starts_lst), max(ends_lst))

    ax.set_yticks([])
    ax.set_xlabel('Time')
    
    custom_legends = [plt.Line2D([0], [0], color=info['color'], lw=4) for info in speaker_info.values()]
    ax.legend(custom_legends, speaker_info.keys(), loc='upper right')


def compute_metrics(
        prediction: Annotation,
        groundtruth: Annotation
        ):
    """ Compute the diarization error rate, number of speakers, purity and coverage metrics.
    Args
    ----
    prediction: Annotation
        The predicted annotation
    groundtruth: Annotation
        The groundtruth annotation
    
    Returns
    -------
    Tuple[float, int, int, float, float]
        The diarization error rate, number of speakers in the prediction, number of speakers in the groundtruth, purity and coverage
    """

    # DER
    metric_der = DiarizationErrorRate()
    der = metric_der(groundtruth, prediction)

    # Speakers
    num_speakers_prediction = len(set([spk for _, _, spk in prediction.itertracks(yield_label=True)]))
    num_speakers_groundtruth = len(set([spk for _, _, spk in groundtruth.itertracks(yield_label=True)]))

    # Overlap
    metric_purity = DiarizationPurity()
    purity = metric_purity(groundtruth, prediction)
    metric_coverage = DiarizationCoverage()
    coverage = metric_coverage(groundtruth, prediction)

    return der, num_speakers_prediction, num_speakers_groundtruth, purity, coverage


def split_by_duration(
        annotation: Annotation, 
        duration: float=10*60,
        nb_parts: int=6
        ) -> list:
    """Split the annotation into intervals of the given duration.
    Args
    ----
    annotation: Annotation
        The annotation to split
    duration: float
        The duration of each interval
    nb_parts: int
        The number of intervals to split the annotation into

    Returns
    -------
    list
        A list of annotations, each corresponding to an interval
    """

    interval_annotations = []

    for _ in range(nb_parts):
        # from i*duration to (i+1)*duration except the last one : from 5*duration to the end
        interval_annotations.append(Annotation())

    for segment, track, label in annotation.itertracks(yield_label=True):
        
        # Get the start time of the interval the current segment is in
        start_interval  = int(segment.start//duration)
        start_interval = min(start_interval, nb_parts-1)

        end_interval  = int(math.ceil(segment.end / duration))
        end_interval = min(end_interval, nb_parts)

        for interval_index in range(start_interval, end_interval):
            interval_start = interval_index * duration
            interval_end = (interval_index + 1) * duration
            
            if interval_end<(nb_parts-1):
                clipped_segment = Segment(max(segment.start, interval_start), min(segment.end, interval_end))
            else:
                clipped_segment = Segment(max(segment.start, interval_start), segment.end)

            # Add the clipped segment to the corresponding interval annotation
            clipped_interval_index = min(interval_index, nb_parts-1)

            interval_annotations[clipped_interval_index][clipped_segment, track] = label

    return interval_annotations