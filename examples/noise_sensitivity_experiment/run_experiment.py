from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm
from tqdm import tqdm
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd  # Import pandas for CSV file handling

def add_noise_to_audio(audio_file, noise_level, noisy_files_dir):
    signal, sample_rate = sf.read(audio_file)
    noise = np.random.normal(0, noise_level, signal.shape)
    noisy_signal = signal + noise
    noisy_audio_file = os.path.join(noisy_files_dir, f'noisy_audio_{noise_level}.wav')
    sf.write(noisy_audio_file, noisy_signal, sample_rate)
    return noisy_audio_file

def main():
    token = open(".tokens").read().split("=")[1].strip()
    audiofile = './audio.wav'
    labelfile = './label.rttm'
    noisy_files_dir = './noisy_files'
    stats_file = './stats.csv'

    if not os.path.exists(noisy_files_dir):
        os.makedirs(noisy_files_dir)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token)

    _, groundtruth = load_rttm(labelfile).popitem()

    metric = DiarizationErrorRate()
    noise_levels = np.linspace(0, 0.3, 15) 
    stats = []

    for noise_level in tqdm(noise_levels, desc='Adding Noise and Computing DER'):
        noisy_audiofile = add_noise_to_audio(audiofile, noise_level, noisy_files_dir)
        diarization = pipeline(noisy_audiofile)

        prediction_file = os.path.join(noisy_files_dir, f'prediction_with_noise_{noise_level}.rttm')
        with open(prediction_file, "w") as f:
            diarization.write_rttm(f)

        _, prediction = load_rttm(prediction_file).popitem()

        # Compute DER
        der = metric(groundtruth, prediction)
        stats.append({'Noise Level': noise_level, 'DER': der})
        print(f"Noise Level: {noise_level}, DER: {der * 100:.2f}%")

    # Convert stats to DataFrame and save as CSV
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(stats_file, index=False)

    # Plot DER vs. Noise Level
    plt.plot(df_stats['Noise Level'], df_stats['DER'], marker='o')
    plt.xlabel('Noise Level')
    plt.ylabel('DER')
    plt.title('DER vs. Noise Level')
    plt.grid(True)
    plt.savefig('./plots/der_vs_noise_level.png')
    plt.show()

if __name__ == "__main__":
    main()
