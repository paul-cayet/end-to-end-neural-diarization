from pyannote.audio import Pipeline
import matplotlib.pyplot as plt
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
import os
import time
import glob
import argparse
import dotenv
from typing import List

dotenv.load_dotenv(dotenv.find_dotenv())

def benchmark(
        benchmark_folder: str,
        files_to_benchmark: List[str],
        device: torch.device
        ):
    """Benchmark the CPU vs GPU for the audio files inference
    Args:
    -------
    benchmark_folder: str
        Path to the folder where the results will be saved
    files_to_benchmark: List[str]
        List of the audio files to benchmark
    device: torch.device
        Device to use for the inference

    Returns:
    -------
    Y: List[float]
        List of the time to do the inference for each audio file
    X: List[float]
        List of the duration of each audio file
    """

    Y = []
    X = []

    token = os.getenv("HF_TOKEN")
    assert token is not None, "HF_TOKEN is not defined (HuggingFace token required to download the model)"

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token).to(device)
    
    if os.path.exists(f"{benchmark_folder}/CPUvsGPU_{device}_Y.npy") and os.path.exists(f"{benchmark_folder}/CPUvsGPU_{device}_X.npy"):
        X, Y = np.load(f"{benchmark_folder}/CPUvsGPU_{device}_X.npy").tolist(), np.load(f"{benchmark_folder}/CPUvsGPU_{device}_Y.npy").tolist()
        if len(X) == len(files_to_benchmark):
            return Y, X
        
    for file in tqdm(files_to_benchmark):
        data, sample_rate = torchaudio.load(file)
        duration = data.size(1)/sample_rate
        
        y1 = time.time() 
        diarization = pipeline(file)
        y2 = time.time()
        Y.append(y2 - y1)
        X.append(duration)

    # Save the results
    np.save(f"{benchmark_folder}/CPUvsGPU_{device}_Y.npy", Y)
    np.save(f"{benchmark_folder}/CPUvsGPU_{device}_X.npy", X)

    return Y, X

def main(
        path: str, 
        num_files: int = 100
        ):
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "CUDA not available"
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    devices = [torch.device("cpu"), torch.device("cuda")]

    files_to_benchmark = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    np.random.shuffle(files_to_benchmark)
    files_to_benchmark = files_to_benchmark[:num_files]

    benchmark_folder = os.path.relpath("../../../results/benchmark")
    benchmark_folder = os.path.abspath(benchmark_folder)
    os.makedirs(benchmark_folder, exist_ok=True)
    file_path = os.path.join(benchmark_folder, "loglog_CPUvsGPU.png")

    print(f"Results will be saved in {benchmark_folder}")

    plt.figure()

    for device in devices:
        print(f"Device: {device}")
        Y, X = benchmark(benchmark_folder, files_to_benchmark, device)

        plt.scatter(X, Y, label='CPU' if device == torch.device("cpu") else 'CUDA')

        
    plt.xlabel('Duration of the audio file (s)')
    plt.ylabel('Time (s)')
    plt.title('Time to do the inference for the audio files')

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

    plt.gcf().set_size_inches(10, 10)

    plt.savefig(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU for audio file inference')
    parser.add_argument('--audio', type=str, default='', help='Path to the audio files')
    parser.add_argument('--n', type=int, default=100, help='Number of files to benchmark')
    args = parser.parse_args()
    
    assert args.path != '', "Path to the audio files is required"
    assert os.path.exists(args.path), "Path does not exist"

    path = args.audio
    num_files = args.n

    print(f"Path to audio files: {path}")
    print(f"Number of files to benchmark: {num_files}")
    
    main(path, num_files)