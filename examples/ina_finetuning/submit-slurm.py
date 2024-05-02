#!/usr/bin/python

import os
import sys
import subprocess
import tempfile

def get_available_nodes():
    """Returns a list of available nodes within the specified ranges."""
    print("Checking for available nodes...")
    try:
        # Execute squeue and get the current jobs
        result = subprocess.run(['squeue', '--format=%N'], capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully retrieved node information.")
            # Parse the output to get a list of nodes currently in use
            used_nodes = result.stdout.split()
            print(f"Currently used nodes: {used_nodes}")
            all_nodes = [f'sh0{i}' for i in range(1, 7)] + ['sh08'] #+ [f'sh2{i}' for i in range(0, 3)]
            # Determine available nodes by removing used nodes from the all_nodes list
            available_nodes = [node for node in all_nodes if node not in used_nodes]
            print(f"Available nodes: {available_nodes}")
            return available_nodes
        else:
            print("Failed to get node information")
            return []
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def makejob(commit_id, configpath, nruns, available_nodes = None):
    if available_nodes:
        node = available_nodes[0]  # Select the first available node for simplicity
        print(f"Selected node for job: {node}")
        pre_job_script = f"""#!/bin/bash

#SBATCH --job-name=DiDia
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}
#SBATCH --nodelist={node}"""
    else:
        pre_job_script = f"""#!/bin/bash
        
#SBATCH --job-name=DiDia
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=48:00:00
#SBATCH --output=/usr/users/cei2023_2024_dinum_diarization/cayet_pau/cei_dinum/dinum_diarization/logslurms/slurm-%A_%a.out
#SBATCH --error=/usr/users/cei2023_2024_dinum_diarization/cayet_pau/cei_dinum/dinum_diarization/logslurms/slurm-%A_%a.err
#SBATCH --array=1-{nruns}"""
     

    return pre_job_script + f"""
current_dir=`pwd`
export PATH=$PATH:~/.local/bin

echo "Session " ${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}

echo "Running on " $(hostname)

echo "Copying the source directory and data"
date
mkdir $TMPDIR/code
mkdir $TMPDIR/code/data
rsync -r --exclude .git --exclude wandb --exclude temp_rttm_folder --exclude analysis --exclude data --exclude lightning_logs --exclude logslurms --exclude notebooks --exclude notebook --exclude labels --exclude docs --exclude data_msdwild  --exclude checkpoint --exclude venv . $TMPDIR/code
# rsync -r --exclude "*.mp4" /mounts/Datasets3/2024-Diarization/INA_Snowden/medias_reencoded/tv/ $TMPDIR/code/data
# rsync -r --include '*/' --include '*130612FR22100_B.MPG.wav' --include "*130607FR20100_B.MPG.wav" --exclude '*' /mounts/Datasets3/2024-Diarization/INA_Snowden/medias_reencoded/tv/ $TMPDIR/code/data

input_dir="/mounts/Datasets3/2024-Diarization/INA_Snowden/medias_reencoded/tv"
output_dir=$TMPDIR/code/data

for folder in "$input_dir"/*; do
    echo $folder
    for folder in "$folder"/*; do
        echo $folder
        for audio_file in "$folder"/*.wav; do
            filename=$(basename "$audio_file" .MPG.wav)
            if [ -f "$audio_file" ]; then
            # if [[ "$filename" == "130612FR22100_B" || "$filename" == "130607FR20100_B" ]]; then
                echo $filename
                ffmpeg -i "$audio_file" -f segment -segment_time 60 -c copy "$output_dir/${{filename}}_%03d.MPG.wav"
            fi
        done
    done
done

# echo "Checking out the correct version of the code commit_id {commit_id}"
cd $TMPDIR/code
ls 
# git checkout {commit_id}


echo "Setting up the virtual environment"
# python3 -m pip install virtualenv --user
# virtualenv -p python3 venv

python3 -m venv venv
source venv/bin/activate

# Install the library
echo "Installing the project library"
python -m pip install -r requirements.txt
sed -i 's/mode="raise"/mode="pad"/g' ./venv/lib/python3.8/site-packages/pyannote/audio/core/io.py

#
rm -rf ./venv/lib/python3.8/site-packages/pyannote/audio/utils/permutation.py
cp /usr/users/cei2023_2024_dinum_diarization/ibanez_nic/dinum_diarization/venv/lib/python3.8/site-packages/pyannote/audio/utils/permutation.py ./venv/lib/python3.8/site-packages/pyannote/audio/utils/permutation.py

pip install --upgrade pip
pip install -e .

# source /usr/users/cei2023_2024_dinum_diarization/ibanez_nic/dinum_diarization/venv/bin/activate



echo "Pip list :"
pip list

echo "Modifying database config file"
python3 -u examples/ina_finetuning/modify_database_config.py -c {configpath}

echo "Training"
python3 src/dinum_diarization/finetune/finetune.py -c {configpath}

# tail -f /dev/null

if [[ $? != 0 ]]; then
    exit -1
fi
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# # Ensure all the modified files have been staged and commited
# # This is to guarantee that the commit id is a reliable certificate
# # of the version of the code you want to evaluate
# result = int(
#     subprocess.run(
#         "expr $(git diff --name-only | wc -l) + $(git diff --name-only --cached | wc -l)",
#         shell=True,
#         stdout=subprocess.PIPE,
#     ).stdout.decode()
# )
# if result > 0:
#     print(f"We found {result} modifications either not staged or not commited")
#     raise RuntimeError(
#         "You must stage and commit every modification before submission "
#     )

# commit_id = subprocess.check_output(
#     "git log --pretty=format:'%H' -n 1", shell=True
# ).decode()

# print(f"I will be using the commit id {commit_id}")
commit_id = None

# Ensure the log directory exists
os.system("mkdir -p logslurms")

# if len(sys.argv) not in [2, 3]:
#     print(f"Usage : {sys.argv[0]} config.yaml <nruns|1>")
#     sys.exit(-1)

configpath = sys.argv[1]
if len(sys.argv) == 2:
    nruns = 1
else:
    nruns = int(sys.argv[2])

# Copy the config in a temporary config file
os.system("mkdir -p configs")
tmp_configfilepath = tempfile.mkstemp(dir="./configs", suffix="-config.yml")[1]
os.system(f"cp {configpath} {tmp_configfilepath}")

# Launch the batch jobs
available_nodes = get_available_nodes()
submit_job(makejob(commit_id, tmp_configfilepath, nruns, available_nodes=None))
# submit_job(makejob(commit_id, tmp_configfilepath, nruns, available_nodes=available_nodes))
os.remove("job.sbatch")
