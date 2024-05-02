import os
from tqdm import tqdm
from typing import List
import json
import glob


def cut_rttm(rttm_files: str, output_dir: str):
    """Cuts rttm annotations from 1 hour to 60 seconds annotations
    (makes loading time ~15x faster).
    
    Args
    ----
    rttm_files: str
        Path to parent directory containing the rttm annotation files
    output_dir: str
        Path to the parent directory to write the modified rttm annotation files
    """
    for file in tqdm(os.listdir(rttm_files)):
        rttm_file = os.path.join(rttm_files, file)

        with open(rttm_file, "r") as f:
            lines = f.readlines()

        modified_annotations = []

        for line in lines:
            fields = line.strip().split()
            filename = fields[1]
            start_time = float(fields[3])
            duration = float(fields[4])
            speaker_id = fields[7]

            # Calculate the new filename and start time
            minute_index = int(start_time // 60)
            new_filename = f"{os.path.splitext(filename)[0]}_{minute_index:03d}"
            new_start_time = start_time % 60

            # Handle annotations spanning multiple files
            while duration > 0:
                remaining_duration = 60 - new_start_time
                if duration <= remaining_duration:
                    new_duration = duration
                else:
                    new_duration = remaining_duration

                # Create the modified annotation line
                modified_line = f"SPEAKER {new_filename} 1 {new_start_time:.2f} {new_duration} <NA> <NA> {speaker_id} <NA> <NA>\n"
                modified_annotations.append(modified_line)

                duration -= new_duration
                minute_index += 1
                new_filename = f"{os.path.splitext(filename)[0]}_{minute_index:03d}"
                new_start_time = 0

        output_rttm_file = os.path.join(output_dir, str(file))
        with open(output_rttm_file, "w") as f:
            f.writelines(modified_annotations)


def generate_train_test_split(
        rttm_dirpath: str, 
        output_filepath: str, 
        test_rttm_glob_pattern: str, 
        all_rttm_glob_pattern: str
    ):
    """Generate a Json file containing the train/test split configuration.
    
    Args
    ----
    rttm_dirpath: str
        Path to parent directory containing the rttm annotation files (in our case the 60 second versions).
    output_filepath: str
        Path to the desired train/test split Json configuration file.
    test_rttm_glob_pattern: str
        Specifier for the test files.
    all_rttm_glob_pattern: str
        Specifier for the annotation files.
    """

    all_rttms = glob.glob(os.path.join(rttm_dirpath, all_rttm_glob_pattern))
    all_test_rttms = glob.glob(os.path.join(rttm_dirpath, test_rttm_glob_pattern))

    mapping = {'val': all_test_rttms}
    mapping['train'] = [x for x in all_rttms if x not in mapping['val']]

    # output_filepath = os.path.join(output_dir, 'train_test_split.json')
    with open(output_filepath, 'w') as f:
        json.dump(mapping, f, indent=2)


def get_train_test_split(filename: str):
    with open(filename) as f:
        mapping = json.load(f)
        train_files = mapping['train']
        val_files = mapping['val']
    return train_files, val_files

        
def concat_files(file_list: List[str], output_file: str):
    with open(output_file, 'w') as outfile:
        for fname in file_list:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
            outfile.write('\n')


def generate_lst_files(input_rttm_path: str, output_lst_path: str):
    """Generate lst annotation files containing the audio files URIs.

    Args
    ----
    input_rttm_path: str
        Path to the file containing the concatenated rttm annotations
    output_lst_path: str
        Path to the desired output lst annotation file
    """
    unique_filenames = set()

    with open(input_rttm_path, 'r') as file:
        for line in file:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                filename = parts[1]
                unique_filenames.add(filename)

    with open(output_lst_path, 'w') as file:
        for filename in unique_filenames:
            file.write(filename + '\n')


def generate_uem_files(rttm_file_path: str, uem_file_path: str):
    """Generate uem annotation files containing the audio files start time and duration.

    Args
    ----
    rttm_file_path: str
        Path to the file containing the concatenated rttm annotations
    uem_file_path: str
        Path to the desired output uem annotation file
    """
    file_timestamps = {}

    with open(rttm_file_path, 'r') as file:
        for line in file:
            if line.startswith("SPEAKER"):
                parts = line.strip().split()
                filename = parts[1]
                start_time = float(parts[3])
                end_time = start_time + float(parts[4])

                if filename not in file_timestamps:
                    file_timestamps[filename] = [start_time, end_time]
                else:
                    file_timestamps[filename][0] = min(file_timestamps[filename][0], start_time)
                    file_timestamps[filename][1] = max(file_timestamps[filename][1], end_time)

    with open(uem_file_path, 'w') as file:
        for filename, timestamps in file_timestamps.items():
            file.write(f"{filename} NA {timestamps[0]:.3f} {timestamps[1]:.3f}\n")


def generate_annotations(
        rttm_files: str,
        output_dir: str,
        train_test_split_path: str,
        output_annotation_dirpath: str,
        test_rttm_glob_pattern: str='130612FR2*.rttm',
        all_rttm_glob_pattern: str='*.rttm',
    ):
    """Generates the different annotation files needed by Pyannote to
    fine-tune a speaker segmentation model.
    
    Args
    ----
    rttm_files: str
        Path to parent directory containing the rttm annotation files
    output_dir: str
        Path to the parent directory to write the modified rttm annotation files
    train_test_split_path: str
        Path to the desired train/test split Json configuration file.
    output_annotation_dirpath: str
        Path to the 'database' configuration file
    test_rttm_glob_pattern: str
        Specifier for the test files.
    all_rttm_glob_pattern: str
        Specifier for the annotation files.
    """
    rttm_train_file_path = output_annotation_dirpath+'few.train.rttm'
    rttm_val_file_path = output_annotation_dirpath+'few.val.rttm'
    lst_train_file_path = output_annotation_dirpath+'filelist_train.lst'
    lst_val_file_path = output_annotation_dirpath+'filelist_val.lst'
    uem_train_file_path = output_annotation_dirpath+'train.uem'
    uem_val_file_path = output_annotation_dirpath+'val.uem'
    
    print("Starting annotation generation from RTTM files...")
    os.makedirs(output_dir, exist_ok=True) # make new dir if it doesn't already exist
    os.makedirs(os.path.dirname(train_test_split_path), exist_ok=True)
    os.makedirs(output_annotation_dirpath, exist_ok=True)
   
    # convert 60 minutes annotations to 60 second annotations.
    cut_rttm(rttm_files, output_dir)

    generate_train_test_split(output_dir, train_test_split_path, test_rttm_glob_pattern, all_rttm_glob_pattern)
    train_files, val_files = get_train_test_split(train_test_split_path)

    # Generate RTTM files
    concat_files(train_files, rttm_train_file_path)
    concat_files(val_files, rttm_val_file_path)
    print('Generated new RTTM files')


    # Generate LST files
    generate_lst_files(rttm_train_file_path, lst_train_file_path)
    generate_lst_files(rttm_val_file_path, lst_val_file_path)
    print('Generated LST files')

    # Generate UEM files
    generate_uem_files(rttm_train_file_path, uem_train_file_path)
    generate_uem_files(rttm_val_file_path, uem_val_file_path)
    print('Generated UEM files')
    print("Annotation generation from RTTM files conversion completed ✅")

