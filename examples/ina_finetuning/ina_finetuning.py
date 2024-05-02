from neural_diarization.labelling import convert_sd_files, generate_annotations
import dotenv
import os
import shutil
from pathlib import Path

dotenv.load_dotenv(dotenv.find_dotenv())

if __name__=="__main__":

    CONVERT_SD_FILES = False
    GENERATE_ANNOTATIONS = False


    sd_dirname = "folder/Metadata2016/speaker_diarization/INA"
    uncut_rttm_parent_dirname = "folder/.../temp_rttm_folder"

    uncut_rttm_dirname = uncut_rttm_parent_dirname + '/converted/'
    cut_rttm_output_dir = "folder/.../temp_rttm_folder/new_converted"
    train_test_split_path = 'folder/.../data_ina/train_test_split.json'
    output_dirpath_parent = os.path.dirname(train_test_split_path)
    output_dirpath = 'folder/.../data_ina/data_annot_finetuning/'
    test_rttm_glob_pattern ='130612FR2*.rttm'
    all_rttm_glob_pattern ='*.rttm'
    
    finetune_config_path = 'folder/.../config.yml'

    database_config_name = 'MyDatabase-small.yml'

    split_audio_data_dirname = 'folder/.../data'
    model_checkpoint_dirname = 'folder/.../checkpoint'

    # Convert SD files into RTTMS
    if CONVERT_SD_FILES:
        convert_sd_files(sd_dirname, uncut_rttm_parent_dirname)

    # Generate Annotations (rttm, uem, lst) files
    if GENERATE_ANNOTATIONS:
        generate_annotations(
            uncut_rttm_dirname,
            cut_rttm_output_dir,
            train_test_split_path,
            output_dirpath,
            test_rttm_glob_pattern,
            all_rttm_glob_pattern
        )

    # Moving Database config to data_ina folder
    current_dirname = Path(__file__).resolve().parent
    shutil.copyfile(
        os.path.join(current_dirname,database_config_name),
        os.path.join(output_dirpath_parent,database_config_name)
    )

