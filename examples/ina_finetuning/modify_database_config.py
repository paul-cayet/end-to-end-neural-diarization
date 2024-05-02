import yaml
import argparse
from pathlib import Path
import os


def generate_database_config(audio_path, annotation_path, path_yml):
    with open(path_yml, "w") as f:
        f.write(yaml.dump({
        "Databases": {
            "Mydata_inabase": f"{audio_path}/{{uri}}.MPG.wav"
        },
            
        "Protocols": {
            "Mydata_inabase": {
            "SpeakerDiarization": {
            "MyProtocol": {
                "scope": "file",
                "train":
                    {"uri": f"{annotation_path}/filelist_train.lst",
                    "annotation": f"{annotation_path}/few.train.rttm",
                    "annotated": f"{annotation_path}/train.uem"},
                "development":{
                    "uri": f"{annotation_path}/filelist_val.lst",
                    "annotation": f"{annotation_path}/few.val.rttm",
                    "annotated": f"{annotation_path}/val.uem"},
        }}}}
    })
    )
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--global_config_path", default="config.yml")
    args = parser.parse_args()
    global_config_path = args.global_config_path

    # get local repo folder path
    current_repo_folder = Path(__file__).resolve().parents[2]

    # get general information
    general_config = yaml.safe_load(open(global_config_path,'r'))
    database_config_name = general_config['dataset']['db_config_path']
    relative_annotation_path = general_config['dataset']['annotation_path']

    # re-generate correct paths
    database_config_path = os.path.join(current_repo_folder,database_config_name)
    audio_path = os.path.join(current_repo_folder,'data')
    annotation_path = os.path.join(current_repo_folder,relative_annotation_path)

    print(f'database_config_path: {database_config_path}')
    print(f'audio_path: {audio_path}')
    print(f'annotation_path: {annotation_path}')
    generate_database_config(audio_path, annotation_path, database_config_path)


    # database_config = yaml.safe_load(open(database_config_path,"r"))
