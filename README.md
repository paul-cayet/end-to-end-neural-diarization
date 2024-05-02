# End to end Neural Diarization

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This code package implements tools to finetune and evaluate end-to-end neural speaker diarization models.

The code is part of a end-of-studies project. The associated report can be found here: ...


# TODO:
- Finalize review for following codes
    - Utils ✅ (generate_database_config should not be in utils/io.py -> is in examples)
    - labeling ✅ (removed gen_subset.py)
    - evaluate ✅
    - finetune ✅ (generate_database_config should not be in finetune/helpers.py -> is in examples)
- Add information in readme ✅
- Clean requirements.txt ✅
- Do something about documentation générale ✅ (deleted, paper reviews put in a folder)
- Do something about analysis ✅ (deleted, added experiment to examples)
- Clean examples ✅
- Remove data ina if not needed ✅ (deleted)
- Remove notebooks ✅
- Clean gitignore ✅
- Add license to src files ✅
- Clean pyproject ✅

Note: @nicolas check dashboard l256-260

# Usage

## Getting started

We recommend creating a Python environment
```
python3 -m venv venv-diarization
source venv-diarization/bin/activate
```

Clone the repository
```
git clone https://github.com/paul-cayet/end-to-end-neural-diarization.git
```

Install the library
```
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## ⚠️ Modifications to the Pyannote Library

**Padding audio files by default instead of raising an error when finetuning the segmentation model**

In `pyannote/audio/core/io.py` in `Audio.crop` replace the argument `mode="raise"` by `mode="pad"`

**Allowing the loading of finetuned models**

In `pyannote/audio/core/model.py` in `Model.from_pretrained` replace the argument `strict: bool = True` by `strict: bool = False`


## Compute statistics over the data

```
python3 -m neural_diarization.evaluate.analyze_dataset -c \
    --data DATASET_NAME \
    --rttm PATH_TO_RTTM_FILES \
    --audio PATH_TO_AUDIO_DIR
```


## Data Preparation for finetuning

Pyannote requires the annotation data to be in a particular format. You can start from annotations such as the ones from [Mediaeval2016 repo](https://github.com/MediaevalPersonDiscoveryTask/Metadata2016).

**Converting `sd` annotations to `rttm` annotations**

```
python3 -m neural_diarization.labelling.convert_sd_files -c \
    --sd_dirname DIR_TO_SD_ANNOTATIONS \
    --rttm_dirname DIR_TO_WRITE_NEW_RTTM_ANNOTATIONS
```

**Generating the train/test `rttm`, `lst` and `uem` from the global `rttm` annotations**\
Also takes care of the conversion from 60minutes files to 60seconds files.

Read [the example use case](examples/ina_finetuning/README.md)

**Precompute audio metadata**\
Used to accelerate training.

```
python3 -m neural_diarization.utils.precompute_torchaudio_info -c \
    --config_path PATH_TO_TRAINING_CONFIG
```



## Finetune a pretrained segmentation model

Read [the example use case](examples/ina_finetuning/README.md)

After having prepared the data, you can use the finetuning script.
```
python3 -m neural_diarization.finetune.finetune -c \
    --config_path PATH_TO_TRAINING_CONFIG
```


## Evaluate a speaker diarization pipeline



**Run inference over test audio**

```
python3 -m neural_diarization.evaluate.predict -c \
    --audio PATH_TO_AUDIO_DIR \
    --ckpt OPTIONAL_PATH_TO_MODEL_CHECKPOINT \
    --n NUMBER_OF_FILES_TO_PREDICT
```

**Evaluate predictions**

```
python3 -m neural_diarization.evaluate.evaluate -c \
    --data DATASET_NAME \
    --labels PATH_TO_LABELS_DIR \
    --chunks NUMBER_OF_CHUNKS_TO_SPLIT_AUDIO_FILES_INTO
```


**Plot evaluation results**

```
python3 -m neural_diarization.evaluate.dashboard -c \
    --data DATASET_NAME \
    --file PATH_TO_EVALUATION_FILE
```



# Citation

```
@article{2024exploring,
  title={Exploring Neural Diarization Techniques for Political Media Monitoring},
  author={Cayet, Paul and Ibanez, Nicolas and Rondier, Lucien},
  journal={Zenodo ...},
  year={2024}
}
```


# License

This work is licensed under a [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).


[![License: GPL v3](https://www.gnu.org/graphics/gplv3-88x31.png)](https://www.gnu.org/licenses/gpl-3.0)

