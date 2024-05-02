# Finetuning Pre-trained models on INA dataset.

Guide to fine-tune a pre-trained Pyannote segmentation model.
Works with the INA diarization annotations from the [Mediaeval2016 repo](https://github.com/MediaevalPersonDiscoveryTask/Metadata2016)

Steps:
- Change the relevant filepaths in the `ina_finetuning.py` file
- Execute `python3 ina_finetuning.py` to generate the labels
- Modify the configuration file `config.yml`

When testing locally, generate the split audio files (training is much more efficient when training with small audio files of 1 minute rather than the original 1 hour long files) using `split_audio_files.sh`.

We used a remote cluster to run our experiments thus used the script `submit-slurm.py`. The code can easily be adapted for local use.

Eiher way, before calling the finetuning script, the repo should look as follows:

```
├── data
│   ├── 130607FR20100_B_000.MPG.wav
│   └── ...
├── data_ina
│   ├── data_annot_finetuning
│   ├── data_annot_small (for testing purposes)
│   ├── torchinfo_all.json
│   ├── train_test_split.json (optional)
│   └── MyDatabase-small.yml
├── src
│   └── neural_diarization...
├── temp_rttm_folder
│   ├── converted
│   │     ├── 130607FR20000_B.rttm
│   │     └── ...
│   └── new_converted
│         ├── 130607FR20000_B.rttm
│         └── ...
├── .env
├── training_config.yml
├── model_config.yml
└── README.md
```