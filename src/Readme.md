# Source Code for Non-Parallel Conversion of Whispered Speech With Masked Cycle-Consistent Generative Adversarial Networks

## Getting Started

Install packages: 

```
pip3 install wheel
pip3 install -r requirements.txt
pip3 install torch torchvision torchaudio
```

**Hint:** Check the [example](./example) folder for a full example on the wTIMIT corpus.  

## Data Preparation 

Currently, two datasets are supported:

1. wTIMIT
2. CSTR VCTK 

Data splitting and other audio preprocessing steps are not part of this repository. 
Hence, you need to take care of creating train/test/val splits and other preprocessing steps (e.g. noise removal) by yourself.

### wTIMIT

- Download the wTIMIT corpus from: http://www.isle.illinois.edu/sst/data/wTIMIT/
- Optional (but recommended): Trim leading and trailing silences of the audio (e.g. with `librosa` or `sox`)
  - Especially the whispered utterances contain a lot of silence, which is rather detrimental for training. 
- Run `preprocess_wtimit.py`

The preprocessing script expects a directory structure like this:

```text
.
├── wtimit_test
│   ├── normal
│   │   └── 015
│   │       ├── s015u100.wav
│   │       └── s015u198.wav
│   └── whisper
│       └── 015
│           ├── s015u100.wav
│           └── s015u198.wav
└── wtimit_train
    ├── normal
    │   └── 015
    │       ├── s015u045.wav
    │       └── s015u046.wav
    └── whisper
        └── 015
            ├── s015u045.wav
            └── s015u046.wav
```

### VCTK

- Download the CSTR VCTK Corpus from https://datashare.ed.ac.uk/handle/10283/3443
- Run `preprocess_vctk.py`

The preprocessing script expects a directory structure like this:

```text
.
├── p225
│      ├── p225_039_mic1.flac
│      └── p225_040_mic1.flac
└── p227
       ├── p227_162_mic1.flac
       └── p227_163_mic1.flac
```

## Training 

The following example runs training for speaker `015` in the wTIMIT corpus. 
The audio data for speaker `015` needs to be preprocessed with `preprocess_wtimit.py` and stored in `wtimit_preprocessed`. 
Note that the source and target speaker IDs for whispered speech conversion are the same. 
The selection between *normal* and *whispered* speech is automatically done in the scripts. 

```shell
python3 -W ignore::FutureWarning train.py \
  --corpus wtimit \
  --speaker_A_id 015 \
  --speaker_B_id 015 \
  --preprocessed_data_dir wtimit_preprocessed \
  --checkpoint_path checkpoints/model_015 \
  --config config.json
```

### Generators

There are two different generator classes: 

- `GeneratorMaskedGLU`: Large generator including 2D convolutional gated-linear unit (GLU) for feature encoding
- `GeneratorMasked`: Smaller generator without GLU-based feature encoder

You can choose between them via the `config.json` file. 

## Inference

The following example shows how to run inference on a pretrained model for speaker `015` in the wTIMIT corpus. 

```shell
python3 -W ignore::FutureWarning inference.py \
  --input_wavs_dir raw_data/wtimit_test \
  --speaker_id 015 \
  --output_dir generated_audio/model_015_00050000 \
  --checkpoint_file checkpoints/model_015/ga_00050000
```


## Acknowledgments

This project is based on: 

- https://github.com/jik876/hifi-gan
- https://github.com/GANtastic3/MaskCycleGAN-VC
- https://github.com/jackaduma/CycleGAN-VC2