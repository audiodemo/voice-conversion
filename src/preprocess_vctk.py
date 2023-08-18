"""
Collect VCTK .flac files and save them as pickle files.
"""
import logging
import os
import argparse
import pickle
import json
import torch
import librosa

from utils import AttrDict
from pathlib import Path
from tqdm import tqdm
from librosa.util import normalize


def process_audio(wavspath, h):
    wav_files = list(Path(wavspath).rglob("*.flac"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .flac files found at {wavspath}")

    audio_list = []
    for wavpath in tqdm(wav_files):
        wav_orig, _ = librosa.load(wavpath, sr=h.sampling_rate, mono=True)
        wav_orig = normalize(wav_orig) * 0.95
        wav_orig = torch.FloatTensor(wav_orig)
        wav_orig = wav_orig.unsqueeze(0)

        # each training example consists of 8192 samples
        if wav_orig.shape[-1] >= h.segment_size:
            audio_list.append(wav_orig.cpu().detach().numpy()[0])

    return audio_list


def save_pickle(variable, fileName):
    with open(fileName, "wb") as f:
        pickle.dump(variable, f)


def preprocess_dataset(audio_data_path, spk_id, hyperparams, save_folder="./data"):
    """
    Preprocesses VCTK dataset.
    :param audio_data_path: Directory containing .wav files of the speaker.
    :param spk_id: ID of the speaker.
    :param hyperparams: Hyperparameters for model training
    :param save_folder: Directory to store preprocessed data.
    """

    logging.info(f"Preprocessing data for speaker: {spk_id}.")
    audio_list = process_audio(audio_data_path, hyperparams)

    if not os.path.exists(os.path.join(save_folder, spk_id)):
        os.makedirs(os.path.join(save_folder, spk_id))

    save_pickle(
        variable=audio_list,
        fileName=os.path.join(save_folder, spk_id, f"{spk_id}_audio.pickle"),
    )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    level = logging.INFO
    logging.basicConfig(format=formatter, level=level)

    parser = argparse.ArgumentParser(description="Preprocess VCTK dataset.")
    parser.add_argument(
        "--data_directory", type=str, help="Directory that contains the dataset."
    )
    parser.add_argument(
        "--preprocessed_data_directory",
        type=str,
        default="vctk_preprocessed",
        help="Directory to store the preprocessed data.",
    )
    parser.add_argument(
        "--speaker_ids",
        nargs="+",
        type=str,
        default=["p225", "p229", "p231", "p232"],
        help="Speaker IDs to be processed.",
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="JSON file containing hyperparameters used for training",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()
    json_config = json.loads(data)
    hyperparams = AttrDict(json_config)

    for speaker_id in args.speaker_ids:
        data_path = os.path.join(args.data_directory, speaker_id)
        preprocess_dataset(
            audio_data_path=data_path,
            spk_id=speaker_id,
            save_folder=args.preprocessed_data_directory,
            hyperparams=hyperparams,
        )
