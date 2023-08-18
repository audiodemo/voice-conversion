"""
Collect wTIMIT .wav files and save them as pickle files.
"""
import logging
import os
import argparse
import pickle
import json

import librosa
import torch
from utils import AttrDict
from pathlib import Path
from tqdm import tqdm
from librosa.util import normalize


def process_audio(wavspath, h):
    wav_files = list(Path(wavspath).rglob("*.wav"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found at {wavspath}")

    audio_list = []
    logging.info(f"Processing {len(wav_files)} files at {wavspath}")

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


def preprocess_dataset(
    normal_data_path, whisper_data_path, spk_id, hyperparams, save_folder="./data"
):
    """
    Preprocesses wTIMIT dataset.
    :param normal_data_path: Directory containing the normally voiced utterances of the speaker.
    :param whisper_data_path: Directory containing the whispered utterances of the speaker.
    :param spk_id: ID of the speaker.
    :param hyperparams: Hyperparameters for model training
    :param save_folder: Directory to store preprocessed data.
    """

    logging.info(f"Preprocessing data for speaker: {spk_id}")
    for nw, data_path in [("normal", normal_data_path), ("whisper", whisper_data_path)]:
        audio_list = process_audio(data_path, hyperparams)

        if not os.path.exists(os.path.join(save_folder, spk_id + "_" + nw)):
            os.makedirs(os.path.join(save_folder, spk_id + "_" + nw))

        save_pickle(
            variable=audio_list,
            fileName=os.path.join(
                save_folder, spk_id + "_" + nw, f"{spk_id}_audio.pickle"
            ),
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    level = logging.INFO
    logging.basicConfig(format=formatter, level=level)

    parser = argparse.ArgumentParser(description="Preprocess wTIMIT dataset.")
    parser.add_argument(
        "--data_directory", type=str, help="Directory that contains the dataset."
    )
    parser.add_argument(
        "--preprocessed_data_directory",
        type=str,
        default="wtimit_preprocessed",
        help="Directory to store the preprocessed data.",
    )
    parser.add_argument(
        "--speaker_ids",
        nargs="+",
        type=str,
        default=["006", "007", "008", "017", "105", "109", "111", "117"],
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
        nrml_data_path = os.path.join(args.data_directory, "normal", speaker_id)
        whsp_data_path = os.path.join(args.data_directory, "whisper", speaker_id)
        preprocess_dataset(
            normal_data_path=nrml_data_path,
            whisper_data_path=whsp_data_path,
            spk_id=speaker_id,
            save_folder=args.preprocessed_data_directory,
            hyperparams=hyperparams,
        )
