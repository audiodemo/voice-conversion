"""
Load a pretrained model and do inference on a set of audio files.
"""
import glob
import logging
import os
import argparse
import json
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from librosa.util import normalize
from scipy.io.wavfile import write
from utils import AttrDict, mel_spectrogram, load_pickle_file
from model import GeneratorMaskedGLU, GeneratorMasked

hyperparams = None
device = None
MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    logging.info(f"Loading {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=device)
    logging.info("Loading complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(
        x,
        hyperparams.n_fft,
        hyperparams.num_mels,
        hyperparams.sampling_rate,
        hyperparams.hop_size,
        hyperparams.win_size,
        hyperparams.fmin,
        hyperparams.fmax,
    )


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + "*")
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ""
    return sorted(cp_list)[-1]


def inference_from_pickle(cli_args):
    if hyperparams.generator == "GeneratorMaskedGLU":
        generator = GeneratorMaskedGLU(hyperparams).to(device)
    else:
        generator = GeneratorMasked(hyperparams).to(device)

    logging.info(f"Using generator {generator.__class__.__name__}")

    state_dict_g = load_checkpoint(cli_args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])
    filelist = load_pickle_file(
        os.path.join(cli_args.input_wavs_dir, f"{cli_args.speaker_id}_audio.pickle")
    )

    logging.info(
        f"Found {len(filelist)} .wav files in pickle file {cli_args.input_wavs_dir}"
    )

    out_path = os.path.join(cli_args.output_dir, cli_args.speaker_id)
    os.makedirs(out_path, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, orig_audio in enumerate(filelist):
            input_audio = torch.FloatTensor(orig_audio)

            input_mel = get_mel(input_audio.unsqueeze(0))

            y_g_hat = generator(
                input_mel.to(device), torch.ones_like(input_mel).to(device)
            )

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            orig_audio = orig_audio * MAX_WAV_VALUE
            orig_audio = orig_audio.astype("int16")

            output_file = os.path.join(out_path, str(i) + "_converted.wav")
            write(output_file, hyperparams.sampling_rate, audio)

            output_file = os.path.join(out_path, str(i) + "_orig.wav")
            write(output_file, hyperparams.sampling_rate, orig_audio)

    logging.info(f"Inference from pickle file finished. Results are at {out_path}")


def inference(cli_args):
    if hyperparams.generator == "GeneratorMaskedGLU":
        generator = GeneratorMaskedGLU(hyperparams).to(device)
    else:
        generator = GeneratorMasked(hyperparams).to(device)

    logging.info(f"Using generator {generator.__class__.__name__}")

    state_dict_g = load_checkpoint(cli_args.checkpoint_file, device)
    generator.load_state_dict(state_dict_g["generator"])

    audio_path = os.path.join(cli_args.input_wavs_dir, cli_args.speaker_id)
    if cli_args.corpus == "wtimit":
        audio_path = os.path.join(
            cli_args.input_wavs_dir, "whisper", cli_args.speaker_id
        )
    filelist = list(Path(audio_path).rglob("*.wav"))

    logging.info(f"Found {len(filelist)} .wav files at {audio_path}")

    out_path = os.path.join(cli_args.output_dir, cli_args.speaker_id)
    os.makedirs(out_path, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        for i, f in enumerate(tqdm(filelist)):
            wav, _ = librosa.load(
                os.path.join(audio_path, f), sr=hyperparams.sampling_rate, mono=True
            )
            wav = normalize(wav) * 0.95
            wav = torch.FloatTensor(wav)

            x = get_mel(wav.unsqueeze(0))

            y_g_hat = generator(x.to(device), torch.ones_like(x).to(device))

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            output_file = os.path.join(out_path, f.name)

            write(output_file, hyperparams.sampling_rate, audio)

    logging.info(f"Inference finished. Results are at {out_path}")


def main():
    logging.info("Initializing inference process.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", help="Directory containing .wav files")
    parser.add_argument(
        "--speaker_id", type=str, default="015", help="Source speaker id."
    )
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--corpus", type=str, default="wtimit")
    parser.add_argument("--from_pickle", action="store_true")

    args = parser.parse_args()

    config_file = os.path.join(os.path.split(args.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global hyperparams
    json_config = json.loads(data)
    hyperparams = AttrDict(json_config)

    torch.manual_seed(hyperparams.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparams.seed)
        device = torch.device("cuda")
        logging.info(f"Inference with CUDA")
    elif torch.backends.mps.is_available():
        torch.cuda.manual_seed(hyperparams.seed)
        device = torch.device("mps")
        logging.info(f"Inference with MPS backend")
    else:
        logging.info(f"Inference on CPU")
        device = torch.device("cpu")

    if args.from_pickle:
        inference_from_pickle(args)
    else:
        inference(args)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    level = logging.INFO
    logging.basicConfig(format=formatter, level=level)
    main()
