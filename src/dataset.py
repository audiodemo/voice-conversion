"""
Dataset for voice conversion.
Returns chunked Mel-spectrogram pairs from two datasets, the corresponding masks, and the underlying audio.
"""
from utils import mel_spectrogram
from torch.utils.data.dataset import Dataset
import torch
import numpy as np


class VCDataset(Dataset):
    def __init__(
        self,
        datasetA,
        datasetB=None,
        n_frames=64,
        max_mask_len=25,
        hop_size=256,
        valid=False,
        n_fft=1024,
        num_mels=80,
        sampling_rate=22050,
        win_size=1024,
        fmin=0.0,
        fmax=None,
    ):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.n_frames = n_frames
        self.valid = valid
        self.max_mask_len = max_mask_len
        self.hop_size = hop_size
        self.segment_size = int(n_frames * hop_size)

        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax

    def __getitem__(self, index):
        dataset_A = self.datasetA
        dataset_B = self.datasetB
        n_frames = self.n_frames
        segment_size = self.segment_size

        if self.valid:
            if dataset_B is None:
                data_A_audio = dataset_A[index]
                data_A_audio = torch.FloatTensor(data_A_audio)
                data_A_mel = mel_spectrogram(
                    data_A_audio.unsqueeze(0),
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                return data_A_mel.squeeze(), data_A_audio
            else:
                data_A_audio = dataset_A[index]
                data_A_audio = torch.FloatTensor(data_A_audio)
                data_A_mel = mel_spectrogram(
                    data_A_audio.unsqueeze(0),
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                data_B_audio = dataset_B[index]
                data_B_audio = torch.FloatTensor(data_B_audio)
                data_B_mel = mel_spectrogram(
                    data_B_audio.unsqueeze(0),
                    self.n_fft,
                    self.num_mels,
                    self.sampling_rate,
                    self.hop_size,
                    self.win_size,
                    self.fmin,
                    self.fmax,
                    center=False,
                )
                return (
                    data_A_mel.squeeze(),
                    data_B_mel.squeeze(),
                    data_A_audio,
                    data_B_audio,
                )

        self.length = min(len(dataset_A), len(dataset_B))
        num_samples = min(len(dataset_A), len(dataset_B))

        train_data_A_idx = np.arange(len(dataset_A))
        train_data_B_idx = np.arange(len(dataset_B))
        np.random.shuffle(train_data_A_idx)
        np.random.shuffle(train_data_B_idx)
        train_data_A_idx_subset = train_data_A_idx[:num_samples]
        train_data_B_idx_subset = train_data_B_idx[:num_samples]

        train_data_A = list()
        train_data_A_audio = list()
        train_mask_A = list()
        train_data_B = list()
        train_data_B_audio = list()
        train_mask_B = list()

        for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
            data_A_audio = dataset_A[idx_A]
            data_A_audio = torch.FloatTensor(data_A_audio)
            samples_A_total = data_A_audio.shape[0]

            assert samples_A_total >= segment_size
            start_A = np.random.randint(samples_A_total - segment_size + 1)
            end_A = start_A + segment_size
            data_A_audio = data_A_audio[start_A:end_A]

            data_A_mel = mel_spectrogram(
                data_A_audio.unsqueeze(0),
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )

            mask_size_A = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_A
            mask_start_A = np.random.randint(0, n_frames - mask_size_A)
            mask_A = np.ones_like(data_A_mel)
            mask_A[:, mask_start_A:mask_start_A + mask_size_A] = 0.0

            train_data_A.append(data_A_mel)
            train_data_A_audio.append(data_A_audio)
            train_mask_A.append(mask_A)

            data_B_audio = dataset_B[idx_B]
            data_B_audio = torch.FloatTensor(data_B_audio)
            samples_B_total = data_B_audio.shape[0]

            assert samples_B_total >= segment_size
            start_B = np.random.randint(samples_B_total - segment_size + 1)
            end_B = start_B + segment_size

            data_B_audio = data_B_audio[start_B:end_B]

            data_B_mel = mel_spectrogram(
                data_B_audio.unsqueeze(0),
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )

            mask_size_B = np.random.randint(0, self.max_mask_len)
            assert n_frames > mask_size_B
            mask_start_B = np.random.randint(0, n_frames - mask_size_B)
            mask_B = np.ones_like(data_B_mel)
            mask_B[:, mask_start_B:mask_start_B + mask_size_B] = 0.0
            train_data_B.append(data_B_mel)
            train_data_B_audio.append(data_B_audio)
            train_mask_B.append(mask_B)

        train_data_A = np.array(train_data_A, dtype=object)
        train_data_B = np.array(train_data_B, dtype=object)
        train_mask_A = np.array(train_mask_A)
        train_mask_B = np.array(train_mask_B)

        return (
            train_data_A[index].squeeze(),
            train_mask_A[index].squeeze(),
            train_data_B[index].squeeze(),
            train_mask_B[index].squeeze(),
            train_data_A_audio[index],
            train_data_B_audio[index],
        )

    def __len__(self):
        if self.datasetB is None:
            return len(self.datasetA)
        else:
            return min(len(self.datasetA), len(self.datasetB))
