"""
Model training procedure.
"""
import logging
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from utils import AttrDict, copy_conf, mel_spectrogram, load_pickle_file
from dataset import VCDataset
from model import (
    GeneratorMasked,
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    GeneratorMaskedGLU,
)
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from loss import discriminator_loss, generator_loss

torch.backends.cudnn.benchmark = True


def train(rank, cli_args, hyperparams):
    global start, validation_loader, start_b, sw
    if hyperparams.num_gpus > 1:
        init_process_group(
            backend=hyperparams.dist_config["dist_backend"],
            init_method=hyperparams.dist_config["dist_url"],
            world_size=hyperparams.dist_config["world_size"] * hyperparams.num_gpus,
            rank=rank,
        )

    torch.cuda.manual_seed(hyperparams.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:{:d}".format(rank))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if hyperparams.generator == "GeneratorMaskedGLU":
        generator_A2B = GeneratorMaskedGLU(hyperparams).to(device)
        generator_B2A = GeneratorMaskedGLU(hyperparams).to(device)
    else:
        generator_A2B = GeneratorMasked(hyperparams).to(device)
        generator_B2A = GeneratorMasked(hyperparams).to(device)

    logging.info(f"Using generator {generator_A2B.__class__.__name__}")

    mpd_A = MultiPeriodDiscriminator().to(device)
    mpd_B = MultiPeriodDiscriminator().to(device)

    msd_A = MultiScaleDiscriminator().to(device)
    msd_B = MultiScaleDiscriminator().to(device)

    mpd_A2 = MultiPeriodDiscriminator().to(device)
    mpd_B2 = MultiPeriodDiscriminator().to(device)

    msd_A2 = MultiScaleDiscriminator().to(device)
    msd_B2 = MultiScaleDiscriminator().to(device)

    if rank == 0:
        logging.info(generator_A2B)
        os.makedirs(cli_args.checkpoint_path, exist_ok=True)
        logging.info(f"checkpoints directory: {cli_args.checkpoint_path}")

    cp_g_a = None
    cp_g_b = None
    cp_do_a = None
    cp_do_b = None
    cp_do_a2 = None
    cp_do_b2 = None
    if os.path.isdir(cli_args.checkpoint_path):
        cp_g_a = scan_checkpoint(cli_args.checkpoint_path, "ga_")
        cp_g_b = scan_checkpoint(cli_args.checkpoint_path, "gb_")

        cp_do_a = scan_checkpoint(cli_args.checkpoint_path, "doa_")
        cp_do_b = scan_checkpoint(cli_args.checkpoint_path, "dob_")

        cp_do_a2 = scan_checkpoint(cli_args.checkpoint_path, "doa2_")
        cp_do_b2 = scan_checkpoint(cli_args.checkpoint_path, "dob2_")

    steps = 0
    if (
        cp_g_a is None
        or cp_do_a is None
        or cp_g_b is None
        or cp_do_b is None
        or cp_do_a2 is None
        or cp_do_b2 is None
    ):
        state_dict_do_a = None
        state_dict_do_b = None
        state_dict_do_a2 = None
        state_dict_do_b2 = None
        last_epoch = -1
    else:
        state_dict_g_a = load_checkpoint(cp_g_a, device)
        state_dict_g_b = load_checkpoint(cp_g_b, device)

        state_dict_do_a = load_checkpoint(cp_do_a, device)
        state_dict_do_b = load_checkpoint(cp_do_b, device)

        state_dict_do_a2 = load_checkpoint(cp_do_a2, device)
        state_dict_do_b2 = load_checkpoint(cp_do_b2, device)

        generator_A2B.load_state_dict(state_dict_g_a["generator"])
        generator_B2A.load_state_dict(state_dict_g_b["generator"])

        mpd_A.load_state_dict(state_dict_do_a["mpd"])
        mpd_B.load_state_dict(state_dict_do_b["mpd"])
        msd_A.load_state_dict(state_dict_do_a["msd"])
        msd_B.load_state_dict(state_dict_do_b["msd"])

        mpd_A2.load_state_dict(state_dict_do_a2["mpd"])
        mpd_B2.load_state_dict(state_dict_do_b2["mpd"])
        msd_A2.load_state_dict(state_dict_do_a2["msd"])
        msd_B2.load_state_dict(state_dict_do_b2["msd"])

        steps = state_dict_do_a["steps"] + 1
        last_epoch = state_dict_do_a["epoch"]

    if hyperparams.num_gpus > 1:
        generator_A2B = DistributedDataParallel(generator_A2B, device_ids=[rank]).to(
            device
        )
        generator_B2A = DistributedDataParallel(generator_B2A, device_ids=[rank]).to(
            device
        )

        mpd_A = DistributedDataParallel(mpd_A, device_ids=[rank]).to(device)
        mpd_B = DistributedDataParallel(mpd_B, device_ids=[rank]).to(device)
        msd_A = DistributedDataParallel(msd_A, device_ids=[rank]).to(device)
        msd_B = DistributedDataParallel(msd_B, device_ids=[rank]).to(device)

        mpd_A2 = DistributedDataParallel(mpd_A2, device_ids=[rank]).to(device)
        mpd_B2 = DistributedDataParallel(mpd_B2, device_ids=[rank]).to(device)
        msd_A2 = DistributedDataParallel(msd_A2, device_ids=[rank]).to(device)
        msd_B2 = DistributedDataParallel(msd_B2, device_ids=[rank]).to(device)

    optim_g_A = torch.optim.AdamW(
        generator_A2B.parameters(),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )
    optim_g_B = torch.optim.AdamW(
        generator_B2A.parameters(),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )

    optim_d_A = torch.optim.AdamW(
        itertools.chain(msd_A.parameters(), mpd_A.parameters()),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )
    optim_d_B = torch.optim.AdamW(
        itertools.chain(msd_B.parameters(), mpd_B.parameters()),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )

    optim_d_A2 = torch.optim.AdamW(
        itertools.chain(msd_A2.parameters(), mpd_A2.parameters()),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )
    optim_d_B2 = torch.optim.AdamW(
        itertools.chain(msd_B2.parameters(), mpd_B2.parameters()),
        hyperparams.learning_rate,
        betas=(hyperparams.adam_b1, hyperparams.adam_b2),
    )

    if state_dict_do_a is not None:
        optim_g_A.load_state_dict(state_dict_do_a["optim_g"])
        optim_d_A.load_state_dict(state_dict_do_a["optim_d"])
        optim_d_A2.load_state_dict(state_dict_do_a2["optim_d"])
    if state_dict_do_b is not None:
        optim_g_B.load_state_dict(state_dict_do_b["optim_g"])
        optim_d_B.load_state_dict(state_dict_do_b["optim_d"])
        optim_d_B2.load_state_dict(state_dict_do_b2["optim_d"])

    scheduler_g_A = torch.optim.lr_scheduler.ExponentialLR(
        optim_g_A, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )
    scheduler_g_B = torch.optim.lr_scheduler.ExponentialLR(
        optim_g_B, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )
    scheduler_d_A = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_A, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )
    scheduler_d_B = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_B, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )
    scheduler_d_A2 = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_A2, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )
    scheduler_d_B2 = torch.optim.lr_scheduler.ExponentialLR(
        optim_d_B2, gamma=hyperparams.lr_decay, last_epoch=last_epoch
    )

    # Initialize dataset of speaker A
    spk_id = cli_args.speaker_A_id
    if cli_args.corpus == "wtimit":
        spk_id = cli_args.speaker_A_id + "_whisper"

    dataset_A_audio = load_pickle_file(
        os.path.join(
            cli_args.preprocessed_data_dir,
            spk_id,
            f"{cli_args.speaker_A_id}_audio.pickle",
        )
    )

    # Initialize dataset of speaker B
    spk_id = cli_args.speaker_B_id
    if cli_args.corpus == "wtimit":
        spk_id = cli_args.speaker_B_id + "_normal"

    dataset_B_audio = load_pickle_file(
        os.path.join(
            cli_args.preprocessed_data_dir,
            spk_id,
            f"{cli_args.speaker_B_id}_audio.pickle",
        )
    )

    trainset = VCDataset(
        datasetA=dataset_A_audio,
        datasetB=dataset_B_audio,
        n_frames=hyperparams.n_frames,
        max_mask_len=hyperparams.max_mask_len,
        hop_size=hyperparams.hop_size,
        n_fft=hyperparams.n_fft,
        num_mels=hyperparams.num_mels,
        win_size=hyperparams.win_size,
        sampling_rate=hyperparams.sampling_rate,
        fmin=hyperparams.fmin,
        fmax=hyperparams.fmax,
    )

    train_sampler = DistributedSampler(trainset) if hyperparams.num_gpus > 1 else None

    train_loader = DataLoader(
        trainset,
        collate_fn=None,
        num_workers=hyperparams.num_workers,
        shuffle=False,
        sampler=train_sampler,
        batch_size=hyperparams.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        validset = VCDataset(
            datasetA=dataset_A_audio,
            datasetB=dataset_B_audio,
            n_frames=hyperparams.num_frames_validation,
            max_mask_len=hyperparams.max_mask_len,
            hop_size=hyperparams.hop_size,
            n_fft=hyperparams.n_fft,
            num_mels=hyperparams.num_mels,
            win_size=hyperparams.win_size,
            sampling_rate=hyperparams.sampling_rate,
            fmin=hyperparams.fmin,
            fmax=hyperparams.fmax,
            valid=True,
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(cli_args.checkpoint_path, "logs"))

    for epoch in range(max(0, last_epoch), cli_args.training_epochs):
        if rank == 0:
            start = time.time()
            logging.info("Epoch: {}".format(epoch + 1))

        if hyperparams.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            real_A_mel, mask_A, real_B_mel, mask_B, real_A, real_B = batch

            real_A = real_A.to(device, dtype=torch.float)
            real_A_mel = real_A_mel.to(device, dtype=torch.float)
            mask_A = mask_A.to(device, dtype=torch.float)

            real_B = real_B.to(device, dtype=torch.float)
            real_B_mel = real_B_mel.to(device, dtype=torch.float)
            mask_B = mask_B.to(device, dtype=torch.float)

            real_A = real_A.unsqueeze(1)
            real_B = real_B.unsqueeze(1)

            # Train discriminator
            generator_A2B.eval()
            generator_B2A.eval()
            mpd_A.train()
            mpd_B.train()
            msd_A.train()
            msd_B.train()
            mpd_A2.train()
            mpd_B2.train()
            msd_A2.train()
            msd_B2.train()

            # Generator forward pass
            generated_A = generator_B2A(real_B_mel, mask_B)
            generated_B = generator_A2B(real_A_mel, mask_A)

            # MPD forward pass
            d_real_A_mpd = mpd_A(real_A)
            d_real_B_mpd = mpd_B(real_B)
            d_real_A2_mpd = mpd_A2(real_A)
            d_real_B2_mpd = mpd_B2(real_B)
            d_fake_A_mpd = mpd_A(generated_A)

            # MSD forward pass
            d_real_A_msd = msd_A(real_A)
            d_real_B_msd = msd_B(real_B)
            d_real_A2_msd = msd_A2(real_A)
            d_real_B2_msd = msd_B2(real_B)
            d_fake_A_msd = msd_A(generated_A)

            # For secondary adversarial loss A->B
            generated_A_mel = mel_spectrogram(
                generated_A.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            cycled_B = generator_A2B(generated_A_mel, torch.ones_like(generated_A_mel))
            d_cycled_B_mpd = mpd_B2(cycled_B)
            d_cycled_B_msd = msd_B2(cycled_B)

            d_fake_B_mpd = mpd_B(generated_B)
            d_fake_B_msd = msd_B(generated_B)

            # For secondary adversarial loss B->A
            generated_B_mel = mel_spectrogram(
                generated_B.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            cycled_A = generator_B2A(generated_B_mel, torch.ones_like(generated_B_mel))
            d_cycled_A_mpd = mpd_A2(cycled_A)
            d_cycled_A_msd = msd_A2(cycled_A)

            # Loss computations
            d_loss_A_real_mpd = discriminator_loss(d_real_A_mpd, real=True)
            d_loss_A_real_msd = discriminator_loss(d_real_A_msd, real=True)
            d_loss_A_fake_mpd = discriminator_loss(d_fake_A_mpd, real=False)
            d_loss_A_fake_msd = discriminator_loss(d_fake_A_msd, real=False)

            d_loss_A_mpd = (d_loss_A_real_mpd + d_loss_A_fake_mpd) / 2.0
            d_loss_A_msd = (d_loss_A_real_msd + d_loss_A_fake_msd) / 2.0

            d_loss_B_real_mpd = discriminator_loss(d_real_B_mpd, real=True)
            d_loss_B_real_msd = discriminator_loss(d_real_B_msd, real=True)
            d_loss_B_fake_mpd = discriminator_loss(d_fake_B_mpd, real=False)
            d_loss_B_fake_msd = discriminator_loss(d_fake_B_msd, real=False)

            d_loss_B_mpd = (d_loss_B_real_mpd + d_loss_B_fake_mpd) / 2.0
            d_loss_B_msd = (d_loss_B_real_msd + d_loss_B_fake_msd) / 2.0

            # Secondary adversarial loss computation
            d_loss_A_cycled_mpd = discriminator_loss(d_cycled_A_mpd, real=False)
            d_loss_A_cycled_msd = discriminator_loss(d_cycled_A_msd, real=False)
            d_loss_B_cycled_mpd = discriminator_loss(d_cycled_B_mpd, real=False)
            d_loss_B_cycled_msd = discriminator_loss(d_cycled_B_msd, real=False)

            d_loss_A2_real_mpd = discriminator_loss(d_real_A2_mpd, real=True)
            d_loss_A2_real_msd = discriminator_loss(d_real_A2_msd, real=True)
            d_loss_B2_real_mpd = discriminator_loss(d_real_B2_mpd, real=True)
            d_loss_B2_real_msd = discriminator_loss(d_real_B2_msd, real=True)

            d_loss_A_2nd_mpd = (d_loss_A2_real_mpd + d_loss_A_cycled_mpd) / 2.0
            d_loss_A_2nd_msd = (d_loss_A2_real_msd + d_loss_A_cycled_msd) / 2.0
            d_loss_B_2nd_mpd = (d_loss_B2_real_mpd + d_loss_B_cycled_mpd) / 2.0
            d_loss_B_2nd_msd = (d_loss_B2_real_msd + d_loss_B_cycled_msd) / 2.0

            # Final discriminator loss
            d_loss_mpd = (d_loss_A_mpd + d_loss_B_mpd) / 2.0 + (
                d_loss_A_2nd_mpd + d_loss_B_2nd_mpd
            ) / 2.0
            d_loss_msd = (d_loss_A_msd + d_loss_B_msd) / 2.0 + (
                d_loss_A_2nd_msd + d_loss_B_2nd_msd
            ) / 2.0

            d_loss = d_loss_mpd + d_loss_msd

            optim_g_A.zero_grad()
            optim_g_B.zero_grad()
            optim_d_A.zero_grad()
            optim_d_B.zero_grad()
            optim_d_A2.zero_grad()
            optim_d_B2.zero_grad()

            d_loss.backward()

            optim_d_A.step()
            optim_d_B.step()
            optim_d_A2.step()
            optim_d_B2.step()

            # Train generator
            generator_A2B.train()
            generator_B2A.train()
            mpd_A.eval()
            mpd_B.eval()
            msd_A.eval()
            msd_B.eval()
            mpd_A2.eval()
            mpd_B2.eval()
            msd_A2.eval()
            msd_B2.eval()

            # Generator forward pass
            fake_B = generator_A2B(real_A_mel, mask_A)
            fake_B_mel = mel_spectrogram(
                fake_B.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            cycle_A = generator_B2A(fake_B_mel, torch.ones_like(fake_B_mel))
            fake_A = generator_B2A(real_B_mel, mask_B)
            fake_A_mel = mel_spectrogram(
                fake_A.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            cycle_B = generator_A2B(fake_A_mel, torch.ones_like(fake_A_mel))
            identity_A = generator_B2A(real_A_mel, torch.ones_like(real_A_mel))
            identity_B = generator_A2B(real_B_mel, torch.ones_like(real_B_mel))

            # Discriminator forward pass
            d_fake_A_mpd = mpd_A(fake_A)
            d_fake_A_msd = msd_A(fake_A)
            d_fake_B_mpd = mpd_B(fake_B)
            d_fake_B_msd = msd_B(fake_B)

            # Discriminator forward pass for secondary discriminator
            d_fake_cycle_A_mpd = mpd_A2(cycle_A)
            d_fake_cycle_A_msd = msd_A2(cycle_A)
            d_fake_cycle_B_mpd = mpd_B2(cycle_B)
            d_fake_cycle_B_msd = msd_B2(cycle_B)

            # Mel-spectrograms for cycle-consistency and identity loss computation
            cycle_A_mel = mel_spectrogram(
                cycle_A.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            cycle_B_mel = mel_spectrogram(
                cycle_B.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            identity_A_mel = mel_spectrogram(
                identity_A.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )
            identity_B_mel = mel_spectrogram(
                identity_B.squeeze(1),
                hyperparams.n_fft,
                hyperparams.num_mels,
                hyperparams.sampling_rate,
                hyperparams.hop_size,
                hyperparams.win_size,
                hyperparams.fmin,
                hyperparams.fmax_for_loss,
            )

            # Cycle-consistency Loss -> L1 error on Mel-spectrograms
            cycleLoss = torch.mean(torch.abs(real_A_mel - cycle_A_mel)) + torch.mean(
                torch.abs(real_B_mel - cycle_B_mel)
            )

            # Generator identity loss -> L1 error on Mel-spectrograms
            identityLoss = torch.mean(
                torch.abs(real_A_mel - identity_A_mel)
            ) + torch.mean(torch.abs(real_B_mel - identity_B_mel))

            # Generator adversarial loss
            g_loss_A2B_mpd = generator_loss(d_fake_B_mpd)
            g_loss_A2B_msd = generator_loss(d_fake_B_msd)

            g_loss_A2B = g_loss_A2B_mpd + g_loss_A2B_msd

            g_loss_B2A_mpd = generator_loss(d_fake_A_mpd)
            g_loss_B2A_msd = generator_loss(d_fake_A_msd)

            g_loss_B2A = g_loss_B2A_mpd + g_loss_B2A_msd

            # Generator secondary adversarial loss
            generator_loss_A2B_2nd_mpd = generator_loss(d_fake_cycle_B_mpd)
            generator_loss_A2B_2nd_msd = generator_loss(d_fake_cycle_B_msd)

            generator_loss_A2B_2nd = (
                generator_loss_A2B_2nd_mpd + generator_loss_A2B_2nd_msd
            )

            generator_loss_B2A_2nd_mpd = generator_loss(d_fake_cycle_A_mpd)
            generator_loss_B2A_2nd_msd = generator_loss(d_fake_cycle_A_msd)

            generator_loss_B2A_2nd = (
                generator_loss_B2A_2nd_mpd + generator_loss_B2A_2nd_msd
            )

            # Total Generator loss
            g_loss = (
                g_loss_A2B
                + g_loss_B2A
                + generator_loss_A2B_2nd
                + generator_loss_B2A_2nd
                + hyperparams.cycle_loss_lambda * cycleLoss
                + hyperparams.identity_loss_lambda * identityLoss
            )

            optim_g_A.zero_grad()
            optim_g_B.zero_grad()
            optim_d_A.zero_grad()
            optim_d_B.zero_grad()
            optim_d_A2.zero_grad()
            optim_d_B2.zero_grad()

            g_loss.backward()
            optim_g_A.step()
            optim_g_B.step()

            if rank == 0:
                if steps % cli_args.stdout_interval == 0:
                    logging.info(
                        f"Steps: {steps:d}, "
                        f"Gen loss total: {g_loss:4.3f}, "
                        f"Disc loss total: {d_loss:4.3f}, "
                        f"Time: {time.time() - start_b:4.1f} seconds"
                    )

                # Save checkpoints
                if steps % cli_args.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/ga_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator_A2B.module
                                if hyperparams.num_gpus > 1
                                else generator_A2B
                            ).state_dict()
                        },
                    )

                    checkpoint_path = "{}/gb_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator_B2A.module
                                if hyperparams.num_gpus > 1
                                else generator_B2A
                            ).state_dict()
                        },
                    )

                    checkpoint_path = "{}/doa_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (
                                mpd_A.module if hyperparams.num_gpus > 1 else mpd_A
                            ).state_dict(),
                            "msd": (
                                msd_A.module if hyperparams.num_gpus > 1 else msd_A
                            ).state_dict(),
                            "optim_g": optim_g_A.state_dict(),
                            "optim_d": optim_d_A.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                    checkpoint_path = "{}/dob_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (
                                mpd_B.module if hyperparams.num_gpus > 1 else mpd_B
                            ).state_dict(),
                            "msd": (
                                msd_B.module if hyperparams.num_gpus > 1 else msd_B
                            ).state_dict(),
                            "optim_g": optim_g_B.state_dict(),
                            "optim_d": optim_d_B.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                    checkpoint_path = "{}/doa2_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (
                                mpd_A2.module if hyperparams.num_gpus > 1 else mpd_A2
                            ).state_dict(),
                            "msd": (
                                msd_A2.module if hyperparams.num_gpus > 1 else msd_A2
                            ).state_dict(),
                            "optim_g": optim_g_A.state_dict(),
                            "optim_d": optim_d_A2.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                    checkpoint_path = "{}/dob2_{:08d}".format(
                        cli_args.checkpoint_path, steps
                    )
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (
                                mpd_B2.module if hyperparams.num_gpus > 1 else mpd_B2
                            ).state_dict(),
                            "msd": (
                                msd_B2.module if hyperparams.num_gpus > 1 else msd_B2
                            ).state_dict(),
                            "optim_g": optim_g_B.state_dict(),
                            "optim_d": optim_d_B2.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % cli_args.summary_interval == 0:
                    sw.add_scalar("training/cycle_loss", cycleLoss, steps)
                    sw.add_scalar("training/identity_loss", identityLoss, steps)
                    sw.add_scalar("training/disc_loss_total", d_loss, steps)
                    sw.add_scalar("training/gen_loss_total", g_loss, steps)

                # Validation
                if steps % cli_args.validation_interval == 0:
                    logging.info(f"Run validation at step {steps}")
                    generator_A2B.eval()
                    generator_B2A.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x_mel, y_mel, x, y = batch
                            y_g_hat = generator_A2B(
                                x_mel.to(device), torch.ones_like(x_mel).to(device)
                            )
                            x_g_hat = generator_B2A(
                                y_mel.to(device), torch.ones_like(y_mel).to(device)
                            )
                            # Pad generated audio so it matches with target audio
                            try:
                                y_g_hat_pad = torch.nn.functional.pad(
                                    y_g_hat,
                                    (0, y.size(1) - y_g_hat.size(2)),
                                    "constant",
                                )
                                y_mel = y_mel.to(device)
                                y_g_hat_mel = mel_spectrogram(
                                    y_g_hat_pad.squeeze(1),
                                    hyperparams.n_fft,
                                    hyperparams.num_mels,
                                    hyperparams.sampling_rate,
                                    hyperparams.hop_size,
                                    hyperparams.win_size,
                                    hyperparams.fmin,
                                    hyperparams.fmax_for_loss,
                                )
                                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                            except RuntimeError as e:
                                logging.error(
                                    f"Unable to compute validation error! Variables: x_mel={x_mel.shape} "
                                    f"y_mel={y_mel.shape} x={x.shape} y={y.shape} "
                                    f"y_g_hat={y_g_hat.shape} x_g_hat={x_g_hat.shape}"
                                )
                                logging.error(e)

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(
                                        "gt/y_tgt_{}".format(j),
                                        y[0],
                                        steps,
                                        hyperparams.sampling_rate,
                                    )
                                    sw.add_audio(
                                        "gt/x_src_{}".format(j),
                                        x[0],
                                        steps,
                                        hyperparams.sampling_rate,
                                    )
                                    sw.add_figure(
                                        "gt/y_spec_tgt_{}".format(j),
                                        plot_spectrogram(y_mel[0].cpu().numpy()),
                                        steps,
                                    )
                                    sw.add_figure(
                                        "gt/x_spec_src_{}".format(j),
                                        plot_spectrogram(x_mel[0].cpu().numpy()),
                                        steps,
                                    )

                                sw.add_audio(
                                    "generated/y_hat_{}".format(j),
                                    y_g_hat[0],
                                    steps,
                                    hyperparams.sampling_rate,
                                )
                                sw.add_audio(
                                    "generated/x_hat_{}".format(j),
                                    x_g_hat[0],
                                    steps,
                                    hyperparams.sampling_rate,
                                )
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat.squeeze(1),
                                    hyperparams.n_fft,
                                    hyperparams.num_mels,
                                    hyperparams.sampling_rate,
                                    hyperparams.hop_size,
                                    hyperparams.win_size,
                                    hyperparams.fmin,
                                    hyperparams.fmax,
                                )
                                x_hat_spec = mel_spectrogram(
                                    x_g_hat.squeeze(1),
                                    hyperparams.n_fft,
                                    hyperparams.num_mels,
                                    hyperparams.sampling_rate,
                                    hyperparams.hop_size,
                                    hyperparams.win_size,
                                    hyperparams.fmin,
                                    hyperparams.fmax,
                                )
                                sw.add_figure(
                                    "generated/y_hat_spec_{}".format(j),
                                    plot_spectrogram(
                                        y_hat_spec.squeeze(0).cpu().numpy()
                                    ),
                                    steps,
                                )
                                sw.add_figure(
                                    "generated/x_hat_spec_{}".format(j),
                                    plot_spectrogram(
                                        x_hat_spec.squeeze(0).cpu().numpy()
                                    ),
                                    steps,
                                )
                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator_A2B.train()
                    generator_B2A.train()

            steps += 1

        scheduler_g_A.step()
        scheduler_g_B.step()

        scheduler_d_A.step()
        scheduler_d_B.step()

        scheduler_d_A2.step()
        scheduler_d_B2.step()

        if rank == 0:
            logging.info(f"Epoch {epoch + 1} took {int(time.time() - start)} seconds")


def main():
    logging.info("Initializing training process.")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speaker_A_id", type=str, default="015", help="Source speaker id."
    )
    parser.add_argument(
        "--speaker_B_id", type=str, default="015", help="Target speaker id"
    )
    parser.add_argument(
        "--preprocessed_data_dir",
        type=str,
        default="wtimit_preprocessed",
        help="Directory containing preprocessed dataset files.",
    )
    parser.add_argument("--checkpoint_path", default="checkpoints/exp1")
    parser.add_argument("--corpus", default="wtimit")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--training_epochs", default=3000, type=int)
    parser.add_argument("--stdout_interval", default=50, type=int)
    parser.add_argument("--checkpoint_interval", default=5000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=1000, type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    hyperparams = AttrDict(json_config)
    copy_conf(args.config, "config.json", args.checkpoint_path)

    torch.manual_seed(hyperparams.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparams.seed)
        hyperparams.num_gpus = torch.cuda.device_count()
        hyperparams.batch_size = int(hyperparams.batch_size / hyperparams.num_gpus)
        logging.info(f"Using CUDA. Batch size per GPU: {hyperparams.batch_size}")
    elif torch.backends.mps.is_available():
        torch.cuda.manual_seed(hyperparams.seed)
        hyperparams.num_gpus = 1
        hyperparams.batch_size = int(hyperparams.batch_size / hyperparams.num_gpus)
        logging.info(f"Using MPS backend. Batch size per GPU: {hyperparams.batch_size}")

    if hyperparams.num_gpus > 1:
        mp.spawn(
            train,
            nprocs=hyperparams.num_gpus,
            args=(
                args,
                hyperparams,
            ),
        )
    else:
        train(0, args, hyperparams)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    level = logging.INFO
    logging.basicConfig(format=formatter, level=level)
    main()
