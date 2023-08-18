"""
Helper functions to compute adversarial losses for generator and discriminator.
"""
import torch


def discriminator_loss(disc_outputs, real=True):
    total_loss = 0.0
    for out in disc_outputs:
        if real:
            loss = torch.mean((1 - out) ** 2)
        else:
            loss = torch.mean((0 - out) ** 2)
        total_loss += loss
    return total_loss


def generator_loss(disc_outputs):
    total_loss = 0.0
    for out in disc_outputs:
        loss = torch.mean((1 - out) ** 2)
        total_loss += loss
    return total_loss
