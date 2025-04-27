# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
import numpy as np

def calculate_psf(W_rec, pupila_mask):
    fase_W = 2 * np.pi * W_rec
    pupil_function = pupila_mask * np.exp(1j * fase_W)
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))
    PSF = np.abs(E_focal)**2
    PSF /= PSF.max()
    PSF_log = np.log10(PSF + 1e-8)
    return PSF, PSF_log