# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np

import numpy as np

def calculate_psf(W_rec, pupila_mask):
    # Convertir frente de onda en radianes
    fase_W = 2 * np.pi * W_rec

    # Función pupilar compleja
    pupil_function = pupila_mask * np.exp(1j * fase_W)

    # FFT normalizada correctamente para generar la PSF
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))

    # Intensidad (PSF)
    PSF = np.abs(E_focal)**2

    # Normalización para visualización correcta
    PSF /= PSF.sum()

    # Versión logarítmica para visualización
    PSF_log = np.log10(PSF + 1e-8)

    return PSF, PSF_log
