# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from scipy.ndimage import gaussian_filter
from .zernike import fit_zernike

def calculate_interferogram(reference_intensity, reference_frequency, wavefront, annular_mask):
    N = wavefront.shape[0]
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # Convertir a fase en radianes si wavefront está en longitudes de onda
    lambda_equiv = 1.0  # Asume que ya está en longitudes de onda
    fase_W = 2 * np.pi * (wavefront / lambda_equiv)

    # Tilt lineal en X+Y
    tilt = 2 * np.pi * reference_frequency * ((X + Y) / 2)

    # Fase total
    fase_total = fase_W + tilt

    # Interferencia con onda plana
    campo_aberrado = np.exp(1j * fase_total)
    campo_ref = 1
    interferogram = reference_intensity * np.abs(campo_aberrado + campo_ref)**2

    # Opcional: aplicar máscara para evitar ruido fuera de la pupila
    interferogram *= (annular_mask > 0)

    return interferogram

def rellenar_pupila(wavefront, mask, max_order=23):
    """
    Rellena el centro obstruido de la pupila usando ajuste Zernike,
    como lo hace WinRoddier.

    Retorna un wavefront continuo sobre el disco completo.
    """
    # Crear base Zernike completa (sobre toda la imagen)
    h, w = wavefront.shape
    R = w // 2
    center = (w // 2, h // 2)

    # Ajustar coeficientes Zernike solo en la zona válida
    coeffs, base = fit_zernike(wavefront, mask, R, center, max_order)

    # Reconstruir frente completo con todos los modos
    wavefront_full = np.sum(base * coeffs[:, None, None], axis=0)
    return wavefront_full