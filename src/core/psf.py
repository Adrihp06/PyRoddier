# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
import numpy as np


def calculate_psf(wavefront, pupila_mask, wavelength=556):
    """
    Calcula la función de dispersión de punto (PSF) a partir del frente de onda.

    Método:
    Aplica la transformada de Fourier a la función de pupila compleja para obtener
    la distribución de intensidad en el plano focal (PSF). La fase se calcula
    convirtiendo el frente de onda de unidades de longitud de onda a radianes.

    Parámetros:
    - wavefront: array 2D - Frente de onda aberrado en unidades de longitud de onda (λ)
    - pupila_mask: array 2D - Máscara binaria de la pupila (0 o 1)
    - wavelength: float - Longitud de onda en nanómetros (default: 556 nm, verde)

    Retorna:
    - PSF: array 2D - Función de dispersión de punto normalizada (máximo = 1.0)
    - PSF_log: array 2D - PSF en escala logarítmica (log10), útil para visualización

    Raises:
    - ValueError: Si wavelength <= 0, si wavefront o pupila_mask no son finitos,
                  o si las dimensiones no coinciden

    Nota física:
    - El frente de onda se asume en unidades de λ (longitudes de onda)
    - La conversión a fase: φ = 2π × (wavefront / λ) × (2π / λ_nm)
    - El factor 1e-8 en PSF_log evita log(0)
    """
    # Validaciones
    if wavelength <= 0:
        raise ValueError(f"wavelength debe ser positivo, recibido: {wavelength}")

    if not np.all(np.isfinite(wavefront)):
        raise ValueError("wavefront contiene valores no finitos (NaN o Inf)")

    if not np.all(np.isfinite(pupila_mask)):
        raise ValueError("pupila_mask contiene valores no finitos (NaN o Inf)")

    if wavefront.shape != pupila_mask.shape:
        raise ValueError(
            f"Las dimensiones no coinciden: wavefront {wavefront.shape} vs pupila_mask {pupila_mask.shape}"
        )

    # Cálculo del PSF
    wavelength_conversion_factor = 2 * np.pi / wavelength
    fase_W = 2 * np.pi * wavefront * wavelength_conversion_factor
    pupil_function = pupila_mask * np.exp(1j * fase_W)

    # Transformada de Fourier para obtener campo eléctrico en el plano focal
    E_focal = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_function)))

    # Intensidad (PSF)
    PSF = np.abs(E_focal) ** 2

    # Validar que PSF no es cero
    if np.max(PSF) == 0:
        raise ValueError("PSF calculado es cero, verificar entradas")

    # Normalizar
    PSF /= PSF.max()

    # Escala logarítmica para visualización
    PSF_log = np.log10(PSF + 1e-8)

    return PSF, PSF_log