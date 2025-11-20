# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from astropy.io import fits
import numpy as np
from scipy.ndimage import center_of_mass


def find_center(img, threshold=0.1):
    """
    Calcula el centro de masa de una imagen con umbral para píxeles significativos.

    Parámetros:
    - img: array 2D - Imagen de entrada
    - threshold: float - Umbral relativo (0-1) para considerar píxeles significativos (default: 0.1)

    Retorna:
    - tuple (cx, cy) - Coordenadas del centro de masa en píxeles (x, y)

    Nota: Esta función normaliza la imagen y aplica un umbral antes de calcular
    el centro de masa para evitar que el ruido de fondo afecte el resultado.
    """
    normalized = img - np.min(img)
    if np.max(normalized) > 0:
        normalized = normalized / np.max(normalized)

    mask = normalized > threshold

    y_indices, x_indices = np.indices(img.shape)

    total_mass = np.sum(normalized[mask])
    if total_mass > 0:
        com_y = np.sum(y_indices[mask] * normalized[mask]) / total_mass
        com_x = np.sum(x_indices[mask] * normalized[mask]) / total_mass
    else:
        com_y, com_x = np.array(img.shape) // 2

    return com_x, com_y


def load_fits_image(path):
    """
    Carga una imagen FITS y la convierte a float64.

    Parámetros:
    - path: str - Ruta al archivo FITS

    Retorna:
    - array 2D - Datos de la imagen en formato float64

    Raises:
    - FileNotFoundError: Si el archivo no existe
    - OSError: Si el archivo FITS está corrupto
    """
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)


def apply_mask(img, mask):
    """
    Aplica una máscara binaria a una imagen.

    Parámetros:
    - img: array 2D - Imagen de entrada
    - mask: array 2D - Máscara binaria (0 o 1)

    Retorna:
    - array 2D - Imagen enmascarada (píxeles fuera de la máscara = 0)
    """
    return img * mask


def calculate_center_of_mass(image):
    """
    Alias de find_center para retrocompatibilidad.

    DEPRECATED: Usar find_center() directamente.
    """
    cx, cy = find_center(image, threshold=0.1)
    return int(cy), int(cx)
