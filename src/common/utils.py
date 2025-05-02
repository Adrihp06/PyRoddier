# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from astropy.io import fits
import numpy as np
from scipy.ndimage import center_of_mass
def find_center(img):
    cy, cx = center_of_mass(img)
    return cx, cy

def load_fits_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)

def apply_mask(img, mask):
    return img * mask

def calculate_center_of_mass(image):
    """Calcula el centro de masa de la imagen."""
    # Normalizar la imagen para el cálculo
    normalized = image - np.min(image)
    if np.max(normalized) > 0:
        normalized = normalized / np.max(normalized)

    # Crear máscaras para los píxeles significativos
    threshold = 0.1  # Ajusta este valor según sea necesario
    mask = normalized > threshold

    # Calcular índices de las coordenadas
    y_indices, x_indices = np.indices(image.shape)

    # Calcular centro de masa solo de los píxeles significativos
    total_mass = np.sum(normalized[mask])
    if total_mass > 0:
        com_y = np.sum(y_indices[mask] * normalized[mask]) / total_mass
        com_x = np.sum(x_indices[mask] * normalized[mask]) / total_mass
    else:
        # Si no hay píxeles significativos, usar el centro geométrico
        com_y, com_x = np.array(image.shape) // 2

    return int(com_y), int(com_x)