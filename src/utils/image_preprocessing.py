# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from astropy.io import fits
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass, shift

def load_fits_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)

def find_center(img):
    cy, cx = center_of_mass(img)
    return cx, cy

def align_images(intra_img, extra_img):
    corr = fftconvolve(intra_img, extra_img[::-1, ::-1], mode='same')
    max_corr_pos = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    center = np.array(intra_img.shape) // 2
    shift_values = center - max_corr_pos
    extra_aligned = shift(extra_img, shift_values, order=3, mode='constant', cval=0)

    return extra_aligned, shift_values

def generate_annular_mask(intra, extra_aligned):
    valid_intra = intra > (0.05 * intra.max())
    valid_extra = extra_aligned > (0.05 * extra_aligned.max())
    return valid_intra & valid_extra

def generate_perfect_annular_mask(cx, cy, R_in, R_out, img):
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (r >= R_in) & (r <= R_out)


def apply_mask(img, mask):
    return img * mask

def estimate_radii(img, cx, cy, threshold=0.5):
    max_val = img.max()
    mask = img > (threshold * max_val)
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_vals = r[mask]
    R_out = np.max(r_vals)

    R_in = np.min(r_vals)
    return R_out, R_in

def estimate_defocus_mm(r_px, pixel_size_um, focal_length_mm, aperture_mm):
    """
    Estima la cantidad de desenfoque (dz) en milímetros usando el radio observado de la imagen desenfocada.

    Parámetros:
    - r_px: radio en píxeles del patrón desenfocado (estimado sobre la imagen promedio)
    - pixel_size_um: tamaño de píxel en micras
    - focal_length_mm: focal del telescopio en mm
    - aperture_mm: apertura del telescopio en mm

    Retorna:
    - dz_mm: distancia de desenfoque en milímetros
    """
    pixel_size_mm = pixel_size_um / 1000
    theta = np.arctan((aperture_mm / 2) / focal_length_mm)
    dz_mm = (r_px * pixel_size_mm) / np.tan(theta)
    return dz_mm

def preprocess_winroddier(intra_image, extra_image, apertura=900, focal=7200,
                          pixel_scale=15):

    extra_aligned, _ = align_images(intra_image, extra_image)
    intra_aligned = intra_image
    # Normalizar imágenes entre 0 y 1 (ambas con los mismos límites)
    img_avg = 0.5 * (intra_aligned + extra_aligned)
    cx, cy = find_center(img_avg)
    R_out, R_in = estimate_radii(img_avg, cx, cy, threshold=0.3)
    dz_mm = estimate_defocus_mm(R_out, pixel_scale, focal, apertura)
    annular_mask = generate_perfect_annular_mask(cx, cy, R_in, R_out, intra_image)
    intra_masked = apply_mask(intra_aligned, annular_mask)
    extra_masked = apply_mask(extra_aligned, annular_mask)
    delta_I = extra_masked.astype(np.float64) - intra_masked.astype(np.float64)
    I0 = 0.5 * (extra_masked + intra_masked)
    delta_I_norm = np.divide(delta_I, I0, out=np.zeros_like(delta_I), where=I0 != 0)

    return delta_I_norm, annular_mask, (cx, cy), R_out, dz_mm
