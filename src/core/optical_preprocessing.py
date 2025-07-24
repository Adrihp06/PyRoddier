from scipy.signal import fftconvolve
from scipy.ndimage import  shift
from src.common.utils import apply_mask, find_center
import numpy as np

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

def preprocess_roddier(intra_image, extra_image, apertura=900, focal=7200,
                          pixel_scale=15, threshold=0.5):
    """
    Preprocesa las imágenes intra y extra-focales para el análisis de Roddier.
    
    Args:
        intra_image: Imagen intra-focal
        extra_image: Imagen extra-focal
        apertura: Apertura del telescopio en mm
        focal: Distancia focal en mm
        pixel_scale: Escala de pixel en micras
        threshold: Umbral para estimación de radios
    
    Returns:
        tuple: (delta_I_norm, annular_mask, center, R_out, dz_mm)
    """
    # Validar entradas
    if intra_image is None or extra_image is None:
        raise ValueError("Las imágenes intra y extra no pueden ser None")
    
    if intra_image.shape != extra_image.shape:
        raise ValueError(f"Las imágenes deben tener el mismo tamaño: {intra_image.shape} vs {extra_image.shape}")
    
    if apertura <= 0 or focal <= 0 or pixel_scale <= 0:
        raise ValueError("Los parámetros del telescopio deben ser positivos")
    
    if not (0 < threshold < 1):
        raise ValueError("El threshold debe estar entre 0 y 1")
    
    if not (np.all(np.isfinite(intra_image)) and np.all(np.isfinite(extra_image))):
        raise ValueError("Las imágenes contienen valores no finitos")

    try:
        extra_aligned, _ = align_images(intra_image, extra_image)
        intra_aligned = intra_image
        
        # Normalizar imágenes entre 0 y 1 (ambas con los mismos límites)
        img_avg = 0.5 * (intra_aligned + extra_aligned)
        
        # Verificar que la imagen promedio es válida
        if not np.any(img_avg > 0):
            raise ValueError("La imagen promedio no contiene valores positivos")
        
        cx, cy = find_center(img_avg)
        
        # Validar centro
        if not (0 <= cx < img_avg.shape[1] and 0 <= cy < img_avg.shape[0]):
            raise ValueError(f"Centro calculado ({cx}, {cy}) está fuera de los límites de la imagen")
        
        R_out, R_in = estimate_radii(img_avg, cx, cy, threshold=threshold)
        
        # Validar radios
        if R_out <= 0 or R_in < 0 or R_in >= R_out:
            raise ValueError(f"Radios inválidos: R_in={R_in}, R_out={R_out}")
        
        # Verificar que los radios están dentro de la imagen
        max_radius = min(cx, cy, img_avg.shape[1] - cx, img_avg.shape[0] - cy)
        if R_out > max_radius:
            raise ValueError(f"Radio exterior {R_out} es mayor que el máximo posible {max_radius}")
        
        dz_mm = estimate_defocus_mm(R_out, pixel_scale, focal, apertura)
        
        # Validar defocus
        if dz_mm <= 0 or not np.isfinite(dz_mm):
            raise ValueError(f"Distancia de defocus inválida: {dz_mm}")
        
        annular_mask = generate_perfect_annular_mask(cx, cy, R_in, R_out, intra_image)
        
        # Verificar que la máscara tiene píxeles válidos
        if not np.any(annular_mask):
            raise ValueError("La máscara anular está vacía")
        
        intra_masked = apply_mask(intra_aligned, annular_mask)
        extra_masked = apply_mask(extra_aligned, annular_mask)
        delta_I = extra_masked.astype(np.float64) - intra_masked.astype(np.float64)
        I0 = 0.5 * (extra_masked + intra_masked)
        delta_I_norm = np.divide(delta_I, I0, out=np.zeros_like(delta_I), where=I0 != 0)
        
        # Verificar resultado final
        if not np.all(np.isfinite(delta_I_norm)):
            raise ValueError("El resultado delta_I_norm contiene valores no finitos")

        return delta_I_norm, annular_mask, (cx, cy), R_out, dz_mm
    
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Error inesperado en preprocesamiento: {e}")
