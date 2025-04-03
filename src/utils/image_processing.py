from astropy.io import fits
import numpy as np
from scipy.ndimage import gaussian_filter, center_of_mass, shift
from skimage import measure
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter


def load_fits_image(file_path):
    with fits.open(file_path) as hdul:
        return hdul[0].data

def filter_noise(img, gaussian_sigma=1, threshold_ratio=0.09):
    smoothed_img = gaussian_filter(img, sigma=gaussian_sigma)
    smoothed_img = smoothed_img / np.max(smoothed_img)
    threshold = threshold_ratio * np.max(smoothed_img)
    binary_mask = smoothed_img > threshold
    return smoothed_img * binary_mask

def center_image(img, mask):
    """
    Centra la imagen exactamente como WinRoddier:
    - Calcula el centroide solo sobre la zona enmascarada.
    - Desplaza la imagen para colocar el centroide en el centro exacto del arreglo.

    Parámetros:
    - img: Imagen original.
    - mask: Máscara binaria (pupila efectiva).

    Retorna:
    - Imagen alineada con el centroide en el centro exacto del arreglo.
    """
    # Aplicar máscara para calcular centroide preciso
    masked_img = img * mask

    # Calcular centroide exacto sobre la zona efectiva
    cy, cx = center_of_mass(masked_img)

    # Centro del arreglo (exacto)
    img_center = (np.array(img.shape) - 1) / 2.0

    # Calcular desplazamiento exacto
    shift_values = img_center - np.array([cy, cx])

    # Aplicar desplazamiento (subpíxel)
    aligned_img = shift(img, shift_values, order=3, mode='constant', cval=0)

    return aligned_img

def crop_center(img, crop_size):
    """
    Recorta la imagen centrada exactamente en el centro del arreglo, igual que WinRoddier.

    Parámetros:
    - img: Imagen alineada.
    - crop_size: Tamaño deseado del recorte (píxeles).

    Retorna:
    - Imagen recortada centrada.
    """
    center_y, center_x = np.array(img.shape) // 2
    half_size = crop_size // 2
    return img[
        center_y - half_size : center_y + half_size,
        center_x - half_size : center_x + half_size
    ]

def align_images_winroddier(intra_img, extra_img, masking_value=0.3):
    # 1. Máscaras individuales con umbral relativo
    threshold_intra = masking_value * intra_img.max()
    threshold_extra = masking_value * extra_img.max()
    mask_intra = intra_img > threshold_intra
    mask_extra = extra_img > threshold_extra

    # 2. Máscara común
    common_mask = mask_intra & mask_extra

    # 3. Aplicar máscara a las imágenes originales
    intra_masked = intra_img * common_mask
    extra_masked = extra_img * common_mask

    # 4. Suavizado ligero para mejorar la correlación
    intra_masked = gaussian_filter(intra_masked, sigma=1)
    extra_masked = gaussian_filter(extra_masked, sigma=1)

    # 5. Correlación cruzada (FFT)
    corr = fftconvolve(intra_masked, extra_masked[::-1, ::-1], mode='same')

    # 6. Obtener desplazamiento
    max_corr_pos = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    center = np.array(intra_img.shape) // 2
    shift_values = center - max_corr_pos

    # 7. Aplicar desplazamiento al extra_focal
    extra_aligned = shift(extra_img, shift_values, order=3, mode='constant', cval=0)
    mask_extra_aligned = shift(mask_extra.astype(float), shift_values, order=0) > 0.5

    # 8. Máscara común final sobre imágenes alineadas
    common_mask_final = mask_intra & mask_extra_aligned
    intra_aligned = intra_img * common_mask_final
    extra_aligned = extra_aligned * common_mask_final

    return intra_aligned, extra_aligned, common_mask_final



def generate_common_annular_mask(intra_aligned, extra_aligned, masking_value=0.5,
                                 secondary_diameter_m=0.0, primary_diameter_m=1.0, pixel_scale_m=1e-6):
    """
    Genera una máscara anular común exactamente como WinRoddier:
    - Máscara individual para cada imagen alineada.
    - Máscara común por intersección lógica (AND).
    - Usa diámetro físico del espejo secundario y primario para radio interior.
    """

    """
    Genera la máscara anular exactamente como WinRoddier:
    - Suma las imágenes alineadas.
    - Aplica umbral relativo sobre la imagen combinada.
    - Calcula centro y radios desde la máscara.
    """

    # Imagen combinada
    combined = intra_aligned + extra_aligned
    # Máscara binaria sobre intensidad combinada
    threshold = masking_value * combined.max()
    mask_combined = combined > threshold

    # Calcular centroide y radios
    cy, cx = center_of_mass(mask_combined.astype(float))
    y, x = np.indices(mask_combined.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    R_out = r[mask_combined].max()
    obstruction_ratio = secondary_diameter_m / primary_diameter_m
    R_in = obstruction_ratio * R_out

    annular_mask = (r <= R_out) & (r >= R_in)

    return annular_mask, (cx, cy), R_in, R_out
# Ejemplo de uso:
# annular_mask, center, R_in, R_out = generate_annular_mask_winroddier(img, 0.5, 0.3, 1.0, 1e-6)



def preprocess_images(intra_image, extra_image):
    """Preprocesa las imágenes intra-focal y extra-focal."""
    # Aplicar filtro gaussiano para reducir el ruido
    intra_smooth = gaussian_filter(intra_image, sigma=2.0)
    extra_smooth = gaussian_filter(extra_image, sigma=2.0)

    # Normalizar las imágenes
    intra_norm = (intra_smooth - np.min(intra_smooth)) / (np.max(intra_smooth) - np.min(intra_smooth))
    extra_norm = (extra_smooth - np.min(extra_smooth)) / (np.max(extra_smooth) - np.min(extra_smooth))

    return intra_norm, extra_norm


def calculate_center_of_mass(image):
    """Calcula el centro de masa de la imagen."""
    # Generar máscara binaria
    mask = generate_binary_mask(image)

    # Calcular el centro de masa
    com_y, com_x = measure.center_of_mass(mask)

    return int(com_y), int(com_x)

def normalize_images(intra, extra):
    """Iguala la energía total de las imágenes recortadas."""
    sum_intra = np.sum(intra)
    sum_extra = np.sum(extra)
    if sum_extra == 0:
        raise ValueError("Extra-focal con suma cero")
    extra_scaled = extra * (sum_intra / sum_extra)
    return intra, extra_scaled