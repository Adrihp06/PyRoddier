import numpy as np
from scipy.ndimage import shift
from scipy.signal import fftconvolve
from scipy.fft import fft2, ifft2, fftfreq


def _calculate_wavefront(intra_focal_image, extra_focal_image, annular_mask, center):
    """
    Calcula y reconstruye el frente de onda a partir de las imágenes intrafocal y extrafocal,
    aplicando el test de Roddier y la máscara anular generada con precisión.

    Parámetros:
    -----------
    intra_focal_image : ndarray
        Imagen intrafocal (preprocesada).
    extra_focal_image : ndarray
        Imagen extrafocal (preprocesada y alineada).
    annular_mask : ndarray
        Máscara anular binaria generada por la función propuesta.
    center : tuple (cx, cy)
        Coordenadas del centro exacto de la pupila, obtenidas con la máscara.

    Retorna:
    --------
    wavefront_clean : ndarray
        Frente de onda reconstruido (con piston y tilt removidos).
    """
    pupil_mask_float = annular_mask.astype(float)

    # Diferencia normalizada (Test de Roddier)
    epsilon = 1e-8
    image_sum = intra_focal_image + extra_focal_image + epsilon
    image_diff = intra_focal_image - extra_focal_image
    normalized_diff = image_diff / image_sum

    # FFT de la imagen diferencial normalizada
    normalized_diff_fft = fft2(normalized_diff)

    # Construcción de frecuencias espaciales
    height, width = normalized_diff.shape
    freq_x = fftfreq(width)
    freq_y = fftfreq(height)
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)
    freq_squared = freq_x_grid**2 + freq_y_grid**2

    # Componente DC (frecuencia cero)
    normalized_diff_fft[0, 0] = 0.0

    # Cálculo del frente de onda en espacio de Fourier
    wavefront_fft = np.zeros_like(normalized_diff_fft, dtype=complex)
    nonzero_freq = freq_squared > 1e-8
    wavefront_fft[nonzero_freq] = normalized_diff_fft[nonzero_freq] / (-freq_squared[nonzero_freq])

    # Frente de onda en espacio real (transformada inversa)
    wavefront = ifft2(wavefront_fft).real

    # Aplicación directa de la máscara anular
    wavefront *= pupil_mask_float

    # === CORRECCIÓN PRINCIPAL: uso del centro real (center) ===
    cx, cy = center  # centro exacto obtenido con la máscara

    # Coordenadas respecto al centro real de la pupila
    y_coords, x_coords = np.indices(wavefront.shape)
    y_centered = y_coords - cy
    x_centered = x_coords - cx

    # Selección de píxeles válidos dentro de la pupila
    pupil_indices = pupil_mask_float == 1
    pupil_y = y_centered[pupil_indices].flatten()
    pupil_x = x_centered[pupil_indices].flatten()
    pupil_wavefront = wavefront[pupil_indices].flatten()

    # Resolver el plano piston/tip/tilt sobre píxeles válidos
    design_matrix = np.vstack([pupil_x, pupil_y, np.ones_like(pupil_x)]).T
    coefficients, *_ = np.linalg.lstsq(design_matrix, pupil_wavefront, rcond=None)
    tilt_x, tilt_y, piston = coefficients

    # Construir el plano y restarlo del frente de onda
    tilt_plane = piston +tilt_x * x_centered + tilt_y * y_centered
    wavefront = (wavefront - tilt_plane) * pupil_mask_float
    import matplotlib.pyplot as plt
    plt.imshow(wavefront)
    plt.colorbar()
    plt.show()

    return wavefront

def calculate_wavefront(intra_focal_image, extra_focal_image, annular_mask, center):
    """
    Calcula el frente de onda a partir de las imágenes intra y extrafocales
    utilizando el método de Roddier, tal como lo hace WinRoddier.

    No elimina el plano de piston, tip ni tilt, ya que estos se representan
    en los primeros modos de Zernike (Z0-Z2) y WinRoddier permite conservarlos
    para su ajuste posterior.

    Parámetros:
    -----------
    intra_focal_image : ndarray
        Imagen intrafocal alineada.
    extra_focal_image : ndarray
        Imagen extrafocal alineada.
    annular_mask : ndarray
        Máscara anular binaria de la pupila.
    center : tuple (cx, cy)
        Coordenadas del centro exacto de la pupila.

    Retorna:
    --------
    wavefront : ndarray
        Frente de onda reconstruido sin eliminar modos bajos.
    """
    pupil_mask_float = annular_mask.astype(float)

    # Diferencia normalizada (Test de Roddier)
    epsilon = 1e-8
    image_sum = intra_focal_image + extra_focal_image + epsilon
    image_diff = intra_focal_image - extra_focal_image
    normalized_diff = image_diff / image_sum

    # FFT de la imagen diferencial normalizada
    normalized_diff_fft = fft2(normalized_diff)

    # Construcción de frecuencias espaciales
    height, width = normalized_diff.shape
    freq_x = fftfreq(width)
    freq_y = fftfreq(height)
    freq_x_grid, freq_y_grid = np.meshgrid(freq_x, freq_y)
    freq_squared = freq_x_grid**2 + freq_y_grid**2

    # Eliminar componente DC
    normalized_diff_fft[0, 0] = 0.0

    # Cálculo del frente de onda en el dominio de Fourier
    wavefront_fft = np.zeros_like(normalized_diff_fft, dtype=complex)
    nonzero_freq = freq_squared > 1e-8
    wavefront_fft[nonzero_freq] = normalized_diff_fft[nonzero_freq] / (-freq_squared[nonzero_freq])

    # Transformada inversa para obtener frente de onda
    wavefront = ifft2(wavefront_fft).real

    # Aplicar la máscara anular
    wavefront *= pupil_mask_float

    return wavefront
