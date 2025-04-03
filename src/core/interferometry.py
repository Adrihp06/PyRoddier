import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def calculate_interferogram(wavefront, wavelength=632.8e-9, pixel_size=6.4e-6):
    """
    Calcula el interferograma a partir del frente de onda.

    Args:
        wavefront (np.ndarray): Mapa del frente de onda
        wavelength (float): Longitud de onda en metros
        pixel_size (float): Tamaño del píxel en metros

    Returns:
        np.ndarray: Interferograma calculado
    """
    # Calcular la fase
    phase = 2 * np.pi * wavefront / wavelength

    # Calcular el interferograma
    interferogram = 0.5 * (1 + np.cos(phase))

    # Aplicar filtro gaussiano para reducir el ruido
    interferogram = gaussian_filter(interferogram, sigma=1)

    return interferogram

def analyze_interferogram(interferogram, wavelength=632.8e-9, pixel_size=6.4e-6):
    """
    Analiza el interferograma para extraer el frente de onda.

    Args:
        interferogram (np.ndarray): Interferograma a analizar
        wavelength (float): Longitud de onda en metros
        pixel_size (float): Tamaño del píxel en metros

    Returns:
        np.ndarray: Frente de onda reconstruido
    """
    # Calcular la transformada de Fourier del interferograma
    interferogram_ft = fft2(interferogram)

    # Desplazar el espectro al centro
    interferogram_ft = fftshift(interferogram_ft)

    # Extraer la fase
    phase = np.angle(interferogram_ft)

    # Calcular el frente de onda
    wavefront = phase * wavelength / (2 * np.pi)

    # Aplicar filtro gaussiano para reducir el ruido
    wavefront = gaussian_filter(wavefront, sigma=2)

    return wavefront