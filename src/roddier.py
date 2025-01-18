from scipy.fftpack import fft2, fftshift, ifft2
import numpy as np

def _calculate_phase_roddier(img1, img2):
    fft_intra = fftshift(fft2(img1))
    fft_extra = fftshift(fft2(img2))
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_diff = np.angle(np.divide(fft_extra, fft_intra, out=np.zeros_like(fft_intra), where=fft_intra != 0))
    return phase_diff

def calculate_phase_roddier(intrafocal, extrafocal):
    """
    Calcula el frente de onda a partir de las imágenes intra-focal y extra-focal usando el Test de Roddier.

    Args:
        intrafocal (ndarray): Imagen intra-focal.
        extrafocal (ndarray): Imagen extra-focal.
        focal_length (float): Longitud focal del sistema óptico.
        delta_z (float): Diferencia de desenfoque (intrafocal - extrafocal).

    Returns:
        wavefront_reconstructed (ndarray): Frente de onda reconstruido.
    """
    # Paso 1: Calcular la diferencia escalada
    difference = (intrafocal - extrafocal)#/ delta_z

    # Paso 2: Transformada de Fourier para resolver la ecuación de Poisson
    ny, nx = difference.shape
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1e-10  # Evitar división por cero

    # Paso 3: Transformada de Fourier del Laplaciano
    laplacian_fft = np.fft.fft2(difference)

    # Paso 4: Resolver la ecuación de Poisson
    wavefront_fft = laplacian_fft / (4 * np.pi**2 * k_squared)

    # Paso 5: Volver al dominio espacial
    wavefront_reconstructed = np.real(np.fft.ifft2(wavefront_fft))

    return wavefront_reconstructed