from scipy.fftpack import fft2, fftshift, ifft2
import numpy as np

def _calculate_phase_roddier(img1, img2):
    fft_intra = fftshift(fft2(img1))
    fft_extra = fftshift(fft2(img2))
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_diff = np.angle(np.divide(fft_extra, fft_intra, out=np.zeros_like(fft_intra), where=fft_intra != 0))
    return phase_diff

def calculate_phase_roddier(intrafocal, extrafocal):

    difference = intrafocal - extrafocal

    # Paso 4: Calcular el Laplaciano usando Transformadas de Fourier
    ny, nx = difference.shape
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    k_squared = kx**2 + ky**2
    k_squared[0, 0] = 1e-10  # Evitar división por cero

    # Transformada de Fourier del Laplaciano
    laplacian_fft = fft2(difference)

    # Resolver la ecuación de Poisson en el dominio de Fourier
    wavefront_fft = laplacian_fft / (4 * np.pi**2 * k_squared)

    # Volver al dominio espacial
    wavefront_reconstructed = np.real(ifft2(wavefront_fft))

    return wavefront_reconstructed