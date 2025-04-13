import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def calculate_wavefront(delta_I_norm, annular_mask, wavelength_nm=555, dz_mm=None, subtract_tilt_and_defocus=True):
    """
    Calcula el frente de onda a partir de la diferencia normalizada ΔI/I₀
    utilizando el método de Roddier, tal como lo hace WinRoddier.

    Parámetros:
    - delta_I_norm: imagen de diferencia normalizada
    - annular_mask: máscara binaria anular de la pupila
    - wavelength_nm: longitud de onda en nanómetros (por defecto 555nm)
    - dz_mm: distancia de desenfoque en milímetros (si se quiere calibrar en unidades físicas)
    - subtract_tilt_and_defocus: si True, elimina piston, tilt X/Y y defocus

    Retorna:
    - wavefront: frente de onda reconstruido (en radianes si dz_mm se especifica)
    """
    pupil_mask_float = annular_mask.astype(float)

    # FFT de la diferencia normalizada
    normalized_diff_fft = fft2(delta_I_norm)

    # Construcción de frecuencias espaciales
    height, width = delta_I_norm.shape
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

    # Escalado físico si se conoce dz
    if dz_mm is not None:
        wavelength_mm = wavelength_nm / 1e6  # Convertir nm a mm
        factor = (wavelength_mm / (4 * np.pi)) * dz_mm
        wavefront *= factor

    # Aplicar la máscara anular
    wavefront *= pupil_mask_float
    return wavefront
