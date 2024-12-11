from zernike import RZern
import numpy as np
from scipy.linalg import lstsq
from zernpy import ZernPol


def calculate_wavefront_zernike(phase_diff, mask, num_terms=10):
    """
    Calcula los coeficientes de Zernike y reconstruye el frente de onda utilizando un ajuste
    en la región del donut definida por la máscara.

    Parameters:
        phase_diff (np.ndarray): Interferograma obtenido de las imágenes intra y extra focal.
        mask (np.ndarray): Máscara binaria que define la región válida (suma de las imágenes intra y extra focal).
        num_terms (int): Número de términos de Zernike a considerar.

    Returns:
        zernike_coeffs (np.ndarray): Coeficientes ajustados.
        phi_reconstructed (np.ndarray): Frente de onda reconstruido.
    """
    ny, nx = phase_diff.shape

    # Crear un grid cartesiano
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Verificar que la máscara contiene datos válidos
    if np.sum(mask) == 0:
        raise ValueError("La máscara no contiene datos válidos.")

    # Aplicar la máscara para seleccionar solo el frente de onda en la región válida
    wavefront_masked = np.where(mask, phase_diff, np.nan)

    # Generar polinomios de Zernike
    zern = RZern(num_terms)
    zern.make_cart_grid(X, Y)

    # Ajustar los coeficientes de Zernike usando el grid cartesiano
    zernike_coeffs = zern.fit_cart_grid(wavefront_masked)[0]

    # Reconstruir el frente de onda a partir de los coeficientes
    phi_reconstructed = zern.eval_grid(zernike_coeffs, matrix=True)
    phi_reconstructed_masked = np.where(mask, phi_reconstructed, np.nan)

    return zernike_coeffs, phi_reconstructed_masked


def recalculate_wavefront_zernike(phase_diff, mask, zernike_coeffs):
        """
        Reconstruye el frente de onda utilizando los coeficientes de Zernike previamente calculados.

        Parameters:
                phase_diff (np.ndarray): Interferograma obtenido de las imágenes intra y extra focal.
                mask (np.ndarray): Máscara binaria que define la región válida (suma de las imágenes intra y extra focal).
                zernike_coeffs (np.ndarray): Coeficientes de Zernike previamente calculados.

        Returns:
                phi_reconstructed (np.ndarray): Frente de onda reconstruido.
        """
        ny, nx = phase_diff.shape

        # Crear un grid cartesiano
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Verificar que la máscara contiene datos válidos
        if np.sum(mask) == 0:
                raise ValueError("La máscara no contiene datos válidos.")

        # Generar polinomios de Zernike
        zern = RZern(len(zernike_coeffs))
        zern.make_cart_grid(X, Y)

        # Reconstruir el frente de onda a partir de los coeficientes
        phi_reconstructed = zern.eval_grid(zernike_coeffs, matrix=True)
        phi_reconstructed_masked = np.where(mask, phi_reconstructed, np.nan)

        return phi_reconstructed_masked

def _calculate_wavefront_zernike(phase_diff, mask, num_terms=10):
    """
    Calcula los coeficientes de Zernike y reconstruye el frente de onda utilizando un ajuste
    en la región del donut definida por la máscara, usando zernpy.

    Parameters:
        phase_diff (np.ndarray): Interferograma obtenido de las imágenes intra y extra focal.
        mask (np.ndarray): Máscara binaria que define la región válida (suma de las imágenes intra y extra focal).
        num_terms (int): Número de términos de Zernike a considerar.

    Returns:
        zernike_coeffs (np.ndarray): Coeficientes ajustados.
        phi_reconstructed (np.ndarray): Frente de onda reconstruido.
    """
    # Dimensiones del frente de onda
    ny, nx = phase_diff.shape

    # Crear un grid cartesiano
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Verificar que la máscara contiene datos válidos
    if np.sum(mask) == 0:
        raise ValueError("La máscara no contiene datos válidos.")

    # Aplicar la máscara para seleccionar solo el frente de onda en la región válida
    wavefront_masked = np.where(mask, phase_diff, 0)

    # Generar la máscara radial basada en el disco unitario
    radial_mask = R <= 1
    combined_mask = mask & radial_mask

    # Aplanar los datos para ajustarlos
    valid_indices = combined_mask.ravel()
    wavefront_flat = wavefront_masked.ravel()[valid_indices]
    X_flat = X.ravel()[valid_indices]
    Y_flat = Y.ravel()[valid_indices]

    # Crear un objeto de polinomios de Zernike usando zernpy
    zern = ZernPol(normalization='noll', mode='polar')

    # Calcular los coeficientes de Zernike
    r = np.sqrt(X_flat**2 + Y_flat**2)
    theta = np.arctan2(Y_flat, X_flat)
    zernike_coeffs = zern.fit(r, theta, wavefront_flat, num_terms)

    # Reconstruir el frente de onda
    R_full = np.sqrt(X**2 + Y**2)
    Theta_full = np.arctan2(Y, X)
    phi_reconstructed = zern.eval(R_full, Theta_full, zernike_coeffs)
    phi_reconstructed_masked = np.where(mask, phi_reconstructed, np.nan)

    return zernike_coeffs, phi_reconstructed_masked

