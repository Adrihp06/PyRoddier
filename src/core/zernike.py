# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from scipy.special import factorial as fact

def zernike_radial(n, m, rho):
    """
    Cálculo del polinomio radial Zernike R_n^m(rho)
    """
    R = np.zeros_like(rho)
    m = abs(m)
    for k in range((n - m) // 2 + 1):
        coeff = ((-1) ** k * np.math.factorial(n - k)) /\
                (np.math.factorial(k) *
                 np.math.factorial((n + m) // 2 - k) *
                 np.math.factorial((n - m) // 2 - k))
        R += coeff * rho ** (n - 2 * k)
    return R

def zernike_polynomials(shape, mask, R_out, center, max_terms=23):
    """
    Parámetros:
    - shape: (alto, ancho) de la imagen
    - mask: máscara binaria de la pupila (anular)
    - center: (cx, cy) centro de la pupila
    - max_order: orden máximo de los polinomios (por defecto 23)

    Retorna:
    - base: lista de arrays 2D con los polinomios ortonormalizados
    """
    y, x = np.indices(shape)
    cy, cx = center
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    r /= R_out
    r[mask == 0] = 0
    theta[mask == 0] = 0

    def R(n, m, r):
        Rnm = np.zeros_like(r)
        for k in range((n - abs(m)) // 2 + 1):
            num = (-1)**k * fact(n - k)
            den = fact(k) * fact((n + abs(m)) // 2 - k) * fact((n - abs(m)) // 2 - k)
            Rnm += num / den * r**(n - 2 * k)
        return Rnm

    def Z(n, m, r, theta):
        if m == 0:
            return R(n, 0, r)
        elif m > 0:
            return R(n, m, r) * np.cos(m * theta)
        else:
            return R(n, -m, r) * np.sin(-m * theta)

    # Orden de Noll
    noll_indices = [
        (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
        (3, -1), (3, 1), (3, -3), (3, 3), (4, 0),
        (4, -2), (4, 2), (5, -1), (5, 1), (5, -3), (5, 3),
        (5, -5), (5, 5), (6, 0), (6, -2), (6, 2), (6, -4)
    ]

    base = []
    for idx, (n, m) in enumerate(noll_indices[:max_terms]):
        Znm = Z(n, m, r, theta)

        # Normalización analítica (como hace WinRoddier)
        norm_factor = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
        Znm *= norm_factor

        # Aplicar máscara para limitar el dominio
        Znm *= mask
        #Znm /= np.sqrt(np.sum(Znm**2 * mask))
        base.append(Znm)

    return np.array(base)

def fit_zernike(wavefront, mask, R_out, center, max_order=23):
    """Ajusta los coeficientes de Zernike al frente de onda.

    Args:
        wavefront: array 2D con el frente de onda
        mask: array 2D con la máscara
        R_out: radio exterior de la pupila
        center: centro de la pupila (y, x)
        max_order: orden máximo de los polinomios (por defecto 23)

    Returns:
        tuple: (coeficientes, base)
    """
    # Calcular la base de Zernike
    base = zernike_polynomials(wavefront.shape, mask, R_out, center, max_order)

    masked_wavefront = wavefront[mask]
    masked_base = base[:, mask]

    # Mínimos cuadrados para obtener coeficientes
    coeffs, *_ = np.linalg.lstsq(masked_base.T, masked_wavefront, rcond=None)
    return coeffs, base