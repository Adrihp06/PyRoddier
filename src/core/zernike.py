# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

import numpy as np
from scipy.special import factorial as fact


def zernike_polynomials(shape, mask, R_out, center, max_terms=23):
    """
    Genera la base de polinomios de Zernike ortonormalizados según el índice de Noll.

    Parámetros:
    - shape: tuple (alto, ancho) - Dimensiones de la imagen en píxeles
    - mask: array 2D - Máscara binaria de la pupila (anular), valores 0 o 1
    - R_out: float - Radio exterior de la pupila en píxeles
    - center: tuple (cx, cy) - Centro de la pupila en píxeles
    - max_terms: int - Número máximo de términos de Zernike a generar (por defecto 23)

    Retorna:
    - base: array 3D - Array de forma (max_terms, height, width) con los polinomios
            ortonormalizados según la convención de Noll. Normalización: sqrt(2(n+1))
            para m≠0, sqrt(n+1) para m=0.
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
            num = (-1) ** k * fact(n - k)
            den = fact(k) * fact((n + abs(m)) // 2 - k) * fact((n - abs(m)) // 2 - k)
            Rnm += num / den * r ** (n - 2 * k)
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
        (0, 0),  # j=1
        (1, -1),  # j=2
        (1, 1),  # j=3
        (2, -2),  # j=4
        (2, 0),  # j=5
        (2, 2),  # j=6
        (3, -3),  # j=7
        (3, -1),  # j=8
        (3, 1),  # j=9
        (3, 3),  # j=10
        (4, -4),  # j=11
        (4, -2),  # j=12
        (4, 0),  # j=13
        (4, 2),  # j=14
        (4, 4),  # j=15
        (5, -5),  # j=16
        (5, -3),  # j=17
        (5, -1),  # j=18
        (5, 1),  # j=19
        (5, 3),  # j=20
        (5, 5),  # j=21
        (6, -6),  # j=22
        (6, -4),  # j=23
        (6, -2),  # j=24
        (6, 0),  # j=25
        (6, 2),  # j=26
        (6, 4),  # j=27
        (6, 6),  # j=28
    ]

    base = []
    for idx, (n, m) in enumerate(noll_indices[:max_terms]):
        Znm = Z(n, m, r, theta)

        # Normalización analítica (como hace WinRoddier)
        norm_factor = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
        Znm *= norm_factor

        # Aplicar máscara para limitar el dominio
        Znm *= mask
        # Znm /= np.sqrt(np.sum(Znm**2 * mask))
        base.append(Znm)

    return np.array(base)


def fit_zernike(wavefront, mask, R_out, center, max_order=23):
    """Ajusta los coeficientes de Zernike al frente de onda.

    Args:
        wavefront: array 2D con el frente de onda en longitudes de onda (λ)
        mask: array 2D con la máscara binaria de la pupila (valores 0 o 1)
        R_out: radio exterior de la pupila en píxeles
        center: centro de la pupila (y, x) en píxeles
        max_order: orden máximo de los polinomios de Zernike (por defecto 23)

    Returns:
        tuple: (coeficientes, base)
            - coeficientes: array 1D con los coeficientes de Zernike en longitudes de onda (λ)
            - base: array 3D con la base de polinomios de Zernike ortonormalizados

    Nota:
        Los coeficientes están en las mismas unidades que el wavefront de entrada.
        Si el wavefront está en λ, los coeficientes estarán en λ.
    """
    # Validar entradas
    if wavefront is None or mask is None:
        raise ValueError("wavefront y mask no pueden ser None")

    if not np.any(mask):
        raise ValueError("La máscara debe tener al menos un píxel válido")

    if R_out <= 0:
        raise ValueError("R_out debe ser positivo")

    if max_order <= 0:
        raise ValueError("max_order debe ser positivo")

    # Validar que las dimensiones coinciden
    if wavefront.shape != mask.shape:
        raise ValueError(
            f"Las dimensiones del frente de onda {wavefront.shape} no coinciden con la máscara {mask.shape}"
        )

    # Validar que el frente de onda es finito
    if not np.all(np.isfinite(wavefront)):
        raise ValueError("El frente de onda contiene valores no finitos (NaN o Inf)")

    try:
        # Calcular la base de Zernike
        base = zernike_polynomials(wavefront.shape, mask, R_out, center, max_order)

        if base is None or len(base) == 0:
            raise ValueError("No se pudo generar la base de polinomios de Zernike")

        masked_wavefront = wavefront[mask]
        masked_base = base[:, mask]

        # Verificar que tenemos suficientes datos
        if len(masked_wavefront) == 0:
            raise ValueError("No hay datos válidos después de aplicar la máscara")

        if masked_base.shape[1] == 0:
            raise ValueError("La base enmascarada está vacía")

        # Verificar que tenemos más puntos que coeficientes para un ajuste válido
        if masked_base.shape[1] < masked_base.shape[0]:
            raise ValueError(
                f"Insuficientes puntos de datos ({masked_base.shape[1]}) para {masked_base.shape[0]} coeficientes"
            )

        # Mínimos cuadrados para obtener coeficientes
        coeffs, residuals, rank, s = np.linalg.lstsq(
            masked_base.T, masked_wavefront, rcond=None
        )

        # Verificar que el ajuste fue exitoso
        if len(coeffs) != max_order:
            raise ValueError(
                f"Se esperaban {max_order} coeficientes, pero se obtuvieron {len(coeffs)}"
            )

        # Verificar que los coeficientes son finitos
        if not np.all(np.isfinite(coeffs)):
            raise ValueError("Los coeficientes ajustados contienen valores no finitos")

        return coeffs, base

    except np.linalg.LinAlgError as e:
        raise ValueError(f"Error en el ajuste de mínimos cuadrados: {e}")
    except Exception as e:
        raise ValueError(f"Error inesperado en fit_zernike: {e}")


def calculate_rms(coefficients, exclude_piston=True):
    """
    Calcula el RMS de los coeficientes de Zernike

    Args:
        coefficients: array de coeficientes de Zernike en longitudes de onda (λ)
        exclude_piston: si True, excluye el término de pistón (Z1) del cálculo

    Returns:
        float: valor RMS de los coeficientes en longitudes de onda (λ)

    Nota:
        El RMS se calcula como: RMS = sqrt(mean(coeffs²))
        Las unidades del resultado son las mismas que las de los coeficientes de entrada.
    """
    if exclude_piston and len(coefficients) > 1:
        # Excluir el primer coeficiente (pistón)
        coeffs_for_rms = coefficients[1:]
    else:
        coeffs_for_rms = coefficients

    return np.sqrt(np.mean(coeffs_for_rms**2))


def calculate_ptv(wavefront, mask):
    """
    Calcula el Peak-to-Valley (PTV) del frente de onda dentro de la pupila.

    El PTV es la diferencia entre el valor máximo y mínimo del frente de onda
    en la región válida de la pupila.

    Args:
        wavefront: array 2D con el frente de onda en longitudes de onda (λ)
        mask: array 2D con la máscara binaria de la pupila (valores 0 o 1)

    Returns:
        float: valor PTV (Peak-to-Valley) en longitudes de onda (λ)

    Raises:
        ValueError: si wavefront o mask son None, o si no hay píxeles válidos en la máscara

    Nota:
        El PTV se calcula como: PTV = max(wavefront) - min(wavefront)
        Solo se consideran los píxeles dentro de la pupila (mask == 1).
        Las unidades del resultado son las mismas que las del wavefront de entrada.
    """
    if wavefront is None or mask is None:
        raise ValueError("wavefront y mask no pueden ser None")

    if not np.any(mask):
        raise ValueError("La máscara debe tener al menos un píxel válido")

    if wavefront.shape != mask.shape:
        raise ValueError(
            f"Las dimensiones del frente de onda {wavefront.shape} no coinciden con la máscara {mask.shape}"
        )

    # Extraer solo los valores dentro de la pupila
    wavefront_masked = wavefront[mask > 0]

    if len(wavefront_masked) == 0:
        raise ValueError("No hay datos válidos después de aplicar la máscara")

    # Calcular PTV
    ptv = np.max(wavefront_masked) - np.min(wavefront_masked)

    return ptv

