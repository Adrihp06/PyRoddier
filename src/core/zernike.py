import numpy as np

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

def zernike_polynomials(max_order, R_in, R_out, shape, center):
    """
    Genera una base de polinomios de Zernike ortonormal sobre una pupila anular.

    Parámetros:
    - max_order: orden máximo n de los polinomios Zernike
    - R_in: radio interior (obstrucción central)
    - R_out: radio exterior de la pupila
    - shape: tupla (h, w) con el tamaño de la imagen
    - center: (cx, cy) centro de la pupila

    Retorna:
    - zernike_stack: array (n_polinomios, h, w) con cada modo
    - mask: máscara binaria de la pupila anular
    """
    h, w = shape
    y, x = np.indices((h, w))
    cy, cx = center
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Normalizar r sobre la pupila anular a [0, 1]
    rho = (r - R_in) / (R_out - R_in)
    mask = (rho >= 0) & (rho <= 1)

    zernike_stack = []

    for n in range(max_order + 1):
        for m in range(-n, n + 1, 2):
            Rnm = zernike_radial(n, m, rho)
            if m == 0:
                Z = Rnm
            elif m > 0:
                Z = Rnm * np.cos(m * theta)
            else:
                Z = Rnm * np.sin(-m * theta)

            # Normalización como en Noll (anular): sqrt(2(n+1)) excepto para m=0
            norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
            Z *= norm

            # Aplicar máscara
            Z *= mask

            zernike_stack.append(Z)

    zernike_stack = np.array(zernike_stack)
    return zernike_stack, mask

def fit_zernike(wavefront, mask, center, max_order=10):
    """
    Ajusta una serie de polinomios de Zernike al frente de onda proporcionado.

    Retorna:
    - wavefront_reconstruido
    - coeficientes_zernike
    - base_zernike
    """
    R_out = np.sqrt(np.max((mask * ((np.indices(mask.shape)[1] - center[0]) ** 2 +
                                     (np.indices(mask.shape)[0] - center[1]) ** 2))))
    R_in = R_out * 0  # asume sin obstrucción si no se conoce

    base, _ = zernike_polynomials(max_order, R_in, R_out, wavefront.shape, center)

    masked_wavefront = wavefront[mask]
    masked_base = base[:, mask]

    # Mínimos cuadrados para obtener coeficientes
    coeffs, *_ = np.linalg.lstsq(masked_base.T, masked_wavefront, rcond=None)

    # Reconstrucción del frente de onda desde la base y coeficientes
    wavefront_reconstructed = np.sum(base * coeffs[:, None, None], axis=0) * mask

    return wavefront_reconstructed, coeffs, base
