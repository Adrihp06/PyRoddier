from astropy.io import fits
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import center_of_mass, shift

def load_fits_image(path):
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64)

def find_center(img):
    cy, cx = center_of_mass(img)
    return cx, cy

def align_images(intra_img, extra_img):
    # 5. Correlación cruzada (FFT)
    corr = fftconvolve(intra_img, extra_img[::-1, ::-1], mode='same')

    # 6. Obtener desplazamiento
    max_corr_pos = np.array(np.unravel_index(np.argmax(corr), corr.shape))
    center = np.array(intra_img.shape) // 2
    shift_values = center - max_corr_pos

    # 7. Aplicar desplazamiento al extra_focal
    extra_aligned = shift(extra_img, shift_values, order=3, mode='constant', cval=0)

    return extra_aligned, shift_values

def generate_annular_mask(intra, extra_aligned):
    valid_intra = intra > (0.05 * intra.max())
    valid_extra = extra_aligned > (0.05 * extra_aligned.max())
    return valid_intra & valid_extra

def generate_perfect_annular_mask(cx, cy, R_in, R_out, img):
    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    return (r >= R_in) & (r <= R_out)


def apply_mask(img, mask):
    return img * mask

def estimate_radii(img, cx, cy, threshold=0.5):
    max_val = img.max()
    mask = img > (threshold * max_val)

    y, x = np.indices(img.shape)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    r_vals = r[mask]
    R_out = np.max(r_vals)

    # Estimar R_in automáticamente como el valor mínimo de r dentro del anillo
    R_in = np.min(r_vals)
    return R_out, R_in

def estimate_defocus_mm(r_px, pixel_size_um, focal_length_mm, aperture_mm):
    """
    Estima la cantidad de desenfoque (dz) en milímetros usando el radio observado de la imagen desenfocada.

    Parámetros:
    - r_px: radio en píxeles del patrón desenfocado (estimado sobre la imagen promedio)
    - pixel_size_um: tamaño de píxel en micras
    - focal_length_mm: focal del telescopio en mm
    - aperture_mm: apertura del telescopio en mm

    Retorna:
    - dz_mm: distancia de desenfoque en milímetros
    """
    # Convertir píxel a mm
    pixel_size_mm = pixel_size_um / 1000

    # Ángulo subtendido theta (en radianes)
    theta = np.arctan((aperture_mm / 2) / focal_length_mm)

    # Estimación de desenfoque físico
    dz_mm = (r_px * pixel_size_mm) / np.tan(theta)

    return dz_mm

def preprocess_winroddier(intra_image, extra_image, apertura=900, focal=7200,
                         pixel_scale=15, binning=1, iterations=6):
    """
    Preprocesado completo según metodología del artículo original (Roddier 1993).
    Incluye corrección iterativa de Tip, Tilt y Defocus.

    Parámetros:
    - intra_image, extra_image: imágenes intra y extra focales
    - apertura: diámetro del telescopio en mm
    - focal: distancia focal del telescopio en mm
    - espejo_primario: diámetro del espejo primario (apertura) en mm
    - espejo_secundario: diámetro del espejo secundario en mm
    - pixel_scale: tamaño de píxel en micras
    - binning: factor de binning aplicado
    - iterations: número de iteraciones para converger
    Retorna:
    - delta_I_norm, annular_mask, (cx, cy), R_in, R_out, dz_mm
    """

    extra_aligned, _ = align_images(intra_image, extra_image)
    intra_aligned = intra_image
    AF = 0
    for _ in range(iterations):
        img_avg = 0.5 * (intra_aligned + extra_aligned)

        # Estimación inicial de radios y centro
        cx, cy = find_center(img_avg)
        center = (cx, cy)
        R_out, R_in = estimate_radii(img_avg, cx, cy, threshold=0.5)

        annular_mask = generate_perfect_annular_mask(cx, cy, R_in, R_out, intra_image)
        intra_masked = apply_mask(intra_aligned, annular_mask)
        extra_masked = apply_mask(extra_aligned, annular_mask)

        # Diferencia normalizada
        delta_I = extra_masked.astype(np.float64) - intra_masked.astype(np.float64)
        I0 = 0.5 * (extra_masked + intra_masked)
        delta_I_norm = np.divide(delta_I, I0, out=np.zeros_like(delta_I), where=I0 != 0)

        # Wavefront y Zernike inicial
        dz_mm = estimate_defocus_mm(R_out, pixel_scale, focal + AF, apertura)

        wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)
        zernike_coeffs, *_  = fit_zernike(wavefront, annular_mask, R_out, center, max_order=4)
        print(zernike_coeffs)
        tip_x, tilt_y, defocus_z4 = zernike_coeffs[1], zernike_coeffs[2], zernike_coeffs[3]

        # Ajuste focal
        AF = 16 * (focal / apertura)**2 * defocus_z4

        # Recalcular radios tras ajuste focal
        R_out = recalculate_radius(R_out, AF, focal, pixel_scale)
        R_in = recalculate_radius(R_in, AF, focal, pixel_scale)

        # Corregir centros ópticos con Tip y Tilt
        intra_aligned = shift(intra_aligned, shift=(-tilt_y, -tip_x), order=3)
        extra_aligned = shift(extra_aligned, shift=(-tilt_y, -tip_x), order=3)

    # Calcular de nuevo con valores corregidos finales
    dz_mm = estimate_defocus_mm(R_out, pixel_scale, focal + AF, apertura)
    annular_mask = generate_perfect_annular_mask(cx, cy, R_in, R_out, intra_image)
    intra_masked = apply_mask(intra_aligned, annular_mask)
    extra_masked = apply_mask(extra_aligned, annular_mask)
    delta_I = extra_masked.astype(np.float64) - intra_masked.astype(np.float64)
    I0 = 0.5 * (extra_masked + intra_masked)
    delta_I_norm = np.divide(delta_I, I0, out=np.zeros_like(delta_I), where=I0 != 0)

    return delta_I_norm, annular_mask, (cx, cy), R_out, dz_mm


def recalculate_radius(R, AF, focal, pixel_scale_um):
    """
    Recalcula radios en píxeles tras corrección de defocus.
    """
    pixel_scale_mm = pixel_scale_um / 1000
    theta = np.arctan((R * pixel_scale_mm) / focal)
    R_corrected = (focal + AF) * np.tan(theta) / pixel_scale_mm
    return R_corrected


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

def zernike_polynomials(shape, mask, R_out, center, max_terms=23):
    """
    Genera una base ortonormal de polinomios de Zernike sobre una máscara anular,
    siguiendo el orden de Noll (como lo hace WinRoddier).

    Parámetros:
    - shape: (alto, ancho) de la imagen
    - mask: máscara binaria de la pupila (anular)
    - center: (cx, cy) centro de la pupila
    - max_terms: número máximo de términos a generar (por defecto 23)

    Retorna:
    - base: lista de arrays 2D con los polinomios ortonormalizados
    """
    import numpy as np
    from scipy.special import factorial as fact

    y, x = np.indices(shape)
    cy, cx = center
    x = x - cx
    y = y - cy
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    r /= R_out  # normalizar al radio máximo en la máscara
    r[mask == 0] = 0
    theta[mask == 0] = 0

    def R(n, m, r):
        """Polinomio radial de Zernike"""
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

    # Índices (n, m) en orden de Noll para los primeros 23 términos
    noll_indices = [
        (0, 0), (1, 1), (1, -1), (2, 0), (2, -2), (2, 2),
        (3, -1), (3, 1), (3, -3), (3, 3), (4, 0),
        (4, -2), (4, 2), (5, -1), (5, 1), (5, -3), (5, 3),
        (5, -5), (5, 5), (6, 0), (6, -2), (6, 2), (6, -4)
    ]

    base = []
    for idx, (n, m) in enumerate(noll_indices[:max_terms]):
        Znm = Z(n, m, r, theta)
        Znm *= mask

        # Normalización Noll sobre la máscara
        norm_factor = np.sqrt((2 * (n + 1)) if m != 0 else (n + 1))
        Znm /= np.sqrt(np.sum(Znm**2 * mask))  # ortonormalizar sobre máscara
        base.append(Znm)

    return np.array(base)

def fit_zernike(wavefront, mask, R_out, center, max_order=10):
    """
    Ajusta una serie de polinomios de Zernike al frente de onda proporcionado.

    Retorna:
    - wavefront_reconstruido
    - coeficientes_zernike
    - base_zernike
    """

    base = zernike_polynomials(wavefront.shape, mask, R_out, center, max_order)

    masked_wavefront = wavefront[mask]
    masked_base = base[:, mask]

    # Mínimos cuadrados para obtener coeficientes
    coeffs, *_ = np.linalg.lstsq(masked_base.T, masked_wavefront, rcond=None)

    # Reconstrucción del frente de onda desde la base y coeficientes
    wavefront_reconstructed = np.sum(base * coeffs[:, None, None], axis=0) * mask

    return wavefront_reconstructed, coeffs, base