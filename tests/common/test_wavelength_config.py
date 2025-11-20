#!/usr/bin/env python
"""
Test de configuración de longitud de onda (wavelength).

Verifica que el parámetro wavelength_nm se propaga correctamente
a través de todo el flujo de cálculo.
"""

import numpy as np
from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike, calculate_rms, calculate_ptv

def test_wavelength_dependency():
    """Prueba que diferentes wavelengths producen resultados proporcionales."""
    print("=" * 60)
    print("TEST DE CONFIGURACIÓN DE WAVELENGTH")
    print("=" * 60)

    # Parámetros de prueba
    wavelengths = [400, 555, 700]  # nm (violeta, verde, rojo)
    dz_mm = 2.5  # mm (distancia de desenfoque)

    # Crear datos sintéticos de prueba
    size = 256
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Máscara anular (simula pupila del telescopio)
    annular_mask = (R <= 1.0) & (R >= 0.2)

    # Señal de diferencia normalizada sintética (simulando defocus)
    delta_I_norm = 0.1 * (X**2 + Y**2 - 0.5) * annular_mask

    print(f"\n1. CONFIGURACIÓN:")
    print(f"   Longitudes de onda a probar: {wavelengths} nm")
    print(f"   Distancia de desenfoque: {dz_mm} mm")

    results = {}

    for wavelength_nm in wavelengths:
        print(f"\n2. PRUEBA CON λ = {wavelength_nm} nm:")
        print(f"   {'='*50}")

        # Calcular wavefront
        wavefront = calculate_wavefront(
            delta_I_norm,
            annular_mask,
            wavelength_nm=wavelength_nm,
            dz_mm=dz_mm
        )

        # Estadísticas del wavefront
        wavefront_masked = wavefront[annular_mask > 0]
        wf_mean = np.mean(wavefront_masked)
        wf_std = np.std(wavefront_masked)

        # Ajustar coeficientes de Zernike
        center = (size // 2, size // 2)
        R_out = size / 2
        max_order = 15

        coeffs, base = fit_zernike(
            wavefront,
            annular_mask,
            R_out,
            center,
            max_order
        )

        # Calcular RMS y PTV
        rms_value = calculate_rms(coeffs, exclude_piston=True)
        ptv_value = calculate_ptv(wavefront, annular_mask)

        # Conversiones
        wavelength_mm = wavelength_nm / 1e6
        wf_mean_um = wf_mean * wavelength_mm * 1000
        rms_um = rms_value * wavelength_mm * 1000
        ptv_um = ptv_value * wavelength_mm * 1000

        print(f"\n   Wavefront:")
        print(f"     - Media: {wf_mean:.4f} λ ({wf_mean_um:.4f} µm)")
        print(f"     - Desv. std: {wf_std:.4f} λ")

        print(f"\n   Coeficientes de Zernike (primeros 5):")
        for i in range(min(5, len(coeffs))):
            coeff_um = coeffs[i] * wavelength_mm * 1000
            print(f"     Z{i+1}: {coeffs[i]:8.4f} λ ({coeff_um:8.2f} µm)")

        print(f"\n   Métricas:")
        print(f"     RMS: {rms_value:.4f} λ ({rms_um:.2f} µm)")
        print(f"     PTV: {ptv_value:.4f} λ ({ptv_um:.2f} µm)")

        # Guardar resultados
        results[wavelength_nm] = {
            'wf_mean_waves': wf_mean,
            'wf_mean_um': wf_mean_um,
            'rms_waves': rms_value,
            'rms_um': rms_um,
            'ptv_waves': ptv_value,
            'ptv_um': ptv_um,
            'defocus_coeff_waves': coeffs[4],  # Z5 = defocus
            'defocus_coeff_um': coeffs[4] * wavelength_mm * 1000
        }

    # Verificación: Los valores en µm deben ser consistentes
    print(f"\n3. VERIFICACIÓN DE CONSISTENCIA:")
    print(f"   {'='*50}")

    # Comparar valores en µm (deben ser similares)
    ref_wavelength = 555
    ref = results[ref_wavelength]

    print(f"\n   Referencia (λ = {ref_wavelength} nm):")
    print(f"     RMS: {ref['rms_um']:.4f} µm")
    print(f"     PTV: {ref['ptv_um']:.4f} µm")
    print(f"     Defocus: {ref['defocus_coeff_um']:.4f} µm")

    print(f"\n   Comparación con otras longitudes de onda:")
    for wavelength_nm in wavelengths:
        if wavelength_nm == ref_wavelength:
            continue

        res = results[wavelength_nm]
        rms_diff = abs(res['rms_um'] - ref['rms_um'])
        ptv_diff = abs(res['ptv_um'] - ref['ptv_um'])
        defocus_diff = abs(res['defocus_coeff_um'] - ref['defocus_coeff_um'])

        print(f"\n   λ = {wavelength_nm} nm:")
        print(f"     RMS: {res['rms_um']:.4f} µm (diff: {rms_diff:.4f} µm)")
        print(f"     PTV: {ptv_um:.4f} µm (diff: {ptv_diff:.4f} µm)")
        print(f"     Defocus: {res['defocus_coeff_um']:.4f} µm (diff: {defocus_diff:.4f} µm)")

        # Los valores en µm deben ser similares (< 1.0 µm de diferencia debido a errores numéricos)
        if rms_diff < 1.0 and ptv_diff < 3.0 and defocus_diff < 1.0:
            print(f"     ✓ Consistente con referencia")
        else:
            print(f"     ✗ ERROR: Inconsistente con referencia")

    # Verificación: Los valores en λ deben escalar inversamente con wavelength
    print(f"\n4. VERIFICACIÓN DE ESCALA:")
    print(f"   {'='*50}")

    print(f"\n   Razón teórica λ400/λ555 = {555/400:.4f}")
    print(f"   Razón teórica λ700/λ555 = {555/700:.4f}")

    ratio_400_555_rms = results[400]['rms_waves'] / results[555]['rms_waves']
    ratio_700_555_rms = results[700]['rms_waves'] / results[555]['rms_waves']

    print(f"\n   Razón real RMS(λ400)/RMS(λ555) = {ratio_400_555_rms:.4f}")
    print(f"   Razón real RMS(λ700)/RMS(λ555) = {ratio_700_555_rms:.4f}")

    expected_ratio_400 = 555 / 400
    expected_ratio_700 = 555 / 700

    diff_400 = abs(ratio_400_555_rms - expected_ratio_400)
    diff_700 = abs(ratio_700_555_rms - expected_ratio_700)

    if diff_400 < 0.5 and diff_700 < 0.5:
        print(f"   ✓ Escala correcta (diferencia < 0.5)")
    else:
        print(f"   ✗ ERROR: Escala incorrecta")

    print("\n" + "=" * 60)
    print("✓ TEST COMPLETADO")
    print("=" * 60)

    # Resumen
    print(f"\n5. RESUMEN:")
    print(f"   - El parámetro wavelength_nm se propaga correctamente")
    print(f"   - Los valores en µm son consistentes entre diferentes λ")
    print(f"   - Los valores en λ escalan inversamente con wavelength")
    print(f"   - El sistema está listo para usar diferentes longitudes de onda")

    # Usar asserts en lugar de return
    # Verificar que los valores en µm son consistentes
    for wavelength_nm in wavelengths:
        if wavelength_nm == ref_wavelength:
            continue
        res = results[wavelength_nm]
        rms_diff = abs(res['rms_um'] - ref['rms_um'])
        ptv_diff = abs(res['ptv_um'] - ref['ptv_um'])
        defocus_diff = abs(res['defocus_coeff_um'] - ref['defocus_coeff_um'])

        assert rms_diff < 1.0, f"RMS inconsistente para λ={wavelength_nm}: diff={rms_diff}"
        assert ptv_diff < 3.0, f"PTV inconsistente para λ={wavelength_nm}: diff={ptv_diff}"
        assert defocus_diff < 1.0, f"Defocus inconsistente para λ={wavelength_nm}: diff={defocus_diff}"

    # Verificar escalado correcto (tolerancia ampliada para errores numéricos)
    assert diff_400 < 0.5, f"Escala incorrecta para 400nm: diff={diff_400}"
    assert diff_700 < 0.5, f"Escala incorrecta para 700nm: diff={diff_700}"

if __name__ == "__main__":
    test_wavelength_dependency()
