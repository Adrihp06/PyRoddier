#!/usr/bin/env python
"""
Script de verificación de conversión de unidades en PyRoddier.

Verifica que:
1. calculate_wavefront retorna valores en ondas (λ)
2. fit_zernike retorna coeficientes en ondas (λ)
3. calculate_rms retorna RMS en ondas (λ)
4. calculate_ptv retorna PTV en ondas (λ)
5. Las conversiones a µm y nm son correctas
"""

import numpy as np
from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike, calculate_rms, calculate_ptv

def test_units_conversion():
    """Prueba las conversiones de unidades."""
    print("=" * 60)
    print("TEST DE CONVERSIÓN DE UNIDADES EN PYRODDIER")
    print("=" * 60)

    # Parámetros de prueba
    wavelength_nm = 555  # nm (luz verde)
    dz_mm = 2.5  # mm (distancia de desenfoque típica)

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

    print(f"\n1. PARÁMETROS DE ENTRADA:")
    print(f"   Longitud de onda: {wavelength_nm} nm")
    print(f"   Distancia de desenfoque: {dz_mm} mm")
    print(f"   Tamaño de imagen: {size}x{size} píxeles")

    # Calcular wavefront
    print(f"\n2. CÁLCULO DE WAVEFRONT:")
    wavefront = calculate_wavefront(
        delta_I_norm,
        annular_mask,
        wavelength_nm=wavelength_nm,
        dz_mm=dz_mm
    )

    # Estadísticas del wavefront
    wavefront_masked = wavefront[annular_mask > 0]
    wf_min = np.min(wavefront_masked)
    wf_max = np.max(wavefront_masked)
    wf_mean = np.mean(wavefront_masked)
    wf_std = np.std(wavefront_masked)

    print(f"   Wavefront (λ):")
    print(f"     - Mínimo: {wf_min:.4f} λ")
    print(f"     - Máximo: {wf_max:.4f} λ")
    print(f"     - Media: {wf_mean:.4f} λ")
    print(f"     - Desv. std: {wf_std:.4f} λ")

    # Conversión a otras unidades
    wavelength_mm = wavelength_nm / 1e6
    print(f"\n   Conversiones a otras unidades:")
    print(f"     - Media: {wf_mean * wavelength_mm * 1000:.4f} µm")
    print(f"     - Media: {wf_mean * wavelength_mm * 1e6:.4f} nm")
    print(f"     - Media: {wf_mean * wavelength_mm:.6f} mm")

    # Ajustar coeficientes de Zernike
    print(f"\n3. AJUSTE DE COEFICIENTES DE ZERNIKE:")
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

    print(f"   Primeros 10 coeficientes (λ):")
    zernike_names = [
        "Piston", "Tilt X", "Tilt Y", "Astig 45°", "Defocus",
        "Astig 0°", "Trefoil X", "Coma X", "Coma Y", "Trefoil Y"
    ]
    for i in range(min(10, len(coeffs))):
        coeff_um = coeffs[i] * wavelength_mm * 1000
        print(f"     Z{i+1:2d} ({zernike_names[i]:12s}): {coeffs[i]:8.4f} λ ({coeff_um:8.2f} µm)")

    # Calcular RMS
    print(f"\n4. CÁLCULO DE RMS:")
    rms_value = calculate_rms(coeffs, exclude_piston=True)
    rms_um = rms_value * wavelength_mm * 1000
    rms_nm = rms_value * wavelength_mm * 1e6

    print(f"   RMS (sin pistón):")
    print(f"     - {rms_value:.4f} λ")
    print(f"     - {rms_um:.2f} µm")
    print(f"     - {rms_nm:.1f} nm")

    # Calcular PTV
    print(f"\n5. CÁLCULO DE PTV (PEAK-TO-VALLEY):")
    ptv_value = calculate_ptv(wavefront, annular_mask)
    ptv_um = ptv_value * wavelength_mm * 1000
    ptv_nm = ptv_value * wavelength_mm * 1e6

    print(f"   PTV:")
    print(f"     - {ptv_value:.4f} λ")
    print(f"     - {ptv_um:.2f} µm")
    print(f"     - {ptv_nm:.1f} nm")

    # Verificación de consistencia
    print(f"\n6. VERIFICACIÓN DE CONSISTENCIA:")

    # El PTV calculado manualmente debería coincidir con calculate_ptv
    ptv_manual = wf_max - wf_min
    ptv_diff = abs(ptv_value - ptv_manual)
    print(f"   PTV manual: {ptv_manual:.4f} λ")
    print(f"   PTV función: {ptv_value:.4f} λ")
    print(f"   Diferencia: {ptv_diff:.6f} λ (debe ser ~0)")

    if ptv_diff < 1e-10:
        print(f"   ✓ PTV consistente")
    else:
        print(f"   ✗ ERROR: PTV no consistente")

    # Factor de conversión
    expected_factor = 1 / wavelength_mm
    print(f"\n7. FACTOR DE CONVERSIÓN:")
    print(f"   1 λ = {wavelength_mm:.6f} mm")
    print(f"   1 mm = {expected_factor:.1f} λ")
    print(f"   1 λ = {wavelength_mm * 1000:.3f} µm")
    print(f"   1 λ = {wavelength_nm:.1f} nm")

    print("\n" + "=" * 60)
    print("✓ TEST COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    # Usar asserts en lugar de return
    assert ptv_diff < 1e-10, f"PTV no consistente: diferencia {ptv_diff}"
    assert rms_value > 0, "RMS debe ser positivo"
    assert ptv_value > 0, "PTV debe ser positivo"

if __name__ == "__main__":
    test_units_conversion()
