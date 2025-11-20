# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Módulo de exportación de resultados del Test de Roddier.
Soporta exportación a CSV, JSON y generación de reportes PDF.
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import numpy as np


def export_zernike_to_csv(
    coefficients: np.ndarray,
    output_path: str,
    metadata: Optional[Dict] = None,
    include_names: bool = True
) -> bool:
    """
    Exporta coeficientes de Zernike a un archivo CSV.

    Formato compatible con software astronómico estándar (Zemax, Code V, etc.)

    Parámetros:
    - coefficients: array 1D - Coeficientes de Zernike (índice de Noll)
    - output_path: str - Ruta del archivo CSV de salida
    - metadata: dict - Metadatos opcionales (telescopio, fecha, etc.)
    - include_names: bool - Si True, incluye nombres de aberraciones (default: True)

    Retorna:
    - bool - True si la exportación fue exitosa, False en caso contrario

    Formato CSV:
    ```
    # PyRoddier Zernike Export
    # Date: 2025-11-18 23:30:00
    # Telescope: MyTelescope
    Noll_Index,Coefficient_waves,Coefficient_nm,Aberration_Name
    1,0.123,68.2,Piston
    2,0.045,24.9,Tilt X
    ...
    ```
    """
    try:
        # Nombres de aberraciones según índice de Noll
        zernike_names = [
            "Piston",           # j=1
            "Tilt X",           # j=2
            "Tilt Y",           # j=3
            "Defocus",          # j=4
            "Astigmatism 45°",  # j=5
            "Astigmatism 0°",   # j=6
            "Coma Y",           # j=7
            "Coma X",           # j=8
            "Trefoil Y",        # j=9
            "Trefoil X",        # j=10
            "Spherical",        # j=11
            "2nd Astig 0°",     # j=12
            "2nd Astig 45°",    # j=13
            "Tetrafoil 0°",     # j=14
            "Tetrafoil 22.5°",  # j=15
            "2nd Coma X",       # j=16
            "2nd Coma Y",       # j=17
            "2nd Trefoil X",    # j=18
            "2nd Trefoil Y",    # j=19
            "Pentafoil X",      # j=20
            "Pentafoil Y",      # j=21
            "2nd Spherical",    # j=22
            "Hexafoil 0°",      # j=23
        ]

        # Longitud de onda de referencia (nm)
        wavelength_nm = metadata.get('wavelength_nm', 555) if metadata else 555

        with open(output_path, 'w', newline='') as csvfile:
            # Escribir encabezado con metadatos
            csvfile.write(f"# PyRoddier Zernike Export\n")
            csvfile.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            if metadata:
                for key, value in metadata.items():
                    csvfile.write(f"# {key}: {value}\n")

            csvfile.write("#\n")

            # Escribir datos
            writer = csv.writer(csvfile)

            # Encabezado de columnas
            header = ['Noll_Index', 'Coefficient_waves', 'Coefficient_nm']
            if include_names:
                header.append('Aberration_Name')
            writer.writerow(header)

            # Escribir cada coeficiente
            for i, coeff in enumerate(coefficients, start=1):
                row = [
                    i,  # Índice de Noll (empieza en 1)
                    f"{coeff:.6f}",  # Coeficiente en ondas
                    f"{coeff * wavelength_nm:.3f}"  # Coeficiente en nm
                ]

                if include_names and i <= len(zernike_names):
                    row.append(zernike_names[i - 1])
                elif include_names:
                    row.append(f"Z{i}")

                writer.writerow(row)

        return True

    except Exception as e:
        print(f"Error al exportar a CSV: {e}")
        return False


def export_results_to_json(
    zernike_coeffs: np.ndarray,
    rms: float,
    ptv: float,
    output_path: str,
    telescope_params: Optional[Dict] = None,
    roddier_params: Optional[Dict] = None
) -> bool:
    """
    Exporta todos los resultados del Test de Roddier a formato JSON.

    Formato compatible con otros programas de análisis de datos.

    Parámetros:
    - zernike_coeffs: array 1D - Coeficientes de Zernike
    - rms: float - RMS del frente de onda
    - ptv: float - Peak-to-Valley del frente de onda
    - output_path: str - Ruta del archivo JSON de salida
    - telescope_params: dict - Parámetros del telescopio
    - roddier_params: dict - Parámetros del test de Roddier

    Retorna:
    - bool - True si la exportación fue exitosa, False en caso contrario
    """
    try:
        results = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "pyroddier_version": "1.0.0",
                "test_type": "Roddier Wavefront Analysis"
            },
            "telescope": telescope_params or {},
            "roddier_parameters": roddier_params or {},
            "results": {
                "zernike_coefficients": {
                    "values": zernike_coeffs.tolist(),
                    "unit": "waves",
                    "index_convention": "Noll"
                },
                "wavefront_statistics": {
                    "rms": float(rms),
                    "ptv": float(ptv),
                    "units": "waves"
                }
            }
        }

        with open(output_path, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2)

        return True

    except Exception as e:
        print(f"Error al exportar a JSON: {e}")
        return False


def generate_summary_report(
    zernike_coeffs: np.ndarray,
    rms: float,
    ptv: float,
    output_path: str,
    telescope_params: Optional[Dict] = None,
    top_n_aberrations: int = 5
) -> bool:
    """
    Genera un reporte de texto resumido con las principales aberraciones.

    Útil para análisis rápido sin necesidad de software adicional.

    Parámetros:
    - zernike_coeffs: array 1D - Coeficientes de Zernike
    - rms: float - RMS del frente de onda
    - ptv: float - Peak-to-Valley del frente de onda
    - output_path: str - Ruta del archivo de texto de salida
    - telescope_params: dict - Parámetros del telescopio
    - top_n_aberrations: int - Número de aberraciones principales a destacar

    Retorna:
    - bool - True si la exportación fue exitosa, False en caso contrario
    """
    try:
        zernike_names = [
            "Piston", "Tilt X", "Tilt Y", "Defocus", "Astigmatism 45°",
            "Astigmatism 0°", "Coma Y", "Coma X", "Trefoil Y", "Trefoil X",
            "Spherical", "2nd Astig 0°", "2nd Astig 45°", "Tetrafoil 0°",
            "Tetrafoil 22.5°", "2nd Coma X", "2nd Coma Y", "2nd Trefoil X",
            "2nd Trefoil Y", "Pentafoil X", "Pentafoil Y", "2nd Spherical",
            "Hexafoil 0°"
        ]

        # Encontrar las aberraciones más significativas (excluyendo piston)
        coeffs_abs = np.abs(zernike_coeffs[1:])  # Excluir piston
        top_indices = np.argsort(coeffs_abs)[-top_n_aberrations:][::-1]

        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("PyRoddier - Reporte de Análisis de Frente de Onda\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Parámetros del telescopio
            if telescope_params:
                f.write("Parámetros del Telescopio:\n")
                f.write("-" * 70 + "\n")
                for key, value in telescope_params.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # Estadísticas del frente de onda
            f.write("Estadísticas del Frente de Onda:\n")
            f.write("-" * 70 + "\n")
            f.write(f"  RMS:        {rms:.4f} λ\n")
            f.write(f"  Peak-Valley: {ptv:.4f} λ\n\n")

            # Top aberraciones
            f.write(f"Top {top_n_aberrations} Aberraciones Dominantes:\n")
            f.write("-" * 70 + "\n")
            for rank, idx in enumerate(top_indices, start=1):
                j = idx + 2  # +2 porque excluimos piston y los índices empiezan en 1
                coeff = zernike_coeffs[idx + 1]
                name = zernike_names[idx + 1] if idx + 1 < len(zernike_names) else f"Z{j}"
                f.write(f"  {rank}. {name:20s} (Z{j:2d}): {coeff:+.4f} λ\n")

            f.write("\n" + "=" * 70 + "\n")

        return True

    except Exception as e:
        print(f"Error al generar reporte: {e}")
        return False


def export_all_formats(
    zernike_coeffs: np.ndarray,
    rms: float,
    ptv: float,
    output_dir: str,
    base_filename: str = "roddier_results",
    telescope_params: Optional[Dict] = None,
    roddier_params: Optional[Dict] = None
) -> Dict[str, bool]:
    """
    Exporta resultados en todos los formatos disponibles (CSV, JSON, TXT).

    Función de conveniencia para exportar a múltiples formatos simultáneamente.

    Parámetros:
    - zernike_coeffs: array 1D - Coeficientes de Zernike
    - rms: float - RMS del frente de onda
    - ptv: float - Peak-to-Valley del frente de onda
    - output_dir: str - Directorio de salida
    - base_filename: str - Nombre base para los archivos (default: "roddier_results")
    - telescope_params: dict - Parámetros del telescopio
    - roddier_params: dict - Parámetros del test de Roddier

    Retorna:
    - dict - Estado de cada exportación {'csv': bool, 'json': bool, 'txt': bool}
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # Exportar CSV
    csv_path = output_path / f"{base_filename}.csv"
    results['csv'] = export_zernike_to_csv(
        zernike_coeffs,
        str(csv_path),
        metadata=telescope_params
    )

    # Exportar JSON
    json_path = output_path / f"{base_filename}.json"
    results['json'] = export_results_to_json(
        zernike_coeffs, rms, ptv,
        str(json_path),
        telescope_params=telescope_params,
        roddier_params=roddier_params
    )

    # Generar reporte de texto
    txt_path = output_path / f"{base_filename}_summary.txt"
    results['txt'] = generate_summary_report(
        zernike_coeffs, rms, ptv,
        str(txt_path),
        telescope_params=telescope_params
    )

    return results
