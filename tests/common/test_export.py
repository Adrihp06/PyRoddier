#!/usr/bin/env python
"""Tests para el módulo de exportación de resultados."""

import unittest
import tempfile
import shutil
from pathlib import Path
import csv
import json
import numpy as np
from src.common.export import (
    export_zernike_to_csv,
    export_results_to_json,
    generate_summary_report,
    export_all_formats
)


class TestExport(unittest.TestCase):
    """Test suite para funciones de exportación."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear directorio temporal para tests
        self.test_dir = tempfile.mkdtemp()

        # Datos de prueba
        self.zernike_coeffs = np.array([
            0.123,  # Piston
            0.045,  # Tilt X
            -0.032, # Tilt Y
            0.089,  # Defocus
            -0.012, # Astigmatism 45°
            0.025,  # Astigmatism 0°
            -0.018, # Coma Y
            0.031,  # Coma X
            0.008,  # Trefoil Y
            -0.015, # Trefoil X
        ])

        self.rms = 0.0567
        self.ptv = 0.234

        self.telescope_params = {
            'name': 'Test Telescope',
            'diameter_mm': 200,
            'focal_length_mm': 1000,
            'wavelength_nm': 555
        }

        self.roddier_params = {
            'defocus_distance_mm': 2.5,
            'pixel_size_um': 3.75
        }

    def tearDown(self):
        """Limpieza después de cada test."""
        # Eliminar directorio temporal
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_export_zernike_to_csv_basic(self):
        """Test básico de exportación a CSV."""
        output_path = Path(self.test_dir) / "test_zernike.csv"

        result = export_zernike_to_csv(
            self.zernike_coeffs,
            str(output_path)
        )

        # Verificar que la exportación fue exitosa
        self.assertTrue(result)
        self.assertTrue(output_path.exists())

        # Verificar contenido del archivo
        with open(output_path, 'r') as f:
            content = f.read()

            # Verificar encabezados
            self.assertIn("PyRoddier Zernike Export", content)
            self.assertIn("Date:", content)
            self.assertIn("Noll_Index", content)
            self.assertIn("Coefficient_waves", content)

            # Verificar que hay datos
            self.assertIn("0.123", content)  # Piston

    def test_export_zernike_to_csv_with_metadata(self):
        """Test de exportación a CSV con metadatos."""
        output_path = Path(self.test_dir) / "test_zernike_meta.csv"

        result = export_zernike_to_csv(
            self.zernike_coeffs,
            str(output_path),
            metadata=self.telescope_params
        )

        self.assertTrue(result)

        # Verificar que los metadatos están en el archivo
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Telescope", content)
            self.assertIn("200", content)  # diameter_mm

    def test_export_zernike_to_csv_without_names(self):
        """Test de exportación a CSV sin nombres de aberraciones."""
        output_path = Path(self.test_dir) / "test_zernike_no_names.csv"

        result = export_zernike_to_csv(
            self.zernike_coeffs,
            str(output_path),
            include_names=False
        )

        self.assertTrue(result)

        # Verificar estructura del CSV
        with open(output_path, 'r') as f:
            lines = f.readlines()
            # Encontrar línea de encabezado
            header_line = [l for l in lines if 'Noll_Index' in l][0]
            self.assertNotIn("Aberration_Name", header_line)

    def test_export_zernike_to_csv_invalid_path(self):
        """Test de exportación a CSV con ruta inválida."""
        invalid_path = "/invalid/path/that/does/not/exist/test.csv"

        result = export_zernike_to_csv(
            self.zernike_coeffs,
            invalid_path
        )

        # Debe fallar pero no crashear
        self.assertFalse(result)

    def test_export_results_to_json_basic(self):
        """Test básico de exportación a JSON."""
        output_path = Path(self.test_dir) / "test_results.json"

        result = export_results_to_json(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_path)
        )

        self.assertTrue(result)
        self.assertTrue(output_path.exists())

        # Verificar contenido JSON
        with open(output_path, 'r') as f:
            data = json.load(f)

            # Verificar estructura
            self.assertIn('metadata', data)
            self.assertIn('results', data)

            # Verificar resultados
            self.assertEqual(data['results']['wavefront_statistics']['rms'], self.rms)
            self.assertEqual(data['results']['wavefront_statistics']['ptv'], self.ptv)

            # Verificar coeficientes de Zernike
            exported_coeffs = np.array(data['results']['zernike_coefficients']['values'])
            np.testing.assert_array_almost_equal(exported_coeffs, self.zernike_coeffs)

    def test_export_results_to_json_with_params(self):
        """Test de exportación a JSON con parámetros."""
        output_path = Path(self.test_dir) / "test_results_params.json"

        result = export_results_to_json(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_path),
            telescope_params=self.telescope_params,
            roddier_params=self.roddier_params
        )

        self.assertTrue(result)

        # Verificar parámetros en JSON
        with open(output_path, 'r') as f:
            data = json.load(f)

            self.assertIn('telescope', data)
            self.assertEqual(data['telescope']['name'], 'Test Telescope')

            self.assertIn('roddier_parameters', data)
            self.assertEqual(data['roddier_parameters']['defocus_distance_mm'], 2.5)

    def test_export_results_to_json_invalid_path(self):
        """Test de exportación a JSON con ruta inválida."""
        invalid_path = "/invalid/path/that/does/not/exist/test.json"

        result = export_results_to_json(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            invalid_path
        )

        self.assertFalse(result)

    def test_generate_summary_report_basic(self):
        """Test básico de generación de reporte."""
        output_path = Path(self.test_dir) / "test_summary.txt"

        result = generate_summary_report(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_path)
        )

        self.assertTrue(result)
        self.assertTrue(output_path.exists())

        # Verificar contenido del reporte
        with open(output_path, 'r') as f:
            content = f.read()

            # Verificar encabezados
            self.assertIn("PyRoddier", content)
            self.assertIn("Reporte de Análisis", content)

            # Verificar estadísticas
            self.assertIn("RMS", content)
            self.assertIn("Peak-Valley", content)
            self.assertIn(f"{self.rms:.4f}", content)
            self.assertIn(f"{self.ptv:.4f}", content)

            # Verificar que hay aberraciones listadas
            self.assertIn("Aberraciones Dominantes", content)

    def test_generate_summary_report_with_telescope_params(self):
        """Test de generación de reporte con parámetros del telescopio."""
        output_path = Path(self.test_dir) / "test_summary_params.txt"

        result = generate_summary_report(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_path),
            telescope_params=self.telescope_params,
            top_n_aberrations=3
        )

        self.assertTrue(result)

        # Verificar parámetros del telescopio en el reporte
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("Test Telescope", content)
            self.assertIn("Parámetros del Telescopio", content)

    def test_generate_summary_report_invalid_path(self):
        """Test de generación de reporte con ruta inválida."""
        invalid_path = "/invalid/path/that/does/not/exist/test.txt"

        result = generate_summary_report(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            invalid_path
        )

        self.assertFalse(result)

    def test_export_all_formats_basic(self):
        """Test de exportación a todos los formatos."""
        output_dir = Path(self.test_dir) / "all_formats"

        results = export_all_formats(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_dir)
        )

        # Verificar que todos los formatos se exportaron exitosamente
        self.assertTrue(results['csv'])
        self.assertTrue(results['json'])
        self.assertTrue(results['txt'])

        # Verificar que todos los archivos existen
        self.assertTrue((output_dir / "roddier_results.csv").exists())
        self.assertTrue((output_dir / "roddier_results.json").exists())
        self.assertTrue((output_dir / "roddier_results_summary.txt").exists())

    def test_export_all_formats_custom_filename(self):
        """Test de exportación a todos los formatos con nombre personalizado."""
        output_dir = Path(self.test_dir) / "custom_name"

        results = export_all_formats(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_dir),
            base_filename="my_results",
            telescope_params=self.telescope_params,
            roddier_params=self.roddier_params
        )

        self.assertTrue(results['csv'])
        self.assertTrue(results['json'])
        self.assertTrue(results['txt'])

        # Verificar nombres personalizados
        self.assertTrue((output_dir / "my_results.csv").exists())
        self.assertTrue((output_dir / "my_results.json").exists())
        self.assertTrue((output_dir / "my_results_summary.txt").exists())

    def test_export_all_formats_creates_directory(self):
        """Test que export_all_formats crea el directorio si no existe."""
        output_dir = Path(self.test_dir) / "new_dir" / "nested" / "dir"

        results = export_all_formats(
            self.zernike_coeffs,
            self.rms,
            self.ptv,
            str(output_dir)
        )

        # Verificar que el directorio fue creado
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())

        # Verificar exportación exitosa
        self.assertTrue(results['csv'])
        self.assertTrue(results['json'])
        self.assertTrue(results['txt'])


if __name__ == '__main__':
    unittest.main()
