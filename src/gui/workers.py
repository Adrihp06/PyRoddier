# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""
Worker threads for PyRoddier GUI.
Handles background processing to prevent GUI freezing.
"""

from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike
from src.core.optical_preprocessing import preprocess_roddier


class RoddierWorkerThread(QThread):
    """
    Worker thread for running Roddier test calculations in the background.

    Signals:
        progress: int - Progress percentage (0-100)
        status: str - Status message
        finished: dict - Results dictionary with all calculated data
        error: str - Error message if calculation fails
    """

    # Signals
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cropped_intra, cropped_extra, telescope_params, roddier_params):
        """
        Initialize the worker thread.

        Args:
            cropped_intra: Intra-focal image
            cropped_extra: Extra-focal image
            telescope_params: Dictionary with telescope parameters
            roddier_params: Dictionary with Roddier test parameters
        """
        super().__init__()
        self.cropped_intra = cropped_intra
        self.cropped_extra = cropped_extra
        self.telescope_params = telescope_params
        self.roddier_params = roddier_params
        self._is_running = True

    def run(self):
        """Execute the Roddier test calculations in the background."""
        try:
            # Extract parameters
            apertura = self.telescope_params.get('apertura', 900.0)
            focal = self.telescope_params.get('focal', 7200.0)
            pixel_scale = self.telescope_params.get('tamano_pixel', 15.0)
            max_order = self.roddier_params.get('max_order', 23)
            threshold = self.roddier_params.get('threshold', 0.5)
            wavelength_nm = self.roddier_params.get('wavelength_nm', 555)  # Por defecto 555nm (luz verde)

            # Stage 1: Preprocessing (0-30%)
            self.status.emit("Preprocesando imágenes...")
            self.progress.emit(5)

            if not self._is_running:
                return

            delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
                self.cropped_intra,
                self.cropped_extra,
                apertura=apertura,
                focal=focal,
                pixel_scale=pixel_scale,
                threshold=threshold
            )

            self.progress.emit(30)

            # Validate preprocessing results
            if delta_I_norm is None or annular_mask is None:
                self.error.emit("Error en el preprocesamiento de imágenes.")
                return

            if not np.any(annular_mask):
                self.error.emit("No se encontraron píxeles válidos en la máscara anular.")
                return

            # Stage 2: Wavefront calculation (30-50%)
            self.status.emit("Calculando frente de onda...")
            self.progress.emit(35)

            if not self._is_running:
                return

            wavefront = calculate_wavefront(
                delta_I_norm,
                annular_mask,
                wavelength_nm=wavelength_nm,
                dz_mm=dz_mm
            )

            self.progress.emit(50)

            # Validate wavefront
            if not np.any(np.isfinite(wavefront)):
                self.error.emit("Error en el cálculo del frente de onda.")
                return

            # Stage 3: Zernike fitting (50-90%)
            self.status.emit("Ajustando polinomios de Zernike...")
            self.progress.emit(55)

            if not self._is_running:
                return

            zernike_coeffs, zernike_base = fit_zernike(
                wavefront, annular_mask, R_out, center, max_order
            )

            self.progress.emit(90)

            # Validate coefficients
            if zernike_coeffs is None or len(zernike_coeffs) == 0:
                self.error.emit("Error en el ajuste de coeficientes de Zernike.")
                return

            # Stage 4: Finalization (90-100%)
            self.status.emit("Finalizando...")
            self.progress.emit(95)

            if not self._is_running:
                return

            # Calculate average image
            img_avg = 0.5 * (self.cropped_intra + self.cropped_extra)

            # Prepare results
            results = {
                'zernike_coeffs': zernike_coeffs,
                'zernike_base': zernike_base,
                'annular_mask': annular_mask,
                'wavefront': wavefront,
                'img_avg': img_avg,
                'center': center,
                'R_out': R_out,
                'dz_mm': dz_mm,
                'wavelength_nm': wavelength_nm
            }

            self.progress.emit(100)
            self.status.emit("Completado")

            # Emit finished signal with results
            self.finished.emit(results)

        except Exception as e:
            error_msg = f"Error durante el cálculo: {str(e)}"
            self.error.emit(error_msg)
            import traceback
            traceback.print_exc()

    def stop(self):
        """Stop the worker thread gracefully."""
        self._is_running = False
        self.quit()
        self.wait()
