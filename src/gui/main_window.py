# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QMessageBox, QDialog, QFrame, QScrollArea, QToolBar, QAction)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon
from matplotlib.colors import Normalize
import numpy as np
import os
import json
from pathlib import Path
from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike
from src.common.utils import load_fits_image, calculate_center_of_mass, find_center
from src.core.optical_preprocessing import preprocess_roddier
from src.gui.dialogs.roddiertestresults import RoddierTestResultsWindow
from src.gui.dialogs.roddiertest import RoddierTestDialog
from src.gui.dialogs.config_dialog import ConfigDialog
from src.common.config import get_config_paths
import sys

def get_resource_path(relative_path):
    """Get the absolute path to a resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    return os.path.join(base_path, relative_path)

class FitsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test de Roddier con Zernike")
        self.setGeometry(100, 100, 1200, 800)

        # Get configuration paths
        config_paths = get_config_paths()
        self.config_path = config_paths['telescope_dir']
        self.config_file = Path.home() / '.pyroddier' / 'config.json'

        # Variables de configuración
        self.image_path = None
        self.results_path = None

        # Cargar rutas por defecto
        self.load_default_paths()

        # Crear barra de menú
        self.menubar = self.menuBar()
        self.apply_theme()

        # Crear barra de herramientas
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)
        self.toolbar.setAllowedAreas(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, self.toolbar)  # Asegurar que la barra de herramientas está en la parte superior
        self.toolbar.setStyleSheet("""
            QToolBar {
                background-color: #2b2b2b;
                border: none;
                padding: 4px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: #404040;
            }
            QToolButton:pressed {
                background-color: #505050;
            }
        """)

        # Acción para el Test de Roddier
        self.roddier_action = QAction(QIcon(get_resource_path('icons/roddier.png')), 'Test de Roddier', self)
        self.roddier_action.setStatusTip('Ejecutar Test de Roddier')
        self.roddier_action.triggered.connect(self.run_roddier_test)
        self.toolbar.addAction(self.roddier_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para resetear con el nuevo icono de papelera
        self.reset_action = QAction(QIcon(get_resource_path('icons/trash.png')), 'Borrar', self)
        self.reset_action.setStatusTip('Limpiar imágenes y resetear estado')
        self.reset_action.triggered.connect(self.reset_state)
        self.toolbar.addAction(self.reset_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para centrar imágenes
        self.center_action = QAction(QIcon(get_resource_path('icons/center.png')), 'Centrar', self)
        self.center_action.setStatusTip('Centrar ambas imágenes')
        self.center_action.triggered.connect(self.center_both_images)
        self.toolbar.addAction(self.center_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para configuración
        self.config_action = QAction(QIcon(get_resource_path('icons/settings.png')), 'Configuración', self)
        self.config_action.setStatusTip('Abrir configuración')
        self.config_action.triggered.connect(self.run_config_dialog)
        self.toolbar.addAction(self.config_action)

        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 4px;
            }
            QListWidget {
                background-color: #363636;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
                color: white;
            }
            QListWidget::item {
                padding: 4px;
                border-radius: 2px;
            }
            QListWidget::item:selected {
                background-color: #0d47a1;
            }
            QSlider {
                height: 20px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #404040;
                height: 6px;
                background: #363636;
                margin: 0px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0d47a1;
                border: none;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
            }
            QFrame {
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)

        # Widget central
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Layout para botones de carga
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # Botones de carga
        self.load_intra_button = QPushButton("Cargar imagen intra-focal")
        self.load_intra_button.clicked.connect(self.load_intra_image)
        self.button_layout.addWidget(self.load_intra_button)

        self.load_extra_button = QPushButton("Cargar imagen extra-focal")
        self.load_extra_button.clicked.connect(self.load_extra_image)
        self.button_layout.addWidget(self.load_extra_button)

        # Layout para imágenes
        self.images_layout = QHBoxLayout()
        self.layout.addLayout(self.images_layout)

        # Contenedor para imagen intra-focal con zoom
        self.intra_container = QFrame()
        self.intra_container.setFrameStyle(QFrame.StyledPanel)
        self.intra_layout = QVBoxLayout(self.intra_container)
        self.intra_scroll = QScrollArea()
        self.intra_scroll.setWidgetResizable(True)
        self.intra_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.intra_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.intra_label = QLabel()
        self.intra_label.setAlignment(Qt.AlignCenter)
        self.intra_scroll.setWidget(self.intra_label)
        self.intra_layout.addWidget(self.intra_scroll)
        self.images_layout.addWidget(self.intra_container)

        # Contenedor para imagen extra-focal con zoom
        self.extra_container = QFrame()
        self.extra_container.setFrameStyle(QFrame.StyledPanel)
        self.extra_layout = QVBoxLayout(self.extra_container)
        self.extra_scroll = QScrollArea()
        self.extra_scroll.setWidgetResizable(True)
        self.extra_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.extra_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.extra_label = QLabel()
        self.extra_label.setAlignment(Qt.AlignCenter)
        self.extra_scroll.setWidget(self.extra_label)
        self.extra_layout.addWidget(self.extra_scroll)
        self.images_layout.addWidget(self.extra_container)

        # Variables para el zoom
        self.zoom_factor = 1.0
        self.intra_label.setScaledContents(True)
        self.extra_label.setScaledContents(True)

        # Conectar eventos de rueda del ratón para zoom
        self.intra_scroll.wheelEvent = lambda event: self.handle_wheel_event(event, self.intra_label)
        self.extra_scroll.wheelEvent = lambda event: self.handle_wheel_event(event, self.extra_label)

        # Almacenar las rutas y los datos de las imágenes
        self.intra_image_path = None
        self.extra_image_path = None
        self.intra_image_data = None
        self.extra_image_data = None
        self.intra_pixmap = None
        self.extra_pixmap = None

    def handle_wheel_event(self, event, label):
        """Maneja el evento de la rueda del ratón para hacer zoom."""
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor *= 0.9

            self.zoom_factor = max(0.1, min(self.zoom_factor, 5.0))  # Limitar zoom entre 0.1x y 5x

            if label == self.intra_label and self.intra_pixmap:
                self.update_zoom(self.intra_label, self.intra_pixmap)
            elif label == self.extra_label and self.extra_pixmap:
                self.update_zoom(self.extra_label, self.extra_pixmap)

            event.accept()
        else:
            event.ignore()

    def update_zoom(self, label, original_pixmap):
        """Actualiza el zoom de una imagen."""
        new_width = int(original_pixmap.width() * self.zoom_factor)
        new_height = int(original_pixmap.height() * self.zoom_factor)
        scaled_pixmap = original_pixmap.scaled(new_width, new_height,
                                             Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation)
        label.setPixmap(scaled_pixmap)

        # Mantener el centro después del zoom
        if label == self.intra_label:
            center_y, center_x = np.array(self.intra_image_data.shape) // 2
            self.center_scroll_on_point(self.intra_scroll, center_x, center_y)
        elif label == self.extra_label:
            center_y, center_x = np.array(self.extra_image_data.shape) // 2
            self.center_scroll_on_point(self.extra_scroll, center_x, center_y)

    def process_and_display_image(self, file_path, is_intrafocal=True):
        """Método común para procesar y mostrar imágenes intra y extra-focales.

        Args:
            file_path: Ruta del archivo FITS
            is_intrafocal: True si es imagen intra-focal, False si es extra-focal
        """
        if not file_path:
            return

        # Cargar y almacenar la imagen
        image_data = load_fits_image(file_path)
        if image_data is None:
            return

        # Aplicar transformaciones necesarias para imagen extra-focal
        if not is_intrafocal:
            image_data = np.rot90(image_data, k=2)
        # Calcular el centro de masa
        com_y, com_x = calculate_center_of_mass(image_data)

        # Almacenar datos según el tipo de imagen
        if is_intrafocal:
            self.intra_image_path = file_path
            self.intra_image_data = image_data
            label = self.intra_label
            scroll = self.intra_scroll
        else:
            self.extra_image_path = file_path
            self.extra_image_data = image_data
            label = self.extra_label
            scroll = self.extra_scroll

        # Mostrar la imagen y centrarla
        self.display_image(image_data, label)
        label.adjustSize()
        self.center_scroll_on_point(scroll, com_x, com_y)

    def load_intra_image(self):
        """Carga una imagen intra-focal."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen intra-focal",
            self.image_path if self.image_path else "",
            "FITS Files (*.fits *.fit)"
        )
        if file_path:
            self.process_and_display_image(file_path, is_intrafocal=True)

    def load_extra_image(self):
        """Carga una imagen extra-focal."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar imagen extra-focal",
            self.image_path if self.image_path else "",
            "FITS Files (*.fits *.fit)"
        )
        if file_path:
            self.process_and_display_image(file_path, is_intrafocal=False)

    def display_image(self, image_data, label):
        if image_data is not None:
            norm = Normalize(vmin=np.min(image_data), vmax=np.max(image_data))
            normalized_image = norm(image_data) * 255
            normalized_image = normalized_image.astype(np.uint8)

            height, width = normalized_image.shape
            q_image = QImage(normalized_image.data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)

            # Store the original pixmap
            if label == self.intra_label:
                self.intra_pixmap = pixmap
            else:
                self.extra_pixmap = pixmap

            # Apply current zoom
            self.update_zoom(label, pixmap)

    def run_roddier_test(self):
        """Ejecuta el test de Roddier para analizar el frente de onda."""
        if not self.intra_image_path or not self.extra_image_path:
            QMessageBox.warning(self, "Error", "Por favor, carga las imágenes intra y extra-focal primero.")
            return

        roddier_dialog = RoddierTestDialog(self.intra_image_data, self.extra_image_data, crop_size=250)
        if roddier_dialog.exec_() == QDialog.Accepted:
            cropped_intra, cropped_extra = roddier_dialog.get_cropped_images()
            telescope_params = roddier_dialog.get_telescope_params()
            roddier_params = roddier_dialog.get_roddier_params()
            interferogram_params = roddier_dialog.get_interferogram_params()

            apertura = telescope_params['apertura']
            focal = telescope_params['focal']
            pixel_scale = telescope_params['tamano_pixel']
            max_order = roddier_params['max_order']
            threshold = roddier_params['threshold']

            delta_I_norm, annular_mask, center, R_out, dz_mm = preprocess_roddier(
                cropped_intra,
                cropped_extra,
                apertura=apertura,
                focal=focal,
                pixel_scale=pixel_scale,
                threshold=threshold
            )

            wavefront = calculate_wavefront(delta_I_norm, annular_mask, dz_mm=dz_mm)

            zernike_coeffs, zernike_base = fit_zernike(
                    wavefront, annular_mask, R_out, center, max_order
                )

            # Mostrar resultados en una única ventana
            results_window = RoddierTestResultsWindow("Resultados del Test de Roddier", self)
            results_window.update_plots(
                zernike_coeffs=zernike_coeffs,
                zernike_base=zernike_base,
                annular_mask=annular_mask,
                interferogram_params=interferogram_params,
                telescope_params=telescope_params
            )
            results_window.show()

    def reset_state(self):
        """Resetea el estado de la aplicación a su estado inicial."""
        # Limpiar datos de imágenes
        self.intra_image_path = None
        self.extra_image_path = None
        self.intra_image_data = None
        self.extra_image_data = None
        self.intra_pixmap = None
        self.extra_pixmap = None

        # Limpiar las etiquetas de imagen
        self.intra_label.clear()
        self.extra_label.clear()

        # Reset zoom factor
        self.zoom_factor = 1.0

    def apply_theme(self):
        """Aplica el tema oscuro a la interfaz."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QToolBar {
                background-color: #2b2b2b;
                border: none;
                padding: 4px;
            }
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 4px;
            }
            QToolButton:hover {
                background-color: #404040;
            }
            QToolButton:pressed {
                background-color: #505050;
            }
            QPushButton {
                background-color: #0d47a1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1565c0;
            }
            QPushButton:pressed {
                background-color: #0a3d91;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 4px;
            }
            QFrame {
                border: 1px solid #404040;
                border-radius: 4px;
            }
        """)


    def center_scroll_on_point(self, scroll_area, center_x, center_y):
        """Centra el scroll en un punto específico de la imagen."""
        # Obtener el widget contenido en el scroll area
        content_widget = scroll_area.widget()
        if content_widget is None or not content_widget.pixmap():
            return

        # Obtener las dimensiones del contenido y del viewport
        content_size = content_widget.size()
        viewport_size = scroll_area.viewport().size()

        # Calcular las posiciones del scroll teniendo en cuenta el zoom
        scroll_x = max(0, min(
            int(center_x * self.zoom_factor - viewport_size.width() / 2),
            content_size.width() - viewport_size.width()
        ))
        scroll_y = max(0, min(
            int(center_y * self.zoom_factor - viewport_size.height() / 2),
            content_size.height() - viewport_size.height()
        ))

        # Aplicar el scroll después de un pequeño retraso para asegurar que la imagen está cargada
        QTimer.singleShot(100, lambda: self._apply_scroll(scroll_area, scroll_x, scroll_y))

    def _apply_scroll(self, scroll_area, x, y):
        """Aplica el scroll a las coordenadas especificadas."""
        scroll_area.horizontalScrollBar().setValue(x)
        scroll_area.verticalScrollBar().setValue(y)

    def center_both_images(self):
        """Centra ambas imágenes usando el centro de masa."""
        if self.intra_image_data is not None:
            com_y, com_x = calculate_center_of_mass(self.intra_image_data)
            self.center_scroll_on_point(self.intra_scroll, com_x, com_y)

        if self.extra_image_data is not None:
            com_y, com_x = calculate_center_of_mass(self.extra_image_data)
            self.center_scroll_on_point(self.extra_scroll, com_x, com_y)

    def load_default_paths(self):
        """Carga las rutas por defecto desde el archivo config.json."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if 'image_path' in config:
                        self.image_path = config['image_path']
                    if 'results_path' in config:
                        self.results_path = config['results_path']
        except Exception as e:
            print(f"Error al cargar las rutas por defecto: {str(e)}")

    def run_config_dialog(self):
        """Muestra el diálogo de configuración."""
        dialog = ConfigDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.get_config()
            self.image_path = config['image_path']
            self.results_path = config['results_path']
