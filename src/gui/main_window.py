from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QMessageBox, QCheckBox, QListWidget, QListWidgetItem, QDialog, QSlider, QFrame, QSplitter, QScrollArea, QToolBar, QAction, QMenuBar)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QFont, QIcon
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from astropy.io import fits
import numpy as np
import os
from core.roddier import calculate_wavefront
from core.zernike import fit_zernike
from core.interferometry import calculate_interferogram, analyze_interferogram
from utils.image_processing import preprocess_images, generate_common_annular_mask, align_images_winroddier, normalize_images
from gui.dialogs.results import ResultsWindow
from gui.dialogs.image_crop import ImageCropDialog

class FitsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test de Roddier con Zernike")
        self.setGeometry(100, 100, 1200, 800)

        # Variable para el tema
        self.is_dark_theme = True

        # Crear barra de menú
        self.menubar = self.menuBar()
        self.apply_theme()

        # Crear barra de herramientas
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)  # Añadir la barra de herramientas a la ventana principal
        self.apply_theme()

        # Acción para el Test de Roddier
        self.roddier_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'icons', 'roddier.png')), 'Test de Roddier', self)
        self.roddier_action.setStatusTip('Ejecutar Test de Roddier')
        self.roddier_action.triggered.connect(self.run_roddier_test)
        self.toolbar.addAction(self.roddier_action)

        # Acción para Interferometría
        self.interferometry_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'icons', 'interferometry.png')), 'Interferometría', self)
        self.interferometry_action.setStatusTip('Análisis de Interferometría')
        self.interferometry_action.triggered.connect(self.run_interferometry)
        self.toolbar.addAction(self.interferometry_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para resetear con el nuevo icono de papelera
        self.reset_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'icons', 'trash.png')), 'Borrar', self)
        self.reset_action.setStatusTip('Limpiar imágenes y resetear estado')
        self.reset_action.triggered.connect(self.reset_state)
        self.toolbar.addAction(self.reset_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para centrar imágenes
        self.center_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'icons', 'center.png')), 'Centrar', self)
        self.center_action.setStatusTip('Centrar ambas imágenes')
        self.center_action.triggered.connect(self.center_both_images)
        self.toolbar.addAction(self.center_action)

        # Separador
        self.toolbar.addSeparator()

        # Acción para cambiar el tema
        self.theme_action = QAction(QIcon(os.path.join(os.path.dirname(__file__), '..', '..', 'icons', 'theme.png')), 'Cambiar Tema', self)
        self.theme_action.setStatusTip('Cambiar entre modo claro y oscuro')
        self.theme_action.triggered.connect(self.toggle_theme)
        self.toolbar.addAction(self.theme_action)

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
        image_data = self.load_fits_image(file_path)
        if image_data is None:
            return

        # Aplicar transformaciones necesarias para imagen extra-focal
        if not is_intrafocal:
            image_data = np.flipud(np.fliplr(image_data))

        # Calcular el centro de masa
        com_y, com_x = self.calculate_center_of_mass(image_data)

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
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen intra-focal", "", "FITS Files (*.fits *.fit)")
        self.process_and_display_image(file_path, is_intrafocal=True)

    def load_extra_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen extra-focal", "", "FITS Files (*.fits *.fit)")
        self.process_and_display_image(file_path, is_intrafocal=False)

    def load_fits_image(self, file_path):
        try:
            with fits.open(file_path) as hdul:
                img_data = hdul[0].data
                return img_data
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar la imagen FITS: {e}")
            return None

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

        # Preprocesar las imágenes antes del recorte
        preprocessed_intra, preprocessed_extra = preprocess_images(self.intra_image_data, self.extra_image_data)

        # Abrir diálogo de recorte y obtener parámetros del telescopio
        dialog = ImageCropDialog(preprocessed_intra, preprocessed_extra, crop_size=250)
        if dialog.exec_() == QDialog.Accepted:
            # 2. Obtener imágenes recortadas y parámetros
            cropped_intra, cropped_extra = dialog.get_cropped_images()
            normalized_intra, normalized_extra = normalize_images(cropped_intra, cropped_extra)

            telescope_params = dialog.get_telescope_params()
            primary_diameter = telescope_params['primary_diameter']
            secondary_diameter = telescope_params['secondary_diameter']
            pixel_scale = telescope_params['pixel_scale']
            masking_value = telescope_params['masking_value']
            max_order = telescope_params['max_order']

            # 3. Alineación
            intra_aligned, extra_aligned, _ = align_images_winroddier(
                normalized_intra, normalized_extra, masking_value=masking_value
            )

            # 4. Generar máscara
            annular_mask, center, R_in, R_out = generate_common_annular_mask(
                intra_aligned, extra_aligned,
                masking_value=masking_value,
                secondary_diameter_m=secondary_diameter,
                primary_diameter_m=primary_diameter,
                pixel_scale_m=pixel_scale
            )

            # 5. Calcular frente de onda
            wavefront = calculate_wavefront(intra_aligned, extra_aligned, annular_mask, center)

            # 6. Ajuste de Zernike
            reconstructed_wavefront, zernike_coeffs, zernike_base = fit_zernike(
                wavefront, annular_mask, center, max_order
            )

            # 7. Crear ventana de resultados (como WinRoddier)
            results_window = ResultsWindow("Resultados del Test de Roddier", self)
            results_window.update_plots(
                wavefront=reconstructed_wavefront,
                zernike_coeffs=zernike_coeffs,
                zernike_base=zernike_base,
                annular_mask=annular_mask
            )

            # 8. Mostrar la ventana
            results_window.exec_()

    def run_interferometry(self):
        if not self.intra_image_path or not self.extra_image_path:
            QMessageBox.warning(self, "Error", "Debe cargar ambas imágenes intra y extra-focal.")
            return

        try:
            dialog = ImageCropDialog(self.intra_image_data, self.extra_image_data, crop_size=250)
            if dialog.exec_():
                intra_focal = dialog.crop_image(dialog.intra_image, dialog.intra_x_offset, dialog.intra_y_offset)
                extra_focal = dialog.crop_image(dialog.extra_image, dialog.extra_x_offset, dialog.extra_y_offset)

                # Generate binary mask
                annular_mask, center, R_in, R_out = generate_common_annular_mask(intra_focal, extra_focal)

                # Calcular interferograma
                interferogram = calculate_interferogram(intra_focal)

                # Crear ventana de resultados
                results_window = ResultsWindow("Resultados de Interferometría", self, show_interferogram=True)

                # Actualizar gráficos con la máscara binaria
                results_window.update_plots(interferogram, binary_mask=binary_mask)

                # Formatear resultados numéricos
                results_text = "Análisis de Interferometría:\n"
                results_text += f"Tamaño del interferograma: {interferogram.shape}\n"
                results_text += f"Valor máximo: {np.max(interferogram):.4f}\n"
                results_text += f"Valor mínimo: {np.min(interferogram):.4f}\n"

                results_window.update_results(results_text)
                results_window.show()

            else:
                QMessageBox.warning(self, "Cancelado", "El recorte de imágenes fue cancelado.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en el análisis de interferometría: {e}")

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
        """Aplica el tema actual a la interfaz."""
        if self.is_dark_theme:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QMenuBar {
                    background-color: #363636;
                    color: white;
                    border-bottom: 1px solid #404040;
                }
                QMenuBar::item {
                    padding: 4px 10px;
                    background-color: transparent;
                }
                QMenuBar::item:selected {
                    background-color: #0d47a1;
                }
                QMenu {
                    background-color: #363636;
                    color: white;
                    border: 1px solid #404040;
                }
                QMenu::item:selected {
                    background-color: #0d47a1;
                }
                QToolBar {
                    background-color: #363636;
                    border: none;
                    spacing: 10px;
                    padding: 5px;
                }
                QToolButton {
                    background-color: transparent;
                    border: none;
                    border-radius: 4px;
                    padding: 5px;
                }
                QToolButton:hover {
                    background-color: #0d47a1;
                }
                QToolButton:pressed {
                    background-color: #0a3d91;
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
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #ffffff;
                    color: #000000;
                }
                QMenuBar {
                    background-color: #f0f0f0;
                    color: #000000;
                    border-bottom: 1px solid #d0d0d0;
                }
                QMenuBar::item {
                    padding: 4px 10px;
                    background-color: transparent;
                }
                QMenuBar::item:selected {
                    background-color: #e0e0e0;
                }
                QMenu {
                    background-color: #f0f0f0;
                    color: #000000;
                    border: 1px solid #d0d0d0;
                }
                QMenu::item:selected {
                    background-color: #e0e0e0;
                }
                QToolBar {
                    background-color: #f0f0f0;
                    border: none;
                    spacing: 10px;
                    padding: 5px;
                }
                QToolButton {
                    background-color: transparent;
                    border: none;
                    border-radius: 4px;
                    padding: 5px;
                }
                QToolButton:hover {
                    background-color: #e0e0e0;
                }
                QToolButton:pressed {
                    background-color: #d0d0d0;
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
                    color: #000000;
                    font-size: 14px;
                    padding: 4px;
                }
                QListWidget {
                    background-color: #ffffff;
                    border: 1px solid #d0d0d0;
                    border-radius: 4px;
                    padding: 4px;
                    color: #000000;
                }
                QListWidget::item {
                    padding: 4px;
                    border-radius: 2px;
                }
                QListWidget::item:selected {
                    background-color: #e0e0e0;
                }
                QSlider {
                    height: 20px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #d0d0d0;
                    height: 6px;
                    background: #f0f0f0;
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
                    border: 1px solid #d0d0d0;
                    border-radius: 4px;
                }
            """)

    def toggle_theme(self):
        """Cambia entre el tema claro y oscuro."""
        self.is_dark_theme = not self.is_dark_theme
        self.apply_theme()

    def calculate_center_of_mass(self, image):
        """Calcula el centro de masa de la imagen."""
        # Normalizar la imagen para el cálculo
        normalized = image - np.min(image)
        if np.max(normalized) > 0:
            normalized = normalized / np.max(normalized)

        # Crear máscaras para los píxeles significativos
        threshold = 0.1  # Ajusta este valor según sea necesario
        mask = normalized > threshold

        # Calcular índices de las coordenadas
        y_indices, x_indices = np.indices(image.shape)

        # Calcular centro de masa solo de los píxeles significativos
        total_mass = np.sum(normalized[mask])
        if total_mass > 0:
            com_y = np.sum(y_indices[mask] * normalized[mask]) / total_mass
            com_x = np.sum(x_indices[mask] * normalized[mask]) / total_mass
        else:
            # Si no hay píxeles significativos, usar el centro geométrico
            com_y, com_x = np.array(image.shape) // 2

        return int(com_y), int(com_x)

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
            com_y, com_x = self.calculate_center_of_mass(self.intra_image_data)
            self.center_scroll_on_point(self.intra_scroll, com_x, com_y)

        if self.extra_image_data is not None:
            com_y, com_x = self.calculate_center_of_mass(self.extra_image_data)
            self.center_scroll_on_point(self.extra_scroll, com_x, com_y)
