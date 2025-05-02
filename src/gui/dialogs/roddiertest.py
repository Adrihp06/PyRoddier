from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                          QFrame, QFormLayout, QGroupBox, QLineEdit, QMessageBox, QComboBox, )
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
import os
from src.common.config import get_config_paths
from src.common.utils import calculate_center_of_mass
import json

class RoddierTestDialog(QDialog):
    def __init__(self, intra_image, extra_image, crop_size=250, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test de Roddier")
        self.setModal(True)

        # Get configuration paths
        config_paths = get_config_paths()
        self.config_path = config_paths['telescope_dir']

        # Preprocesar las imágenes
        self.intra_image = intra_image
        self.extra_image = extra_image
        self.crop_size = crop_size
        self.crop_center = None
        self.cropped_intra = None
        self.cropped_extra = None

        # Telescope parameters
        self.telescope_params = {
            'espejo_primario': 0.0,  # en mm
            'espejo_secundario': 0.0,  # en mm
            'focal': 0.0,  # en mm
            'apertura': 0.0,  # en mm
            'tamano_pixel': 0.0,  # en micras
            'binning': "1x1"  # formato "NxN"
        }

        # Roddier test parameters
        self.roddier_params = {
            'max_order': 23,  # orden máximo de Zernike
            'threshold': 0.5,  # threshold para la máscara
            'crop_size': crop_size  # tamaño del recorte
        }

        # Interferogram parameters
        self.interferogram_params = {
            'fringes': 4,  # número de franjas horizontales
            'reference_frequency': 1.0,  # frecuencia de referencia
            'reference_intensity': 0.5  # intensidad de referencia
        }

        # Layout principal
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Layout para las imágenes (lado a lado)
        self.image_container = QFrame()
        self.image_container.setFrameStyle(QFrame.StyledPanel)
        self.image_layout = QHBoxLayout(self.image_container)
        self.image_layout.setSpacing(20)

        # Contenedor para imagen intra-focal
        self.intra_container = QFrame()
        self.intra_container.setFrameStyle(QFrame.StyledPanel)
        self.intra_layout = QVBoxLayout(self.intra_container)
        self.intra_title = QLabel("Imagen Intra-focal")
        self.intra_title.setAlignment(Qt.AlignCenter)
        self.intra_layout.addWidget(self.intra_title)
        self.intra_label = QLabel(self)
        self.intra_label.setAlignment(Qt.AlignCenter)
        self.intra_layout.addWidget(self.intra_label)
        self.image_layout.addWidget(self.intra_container)

        # Contenedor para imagen extra-focal
        self.extra_container = QFrame()
        self.extra_container.setFrameStyle(QFrame.StyledPanel)
        self.extra_layout = QVBoxLayout(self.extra_container)
        self.extra_title = QLabel("Imagen Extra-focal")
        self.extra_title.setAlignment(Qt.AlignCenter)
        self.extra_layout.addWidget(self.extra_title)
        self.extra_label = QLabel(self)
        self.extra_label.setAlignment(Qt.AlignCenter)
        self.extra_layout.addWidget(self.extra_label)
        self.image_layout.addWidget(self.extra_container)

        layout.addWidget(self.image_container)

        # Grupo para los parámetros del telescopio
        telescope_group = QGroupBox("Parámetros del Telescopio")
        telescope_layout = QFormLayout()

        # Campo para espejo primario
        self.espejo_primario_edit = QLineEdit()
        telescope_layout.addRow("Espejo primario (mm):", self.espejo_primario_edit)

        # Campo para espejo secundario
        self.espejo_secundario_edit = QLineEdit()
        telescope_layout.addRow("Espejo secundario (mm):", self.espejo_secundario_edit)

        # Campo para focal
        self.focal_edit = QLineEdit()
        telescope_layout.addRow("Focal (mm):", self.focal_edit)

        # Campo para apertura
        self.apertura_edit = QLineEdit()
        telescope_layout.addRow("Apertura (mm):", self.apertura_edit)

        # Campo para tamaño de pixel
        self.tamano_pixel_edit = QLineEdit()
        telescope_layout.addRow("Tamaño de pixel (μm):", self.tamano_pixel_edit)

        # Campo para binning
        self.binning_edit = QLineEdit()
        self.binning_edit.setText("1x1")
        telescope_layout.addRow("Binning:", self.binning_edit)

        # ComboBox para cargar configuraciones
        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self.load_selected_config)
        telescope_layout.addRow("Cargar configuración:", self.config_combo)

        telescope_group.setLayout(telescope_layout)
        layout.addWidget(telescope_group)

        # Grupo para parámetros del test de Roddier
        roddier_group = QGroupBox("Parámetros del Test de Roddier")
        roddier_layout = QFormLayout()

        # Campo para max_order
        self.max_order_edit = QLineEdit()
        self.max_order_edit.setText("23")
        roddier_layout.addRow("Orden máximo de Zernike:", self.max_order_edit)

        # Campo para threshold
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setText("0.5")
        roddier_layout.addRow("Threshold:", self.threshold_edit)

        roddier_group.setLayout(roddier_layout)
        layout.addWidget(roddier_group)

        # Grupo para parámetros del interferograma
        interferogram_group = QGroupBox("Parámetros del Interferograma")
        interferogram_layout = QFormLayout()

        # Campo para número de franjas
        self.fringes_edit = QLineEdit()
        self.fringes_edit.setText("4")
        interferogram_layout.addRow("Número de franjas horizontales:", self.fringes_edit)

        # Campo para frecuencia de referencia
        self.reference_freq_edit = QLineEdit()
        self.reference_freq_edit.setText("1.0")
        interferogram_layout.addRow("Frecuencia de referencia:", self.reference_freq_edit)

        # Campo para intensidad de referencia
        self.reference_intensity_edit = QLineEdit()
        self.reference_intensity_edit.setText("0.5")
        interferogram_layout.addRow("Intensidad de referencia:", self.reference_intensity_edit)

        interferogram_group.setLayout(interferogram_layout)
        layout.addWidget(interferogram_group)

        # Botones para confirmar o cancelar
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.crop_button = QPushButton("Ejecutar el Test de Roddier")
        self.crop_button.clicked.connect(self.crop_images)
        button_layout.addWidget(self.crop_button)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        # Cargar configuraciones disponibles
        self.load_configurations()

        # Aplicar el mismo estilo que la ventana principal
        self.setStyleSheet("""
            QDialog {
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
            QFrame {
                border: 1px solid #404040;
                border-radius: 4px;
            }
            QLineEdit {
                background-color: #363636;
                color: white;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
            }
            QGroupBox {
                border: 1px solid #404040;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QComboBox {
                background-color: #363636;
                color: white;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 4px;
            }
        """)

        # Mostrar los recortes iniciales
        self.update_images()

    def crop_image(self, image):
        """Recorta la imagen al tamaño especificado centrada en el centro de masa."""
        if image is None:
            return None

        # Calcular el centro de masa
        com_y, com_x = calculate_center_of_mass(image)
        # Calcular los límites del recorte
        half_size = self.crop_size // 2
        y_start = max(0, com_y - half_size)
        y_end = min(image.shape[0], com_y + half_size)
        x_start = max(0, com_x - half_size)
        x_end = min(image.shape[1], com_x + half_size)

        # Asegurarse de que el recorte tenga el tamaño correcto
        cropped = image[y_start:y_end, x_start:x_end]

        # Si el recorte es más pequeño que crop_size, rellenar con ceros
        if cropped.shape[0] < self.crop_size or cropped.shape[1] < self.crop_size:
            padded = np.zeros((self.crop_size, self.crop_size))
            y_offset = (self.crop_size - cropped.shape[0]) // 2
            x_offset = (self.crop_size - cropped.shape[1]) // 2
            padded[y_offset:y_offset+cropped.shape[0],
                   x_offset:x_offset+cropped.shape[1]] = cropped
            return padded

        return cropped

    def update_images(self):
        """Actualiza las imágenes recortadas."""
        if self.intra_image is not None:
            self.cropped_intra = self.crop_image(self.intra_image)
            if self.cropped_intra is not None:
                self.intra_label.setPixmap(self.create_pixmap(self.cropped_intra))

        if self.extra_image is not None:
            self.cropped_extra = self.crop_image(self.extra_image)
            if self.cropped_extra is not None:
                self.extra_label.setPixmap(self.create_pixmap(self.cropped_extra))

    def create_pixmap(self, image_data):
        """Converts image data into a QPixmap for display in QLabel."""
        if image_data is not None and np.any(image_data):
            # Normalizar la imagen para visualización
            normalized = image_data - np.min(image_data)
            if np.max(normalized) > 0:
                normalized = (normalized / np.max(normalized) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)

            # Asegurarse de que los datos estén contiguos en memoria
            normalized = np.ascontiguousarray(normalized)

            # Convert to QImage
            height, width = normalized.shape
            bytes_per_line = width
            q_image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            q_image = q_image.copy()  # Create a deep copy to ensure data ownership

            # Scale while preserving aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            return pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return QPixmap()

    def crop_images(self):
        """Guarda los parámetros del telescopio y recorta las imágenes."""
        # Obtener los parámetros del telescopio usando la validación existente
        telescope_params = self.get_telescope_params()
        if telescope_params is None:
            # No recortar imágenes si los parámetros no son válidos
            self.cropped_intra = None
            self.cropped_extra = None
            return

        # Actualizar los parámetros del telescopio
        self.telescope_params.update(telescope_params)

        # Recortar las imágenes
        self.cropped_intra = self.crop_image(self.intra_image)
        self.cropped_extra = self.crop_image(self.extra_image)

        # Aceptar el diálogo
        self.accept()

    def get_cropped_images(self):
        """Devuelve las imágenes recortadas."""
        return self.cropped_intra, self.cropped_extra

    def get_telescope_params(self):
        """Retorna los parámetros del telescopio."""
        try:
            # Verificar campos vacíos
            if not self.espejo_primario_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Espejo primario' no puede estar vacío.")
                return None
            if not self.espejo_secundario_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Espejo secundario' no puede estar vacío.")
                return None
            if not self.focal_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Focal' no puede estar vacío.")
                return None
            if not self.apertura_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Apertura' no puede estar vacío.")
                return None
            if not self.tamano_pixel_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Tamaño de pixel' no puede estar vacío.")
                return None
            if not self.binning_edit.text():
                QMessageBox.warning(self, "Error", "El campo 'Binning' no puede estar vacío.")
                return None

            return {
                'espejo_primario': float(self.espejo_primario_edit.text()),
                'espejo_secundario': float(self.espejo_secundario_edit.text()),
                'focal': float(self.focal_edit.text()),
                'apertura': float(self.apertura_edit.text()),
                'tamano_pixel': float(self.tamano_pixel_edit.text()),
                'binning': self.binning_edit.text()
            }
        except ValueError:
            QMessageBox.warning(self, "Error", "Por favor, introduce valores numéricos válidos para los parámetros del telescopio.")
            return None

    def get_roddier_params(self):
        """Retorna los parámetros del test de Roddier."""
        try:
            # Validar campos vacíos
            if not self.max_order_edit.text():
                self.max_order_edit.setText("23")
            if not self.threshold_edit.text():
                self.threshold_edit.setText("0.5")

            return {
                'max_order': int(self.max_order_edit.text()),
                'threshold': float(self.threshold_edit.text()),
                'crop_size': self.crop_size
            }
        except ValueError:
            QMessageBox.warning(self, "Error", "Por favor, introduce valores numéricos válidos para los parámetros del test de Roddier.")
            return None

    def get_interferogram_params(self):
        """Retorna los parámetros del interferograma."""
        try:
            # Validar campos vacíos
            if not self.fringes_edit.text():
                self.fringes_edit.setText("4")
            if not self.reference_freq_edit.text():
                self.reference_freq_edit.setText("1.0")
            if not self.reference_intensity_edit.text():
                self.reference_intensity_edit.setText("0.5")

            return {
                'fringes': int(self.fringes_edit.text()),
                'reference_frequency': float(self.reference_freq_edit.text()),
                'reference_intensity': float(self.reference_intensity_edit.text())
            }
        except ValueError:
            QMessageBox.warning(self, "Error", "Por favor, introduce valores numéricos válidos para los parámetros del interferograma.")
            return None

    def load_selected_config(self, index):
        """Carga la configuración seleccionada del ComboBox."""
        if index == 0:  # "Nueva configuración"
            # Establecer valores por defecto
            self.espejo_primario_edit.clear()
            self.espejo_secundario_edit.clear()
            self.focal_edit.clear()
            self.apertura_edit.clear()
            self.tamano_pixel_edit.clear()
            self.binning_edit.setText("1x1")
            return

        config_name = self.config_combo.currentText()
        if not config_name:  # Si no hay nombre, no hacer nada
            return

        config_file = os.path.join(self.config_path, f"{config_name}.json")

        try:
            with open(config_file, 'r') as f:
                params = json.load(f)
                self.espejo_primario_edit.setText(str(params.get('espejo_primario', '')))
                self.espejo_secundario_edit.setText(str(params.get('espejo_secundario', '')))
                self.focal_edit.setText(str(params.get('focal', '')))
                self.apertura_edit.setText(str(params.get('apertura', '')))
                self.tamano_pixel_edit.setText(str(params.get('tamano_pixel', '')))
                self.binning_edit.setText(str(params.get('binning', '1x1')))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error al cargar la configuración: {str(e)}")

    def load_configurations(self):
        """Carga las configuraciones disponibles en el directorio especificado."""
        if not os.path.exists(self.config_path):
            return

        # Guardar el índice actual
        current_index = self.config_combo.currentIndex()
        current_text = self.config_combo.currentText()

        # Limpiar y añadir la opción de nueva configuración
        self.config_combo.clear()
        self.config_combo.addItem("Nueva configuración")

        # Cargar configuraciones existentes
        for file in os.listdir(self.config_path):
            if file.endswith('.json'):
                config_name = file[:-5]  # Remove .json extension
                if config_name:  # Solo añadir si el nombre no está vacío
                    self.config_combo.addItem(config_name)

        # Restaurar la selección anterior si es posible
        if current_text:
            index = self.config_combo.findText(current_text)
            if index >= 0:
                self.config_combo.setCurrentIndex(index)
            else:
                # Si no se encuentra el texto anterior, seleccionar "Nueva configuración"
                self.config_combo.setCurrentIndex(0)