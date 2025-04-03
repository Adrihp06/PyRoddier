from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                          QFrame, QSlider, QMessageBox, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.colors import Normalize
import numpy as np
from utils.image_processing import preprocess_images, center_image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

class ImageCropDialog(QDialog):
    def __init__(self, intra_image, extra_image, crop_size=250, parent=None):
        super().__init__(parent)
        self.intra_image = intra_image
        self.extra_image = extra_image
        self.crop_size = crop_size
        self.crop_center = None
        self.cropped_intra = None
        self.cropped_extra = None
        self.telescope_params = {
            'primary_diameter': 0.0,  # en mm
            'secondary_diameter': 0.0,  # en mm
            'pixel_scale': 0.0,  # en arcsec/pixel
            'masking_value': 0.05,  # valor por defecto para la máscara
            'max_order': 6  # orden máximo de Zernike por defecto
        }

        # Preprocesar las imágenes
        self.intra_image, self.extra_image= preprocess_images(intra_image, extra_image)

        # Variables para los desplazamientos
        self.intra_x_offset = 0
        self.intra_y_offset = 0
        self.extra_x_offset = 0
        self.extra_y_offset = 0

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

        # Botón de centrado para imagen intra-focal
        self.intra_center_button = QPushButton("Centrar")
        self.intra_center_button.clicked.connect(lambda: self.center_image("intra"))
        self.intra_layout.addWidget(self.intra_center_button)

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

        # Botón de centrado para imagen extra-focal
        self.extra_center_button = QPushButton("Centrar")
        self.extra_center_button.clicked.connect(lambda: self.center_image("extra"))
        self.extra_layout.addWidget(self.extra_center_button)

        self.image_layout.addWidget(self.extra_container)

        layout.addWidget(self.image_container)

        # Controles para ajustar los recortes (deslizadores)
        self.slider_container = QFrame()
        self.slider_container.setFrameStyle(QFrame.StyledPanel)
        self.slider_layout = QVBoxLayout(self.slider_container)
        self.add_slider_controls("Intra-focal", self.slider_layout, "intra")
        self.add_slider_controls("Extra-focal", self.slider_layout, "extra")
        layout.addWidget(self.slider_container)

        # Grupo para los parámetros del telescopio
        telescope_group = QGroupBox("Parámetros del Telescopio")
        telescope_layout = QFormLayout()

        # Diámetro del espejo primario
        self.primary_diameter_spin = QDoubleSpinBox()
        self.primary_diameter_spin.setRange(0.0, 10000.0)
        self.primary_diameter_spin.setValue(500.0)  # Valor por defecto en mm
        self.primary_diameter_spin.setSuffix(" mm")
        telescope_layout.addRow("Diámetro del espejo primario:", self.primary_diameter_spin)

        # Diámetro del espejo secundario
        self.secondary_diameter_spin = QDoubleSpinBox()
        self.secondary_diameter_spin.setRange(0.0, 1000.0)
        self.secondary_diameter_spin.setValue(300.0)  # Valor por defecto en mm
        self.secondary_diameter_spin.setSuffix(" mm")
        telescope_layout.addRow("Diámetro del espejo secundario:", self.secondary_diameter_spin)

        # Escala de píxel
        self.pixel_scale_spin = QDoubleSpinBox()
        self.pixel_scale_spin.setRange(0.0, 50.0)
        self.pixel_scale_spin.setValue(28.65)  # Valor por defecto en "/mm
        self.pixel_scale_spin.setSuffix(" \"/mm")
        telescope_layout.addRow("Escala de píxel:", self.pixel_scale_spin)

        # Valor de máscara
        self.masking_value_spin = QDoubleSpinBox()
        self.masking_value_spin.setRange(0.0, 1.0)
        self.masking_value_spin.setValue(0.05)  # Valor por defecto
        self.masking_value_spin.setSingleStep(0.01)
        telescope_layout.addRow("Valor de máscara:", self.masking_value_spin)

        # Orden máximo de Zernike
        self.max_order_spin = QSpinBox()
        self.max_order_spin.setRange(1, 20)
        self.max_order_spin.setValue(6)  # Valor por defecto
        telescope_layout.addRow("Orden máximo de Zernike:", self.max_order_spin)

        telescope_group.setLayout(telescope_layout)
        layout.addWidget(telescope_group)

        # Botones para confirmar o cancelar
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.crop_button = QPushButton("Recortar")
        self.crop_button.clicked.connect(self.crop_images)
        button_layout.addWidget(self.crop_button)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

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

        # Mostrar los recortes iniciales
        self.update_images()

    def add_slider_controls(self, label_text, layout, prefix):
        """Añade controles deslizantes para ajustar los recortes."""
        group_container = QFrame()
        group_container.setFrameStyle(QFrame.StyledPanel)
        group_layout = QVBoxLayout(group_container)

        title = QLabel(f"Ajustes de {label_text}")
        title.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(title)

        # Deslizador horizontal
        x_label = QLabel("Desplazamiento horizontal")
        x_label.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(x_label)

        x_slider = QSlider(Qt.Horizontal)
        x_slider.setRange(-100, 100)
        x_slider.setValue(0)
        x_slider.valueChanged.connect(lambda value: self.update_offset(prefix, "x", value))
        group_layout.addWidget(x_slider)

        # Deslizador vertical
        y_label = QLabel("Desplazamiento vertical")
        y_label.setAlignment(Qt.AlignCenter)
        group_layout.addWidget(y_label)

        y_slider = QSlider(Qt.Horizontal)
        y_slider.setRange(-100, 100)
        y_slider.setValue(0)
        y_slider.valueChanged.connect(lambda value: self.update_offset(prefix, "y", value))
        group_layout.addWidget(y_slider)

        # Almacenar referencias a los sliders
        if prefix == "intra":
            self.intra_x_slider = x_slider
            self.intra_y_slider = y_slider
        else:
            self.extra_x_slider = x_slider
            self.extra_y_slider = y_slider

        layout.addWidget(group_container)

    def crop_image(self, image, x_offset, y_offset):
        """Recorta la imagen con los desplazamientos dados, asegurando dimensiones válidas."""
        # Obtener las dimensiones de la imagen
        height, width = image.shape
        half_size = self.crop_size // 2

        # Calcular el centro de masa
        com_y, com_x = self.calculate_center_of_mass(image)

        # Asegurarse de que el centro de masa es válido
        com_x = min(max(com_x, half_size), width - half_size)
        com_y = min(max(com_y, half_size), height - half_size)

        # Calcular los límites del recorte
        start_x = com_x - half_size + x_offset
        start_y = com_y - half_size + y_offset

        # Asegurarse de que los límites están dentro de la imagen
        if start_x < 0:
            start_x = 0
        if start_y < 0:
            start_y = 0
        if start_x + self.crop_size > width:
            start_x = width - self.crop_size
        if start_y + self.crop_size > height:
            start_y = height - self.crop_size

        # Realizar el recorte
        cropped_image = image[start_y:start_y + self.crop_size, start_x:start_x + self.crop_size]

        # Verificar que el recorte tiene las dimensiones correctas
        if cropped_image.shape != (self.crop_size, self.crop_size):
            # Si el recorte no tiene el tamaño correcto, rellenar con ceros
            result = np.zeros((self.crop_size, self.crop_size))
            result[:cropped_image.shape[0], :cropped_image.shape[1]] = cropped_image
            return result

        return cropped_image

    def update_images(self):
        """Actualiza las imágenes recortadas en los QLabel."""

        intra_crop = self.crop_image(self.intra_image, self.intra_x_offset, self.intra_y_offset)
        extra_crop = self.crop_image(self.extra_image, self.extra_x_offset, self.extra_y_offset)

        self.intra_label.setPixmap(self.create_pixmap(intra_crop))
        self.extra_label.setPixmap(self.create_pixmap(extra_crop))

    def create_pixmap(self, image_data):
        """Converts image data into a QPixmap for display in QLabel."""
        if image_data is not None and np.any(image_data):
            # Normalizar la imagen para visualización
            normalized = image_data - np.min(image_data)
            if np.max(normalized) > 0:
                normalized = (normalized / np.max(normalized) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(image_data, dtype=np.uint8)

            # Convert to QImage
            height, width = normalized.shape
            bytes_per_line = width
            q_image = QImage(normalized.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            # Scale while preserving aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            return pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        return QPixmap()

    def update_offset(self, prefix, axis, value):
        """Actualiza los desplazamientos según el deslizador."""
        if prefix == "intra":
            if axis == "x":
                self.intra_x_offset = value
            elif axis == "y":
                self.intra_y_offset = value
        elif prefix == "extra":
            if axis == "x":
                self.extra_x_offset = value
            elif axis == "y":
                self.extra_y_offset = value

        # Validar que los offsets son válidos antes de actualizar
        try:
            self.update_images()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Recorte inválido: {e}")
            # Reset sliders to prevent invalid values
            if prefix == "intra":
                if axis == "x":
                    self.intra_x_offset = 0
                elif axis == "y":
                    self.intra_y_offset = 0
            elif prefix == "extra":
                if axis == "x":
                    self.extra_x_offset = 0
                elif axis == "y":
                    self.extra_y_offset = 0

    def center_image(self, image_type):
        """Centra la imagen seleccionada usando el centro de masa."""
        if image_type == "intra":
            self.intra_image = center_image(self.intra_image)
            self.intra_x_offset = 0
            self.intra_y_offset = 0
            # Resetear los sliders
            self.intra_x_slider.setValue(0)
            self.intra_y_slider.setValue(0)
        else:
            self.extra_image = center_image(self.extra_image)
            self.extra_x_offset = 0
            self.extra_y_offset = 0
            # Resetear los sliders
            self.extra_x_slider.setValue(0)
            self.extra_y_slider.setValue(0)

        # Actualizar la visualización
        self.update_images()

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

    def on_click(self, event):
        if event.inaxes:
            self.crop_center = (int(event.ydata), int(event.xdata))
            self.crop_images()

    def crop_images(self):
        # Guardar los parámetros del telescopio
        self.telescope_params = {
            'primary_diameter': self.primary_diameter_spin.value(),
            'secondary_diameter': self.secondary_diameter_spin.value(),
            'pixel_scale': self.pixel_scale_spin.value(),
            'masking_value': self.masking_value_spin.value(),
            'max_order': self.max_order_spin.value()
        }

        # Recorta cada imagen usando su offset y el centro de masa
        self.cropped_intra = self.crop_image(
            self.intra_image, self.intra_x_offset, self.intra_y_offset
        )
        self.cropped_extra = self.crop_image(
            self.extra_image, self.extra_x_offset, self.extra_y_offset
        )

        self.accept()

    def get_cropped_images(self):
        if self.cropped_intra is None or self.cropped_extra is None:
            raise ValueError("Las imágenes no han sido recortadas. Por favor, use el botón 'Recortar' o haga clic en las imágenes.")
        return self.cropped_intra, self.cropped_extra

    def get_telescope_params(self):
        return self.telescope_params