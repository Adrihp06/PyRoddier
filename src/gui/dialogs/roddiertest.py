from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                          QFrame, QSpinBox, QDoubleSpinBox, QFormLayout, QGroupBox, QCheckBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
import numpy as np

class RoddierTestDialog(QDialog):
    def __init__(self, intra_image, extra_image, crop_size=250, parent=None):
        super().__init__(parent)
        self.intra_image = intra_image
        self.extra_image = extra_image
        self.crop_size = crop_size
        self.crop_center = None
        self.cropped_intra = None
        self.cropped_extra = None
        self.telescope_params = {
            'apertura': 0.0,  # en mm
            'focal': 0.0,  # en mm
            'pixel_scale': 0.0,  # en arcsec/pixel
            'max_order': 6,  # orden máximo de Zernike por defecto
            'threshold': 0.5  # threshold por defecto
        }

        # Preprocesar las imágenes
        self.intra_image, self.extra_image = intra_image, extra_image

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
        self.image_layout.addWidget(self.extra_container)

        layout.addWidget(self.image_container)

        # Grupo para los parámetros del telescopio
        telescope_group = QGroupBox("Parámetros del Telescopio")
        telescope_layout = QFormLayout()
        # Diámetro del espejo primario
        self.apertura = QDoubleSpinBox()
        self.apertura.setRange(0.0, 2000.0)
        self.apertura.setValue(900)  # Valor por defecto en mm
        self.apertura.setSuffix(" mm")
        telescope_layout.addRow("Apertura del telescopio:", self.apertura)

        # focal
        self.focal = QDoubleSpinBox()
        self.focal.setRange(0.0, 20000.0)
        self.focal.setValue(7200)  # Valor por defecto en mm
        self.focal.setSuffix(" mm")
        telescope_layout.addRow("Distancia focal:", self.focal)

        # Escala de píxel
        self.pixel_scale_spin = QDoubleSpinBox()
        self.pixel_scale_spin.setRange(0.0, 50.0)
        self.pixel_scale_spin.setValue(15)  # Valor por defecto en "/mm
        self.pixel_scale_spin.setSuffix(" \"/mm")
        telescope_layout.addRow("Escala de píxel:", self.pixel_scale_spin)

        # Binning
        self.binning_spin = QSpinBox()
        self.binning_spin.setRange(1, 4)
        self.binning_spin.setValue(1)  # Valor por defecto
        telescope_layout.addRow("Binning:", self.binning_spin)

        # Threshold
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.3, 0.5)
        self.threshold_spin.setValue(0.5)  # Valor por defecto
        self.threshold_spin.setSingleStep(0.01)
        telescope_layout.addRow("Threshold:", self.threshold_spin)

        # Orden máximo de Zernike
        self.numero_de_terminos = QSpinBox()
        self.numero_de_terminos.setRange(1, 28)
        self.numero_de_terminos.setValue(23)  # Valor por defecto
        telescope_layout.addRow("Número de términos:", self.numero_de_terminos)

        telescope_group.setLayout(telescope_layout)
        layout.addWidget(telescope_group)

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
        """)

        # Mostrar los recortes iniciales
        self.update_images()

    def crop_image(self, image):
        """Recorta la imagen centrada en el centro de masa."""
        # Obtener las dimensiones de la imagen
        height, width = image.shape
        half_size = self.crop_size // 2

        # Calcular el centro de masa
        com_y, com_x = self.calculate_center_of_mass(image)

        # Asegurarse de que el centro de masa es válido
        com_x = min(max(com_x, half_size), width - half_size)
        com_y = min(max(com_y, half_size), height - half_size)

        # Calcular los límites del recorte
        start_x = com_x - half_size
        start_y = com_y - half_size

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
        intra_crop = self.crop_image(self.intra_image)
        extra_crop = self.crop_image(self.extra_image)

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

    def crop_images(self):
        """Guarda los parámetros del telescopio y recorta las imágenes."""
        # Guardar los parámetros del telescopio
        self.telescope_params.update({
            'apertura': self.apertura.value(),
            'focal': self.focal.value(),
            'pixel_scale': self.pixel_scale_spin.value(),
            'binning': self.binning_spin.value(),
            'max_order': self.numero_de_terminos.value(),
            'threshold': self.threshold_spin.value()
        })

        # Recortar las imágenes
        self.cropped_intra = self.crop_image(self.intra_image)
        self.cropped_extra = self.crop_image(self.extra_image)

        # Aceptar el diálogo
        self.accept()

    def get_cropped_images(self):
        """Devuelve las imágenes recortadas."""
        return self.cropped_intra, self.cropped_extra

    def get_telescope_params(self):
        """Obtiene los parámetros actuales del telescopio desde los widgets."""
        self.telescope_params.update({
            'apertura': self.apertura.value(),
            'focal': self.focal.value(),
            'pixel_scale': self.pixel_scale_spin.value(),
            'binning': self.binning_spin.value(),
            'max_order': self.numero_de_terminos.value(),
            'threshold': self.threshold_spin.value()
        })
        return self.telescope_params