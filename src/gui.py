from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QWidget, QMessageBox, QCheckBox, QListWidget, QListWidgetItem, QDialog, QSlider)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from astropy.io import fits
import numpy as np
from roddier import calculate_phase_roddier
from zernike_coeff import calculate_wavefront_zernike, recalculate_wavefront_zernike
from utils import preprocess_images, generate_binary_mask

class FitsViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Test de Roddier con Zernike")
        self.setGeometry(100, 100, 1200, 800)

        # Widget central
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout principal
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Botones y área para cargar imágenes
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout)

        # Botón para cargar imágenes intra-focal y extra-focal
        self.load_intra_button = QPushButton("Cargar imagen intra-focal")
        self.load_intra_button.clicked.connect(self.load_intra_image)
        self.button_layout.addWidget(self.load_intra_button)

        self.load_extra_button = QPushButton("Cargar imagen extra-focal")
        self.load_extra_button.clicked.connect(self.load_extra_image)
        self.button_layout.addWidget(self.load_extra_button)

        # Área de visualización para imágenes
        self.image_layout = QHBoxLayout()
        self.layout.addLayout(self.image_layout)

        # Etiquetas para las imágenes
        self.intra_label = QLabel("Imagen Intra-focal")
        self.intra_label.setFixedSize(500, 500)
        self.image_layout.addWidget(self.intra_label)

        self.extra_label = QLabel("Imagen Extra-focal")
        self.extra_label.setFixedSize(500, 500)
        self.image_layout.addWidget(self.extra_label)

        # Layout para gráficos
        self.graph_layout = QHBoxLayout()
        self.layout.addLayout(self.graph_layout)

        # Canvas para gráficos (Reconstructed Wavefront y Zernike Coefficients)
        self.wavefront_figure = Figure()
        self.wavefront_canvas = FigureCanvas(self.wavefront_figure)
        self.graph_layout.addWidget(self.wavefront_canvas)

        self.zernike_figure = Figure()
        self.zernike_canvas = FigureCanvas(self.zernike_figure)
        self.graph_layout.addWidget(self.zernike_canvas)

        # Lista interactiva de selección de Zernike
        self.zernike_selection_layout = QVBoxLayout()
        self.layout.addLayout(self.zernike_selection_layout)
        self.zernike_label = QLabel("Seleccionar Polinomios de Zernike")
        self.zernike_selection_layout.addWidget(self.zernike_label)
        self.zernike_list = QListWidget()
        self.zernike_list.setSelectionMode(QListWidget.MultiSelection)
        self.zernike_selection_layout.addWidget(self.zernike_list)

        # Widget para mostrar valores de coeficientes
        self.zernike_values_label = QLabel("Valores de los Coeficientes de Zernike:")
        self.zernike_selection_layout.addWidget(self.zernike_values_label)

        # Botón para ejecutar el test de Roddier
        self.run_test_button = QPushButton("Ejecutar Test de Roddier")
        self.run_test_button.clicked.connect(self.run_roddier_test)
        self.layout.addWidget(self.run_test_button)

        # Botón para actualizar phi reconstruido
        self.update_phi_button = QPushButton("Actualizar Frente de Onda")
        self.update_phi_button.clicked.connect(self.update_phi_reconstruction)
        self.layout.addWidget(self.update_phi_button)

        # Almacenar las rutas y los datos de las imágenes
        self.intra_image_path = None
        self.extra_image_path = None
        self.intra_image_data = None
        self.extra_image_data = None
        self.zernike_coefficients = []
        self.phi_reconstructed = None
        self.binary_mask = None

    def load_intra_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen intra-focal", "", "FITS Files (*.fits *.fit)")
        if file_path:
            self.intra_image_path = file_path
            self.intra_image_data = self.load_fits_image(file_path)
            self.display_image(self.intra_image_data, self.intra_label)

    def load_extra_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen extra-focal", "", "FITS Files (*.fits *.fit)")
        if file_path:
            self.extra_image_path = file_path
            self.extra_image_data = self.load_fits_image(file_path)
            self.display_image(self.extra_image_data, self.extra_label)

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
            q_image = QImage(normalized_image.data, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap.scaled(label.width(), label.height()))

    def run_roddier_test(self):
        if not self.intra_image_path or not self.extra_image_path:
            QMessageBox.warning(self, "Error", "Debe cargar ambas imágenes intra y extra-focal.")
            return

        try:
            dialog = ImageCropDialog(self.intra_image_data, self.extra_image_data, crop_size=250)
            if dialog.exec_():
                intra_focal = dialog.crop_image(dialog.intra_image, dialog.intra_x_offset, dialog.intra_y_offset)
                extra_focal = dialog.crop_image(dialog.extra_image, dialog.extra_x_offset, dialog.extra_y_offset)

                self.binary_mask = generate_binary_mask(intra_focal, extra_focal, threshold=0.05)

                reconstructed_wavefront = calculate_phase_roddier(intra_focal, extra_focal)

                zernike_coeffs, phi_reconstructed_masked = calculate_wavefront_zernike(
                    reconstructed_wavefront, self.binary_mask, num_terms=5
                )

                self.zernike_coefficients = zernike_coeffs
                self.phi_reconstructed = phi_reconstructed_masked

                self.populate_zernike_list(len(zernike_coeffs))
                self.display_wavefront(phi_reconstructed_masked)
                self.display_zernike_coefficients(zernike_coeffs)
                self.display_zernike_values(zernike_coeffs)
            else:
                QMessageBox.warning(self, "Cancelado", "El recorte de imágenes fue cancelado.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en el Test de Roddier: {e}")

    def update_phi_reconstruction(self):
        if self.phi_reconstructed is None or self.binary_mask is None:
            QMessageBox.warning(self, "Error", "Primero debe ejecutar el Test de Roddier.")
            return

        try:
            selected_indices = [self.zernike_list.row(item) for item in self.zernike_list.selectedItems()]
            selected_coeffs = np.zeros_like(self.zernike_coefficients)

            for idx in selected_indices:
                selected_coeffs[idx] = self.zernike_coefficients[idx]

            phi_reconstructed_masked = recalculate_wavefront_zernike(
                self.phi_reconstructed, self.binary_mask, selected_coeffs
            )

            self.display_wavefront(phi_reconstructed_masked)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al actualizar el frente de onda: {e}")

    def display_wavefront(self, phi_reconstructed_masked):
        ax = self.wavefront_figure.add_subplot(111)
        ax.clear()
        im = ax.imshow(phi_reconstructed_masked, origin='lower', extent=(-1,1,-1,1), cmap="nipy_spectral")
        self.wavefront_figure.colorbar(im, ax=ax)
        ax.set_title("Frente de Onda Reconstruido")
        self.wavefront_canvas.draw()

    def display_zernike_coefficients(self, coefficients):
        ax = self.zernike_figure.add_subplot(111)
        ax.clear()

        labels = [f"Z{i+1}" for i in range(len(coefficients))]
        ax.bar(labels, coefficients, color="blue")
        ax.set_title("Coeficientes de Zernike")
        ax.set_xlabel("Términos de Zernike")
        ax.set_ylabel("Valor")
        self.zernike_canvas.draw()

    def display_zernike_values(self, coefficients):
        values_text = "\n".join([f"Z{i+1}: {coeff:.4f}" for i, coeff in enumerate(coefficients)])
        self.zernike_values_label.setText(f"Valores de los Coeficientes de Zernike:\n{values_text}")

    def populate_zernike_list(self, num_terms):
        self.zernike_list.clear()
        for i in range(num_terms):
            item = QListWidgetItem(f"Zernike Term Z{i+1}")
            item.setCheckState(True)
            self.zernike_list.addItem(item)



class ImageCropDialog(QDialog):
    def __init__(self, intra_image, extra_image, crop_size=250, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ajustar y confirmar recorte de imágenes")
        self.setGeometry(200, 200, 1000, 500)
        self.crop_size = crop_size
        self.intra_image, self.extra_image = preprocess_images(intra_image, extra_image)


        # Variables para los desplazamientos
        self.intra_x_offset = 0
        self.intra_y_offset = 0
        self.extra_x_offset = 0
        self.extra_y_offset = 0

        # Layout principal
        layout = QVBoxLayout(self)

        # Layout para las imágenes (lado a lado)
        self.image_layout = QHBoxLayout()
        self.intra_label = QLabel(self)
        self.extra_label = QLabel(self)
        self.image_layout.addWidget(self.intra_label)
        self.image_layout.addWidget(self.extra_label)
        layout.addLayout(self.image_layout)

        # Controles para ajustar los recortes (deslizadores)
        self.slider_layout = QVBoxLayout()
        self.add_slider_controls("Intra-focal", layout, "intra")
        self.add_slider_controls("Extra-focal", layout, "extra")

        # Botones para confirmar o cancelar
        button_layout = QHBoxLayout()
        self.confirm_button = QPushButton("Confirmar")
        self.confirm_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.confirm_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Mostrar los recortes iniciales
        self.update_images()

    def add_slider_controls(self, label_text, layout, prefix):
        """Añade controles deslizantes para ajustar los recortes."""
        layout.addWidget(QLabel(f"Ajustes de {label_text}"))
        x_slider = QSlider(Qt.Horizontal)
        x_slider.setRange(-100, 100)
        x_slider.valueChanged.connect(lambda value: self.update_offset(prefix, "x", value))
        layout.addWidget(QLabel("Desplazamiento horizontal"))
        layout.addWidget(x_slider)

        y_slider = QSlider(Qt.Horizontal)
        y_slider.setRange(-100, 100)
        y_slider.valueChanged.connect(lambda value: self.update_offset(prefix, "y", value))
        layout.addWidget(QLabel("Desplazamiento vertical"))
        layout.addWidget(y_slider)


    def crop_image(self, image, x_offset, y_offset):
        """Recorta la imagen con los desplazamientos dados, asegurando dimensiones válidas."""
        center_y, center_x = np.array(image.shape) // 2
        half_size = self.crop_size // 2

        # Calcular los límites del recorte
        start_x = max(center_x - half_size + x_offset, 0)
        start_y = max(center_y - half_size + y_offset, 0)
        end_x = min(start_x + self.crop_size, image.shape[1])
        end_y = min(start_y + self.crop_size, image.shape[0])

        # Asegurarse de que el recorte tiene un tamaño válido
        if end_x <= start_x or end_y <= start_y:
            # Crear una región vacía en caso de un recorte inválido
            return np.zeros((self.crop_size, self.crop_size))
        # Realizar el recorte
        cropped_image = image[start_y:end_y, start_x:end_x]
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
            # Print the raw image data shape
            print(f"Image shape before normalization: {image_data.shape}")
            print(f"Min value: {np.min(image_data)}, Max value: {np.max(image_data)}")

            # Normalize if necessary
            if np.min(image_data) < 0 or np.max(image_data) > 255:
                norm = Normalize(vmin=np.min(image_data), vmax=np.max(image_data))
                image_data = norm(image_data) * 255
            image_data = image_data.astype(np.uint8)

            # Optionally transpose or rotate to correct orientation
            # Uncomment if needed
            # image_data = np.transpose(image_data)
            # image_data = np.rot90(image_data, k=1)  # Rotates 90 degrees counter-clockwise

            # Print after possible transformations
            print(f"Image shape after normalization: {image_data.shape}")

            # Convert to QImage
            height, width = image_data.shape
            bytes_per_line = width
            q_image = QImage(
            image_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )

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
