from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                            QSpinBox, QDoubleSpinBox, QPushButton, QFormLayout)
from PyQt5.QtCore import Qt

class InterferogramConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del Interferograma")
        self.setModal(True)

        # Layout principal
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Form layout para los parámetros
        form_layout = QFormLayout()

        # Número de franjas horizontales
        self.fringes_spinbox = QSpinBox()
        self.fringes_spinbox.setRange(1, 20)
        self.fringes_spinbox.setValue(4)
        form_layout.addRow("Número de franjas horizontales:", self.fringes_spinbox)

        # Frecuencia de referencia
        self.reference_freq_spinbox = QDoubleSpinBox()
        self.reference_freq_spinbox.setRange(0.1, 10.0)
        self.reference_freq_spinbox.setValue(1.0)
        self.reference_freq_spinbox.setSingleStep(0.1)
        form_layout.addRow("Frecuencia de referencia:", self.reference_freq_spinbox)

        # Intensidad de referencia
        self.reference_intensity_spinbox = QDoubleSpinBox()
        self.reference_intensity_spinbox.setRange(0.1, 1.0)
        self.reference_intensity_spinbox.setValue(0.5)
        self.reference_intensity_spinbox.setSingleStep(0.1)
        form_layout.addRow("Intensidad de referencia:", self.reference_intensity_spinbox)

        layout.addLayout(form_layout)

        # Botones
        button_layout = QHBoxLayout()

        self.ok_button = QPushButton("Aceptar")
        self.ok_button.clicked.connect(self.accept)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

    def get_config(self):
        """Retorna la configuración actual del interferograma."""
        return {
            'fringes': self.fringes_spinbox.value(),
            'reference_frequency': self.reference_freq_spinbox.value(),
            'reference_intensity': self.reference_intensity_spinbox.value()
        }