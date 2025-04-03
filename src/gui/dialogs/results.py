import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QCheckBox, QScrollArea,
                             QWidget, QHBoxLayout, QPushButton, QFileDialog)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtGui import QColor, QPalette
from scipy.fft import fft2, ifft2, fftfreq

ZERN_NAMES = [
    "Piston", "Tilt X", "Tilt Y", "Defocus",
    "Astigmatismo 45°", "Astigmatismo 0°", "Coma Y", "Coma X",
    "Trefoil Y", "Trefoil X", "Esférica primaria",
    "Astigmatismo secundario 45°", "Astigmatismo secundario 0°",
    "Tetrafoil Y", "Tetrafoil X", "Coma secundaria Y", "Coma secundaria X",
    "Trefoil secundario Y", "Trefoil secundario X", "Esférica secundaria"
]

class ResultsWindow(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.zernike_coeffs = None
        self.zernike_base = None
        self.zernike_checks = []

        layout = QVBoxLayout(self)

        self.wavefront_fig = Figure(figsize=(5, 5), dpi=100)
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)
        self.wavefront_canvas = FigureCanvas(self.wavefront_fig)
        layout.addWidget(self.wavefront_canvas)

        self.checkbox_area = QScrollArea()
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_area.setWidgetResizable(True)
        self.checkbox_area.setWidget(self.checkbox_widget)
        layout.addWidget(self.checkbox_area)

        # Botones inferiores
        button_layout = QHBoxLayout()
        export_button = QPushButton("Exportar Resultados")
        export_button.clicked.connect(self.export_results)
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(export_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def update_plots(self, wavefront, zernike_coeffs, zernike_base, annular_mask=None):
        self.zernike_coeffs = zernike_coeffs
        self.zernike_base = zernike_base
        self.annular_mask = annular_mask

        self._create_checkboxes()
        self._update_wavefront_plot()

    def _create_checkboxes(self):
        for cb in self.zernike_checks:
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.zernike_checks = []

        for i, coeff in enumerate(self.zernike_coeffs):
            name = ZERN_NAMES[i] if i < len(ZERN_NAMES) else f"Z{i+1}"
            label = f"Z{i+1} – {name} ({coeff:.3f})"
            cb = QCheckBox(label)
            cb.setChecked(i != 0)
            cb.stateChanged.connect(self._update_wavefront_plot)

            magnitude = abs(coeff)
            color = QColor("lightgray")
            if magnitude > 0.01:
                color = QColor("#FFCC66")
            if magnitude > 0.05:
                color = QColor("#FF9966")
            if magnitude > 0.1:
                color = QColor("#FF6666")

            palette = cb.palette()
            palette.setColor(QPalette.Base, color)
            palette.setColor(QPalette.Window, color)
            cb.setAutoFillBackground(True)
            cb.setPalette(palette)

            self.checkbox_layout.addWidget(cb)
            self.zernike_checks.append(cb)

    def _update_wavefront_plot(self):
        if self.zernike_base is None or self.zernike_coeffs is None:
            return

        active_contrib = np.zeros_like(self.zernike_base[0])
        for i, cb in enumerate(self.zernike_checks):
            if cb.isChecked():
                active_contrib += self.zernike_coeffs[i] * self.zernike_base[i]

        # Aplicar máscara anular si existe
        if hasattr(self, 'annular_mask') and self.annular_mask is not None:
            active_contrib = active_contrib * self.annular_mask

        # Limpiar figura y ejes anteriores
        self.wavefront_ax.clear()
        self.wavefront_fig.clf()
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)

        # Visualizar sin escalas fijas
        im = self.wavefront_ax.imshow(
            active_contrib,
            origin='lower',
            cmap="nipy_spectral",
            aspect='equal'
        )

        self.wavefront_fig.colorbar(im, ax=self.wavefront_ax)
        self.wavefront_ax.set_title("Suma de modos Zernike seleccionados")
        self.wavefront_canvas.draw()

    def export_results(self):
        if self.zernike_coeffs is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar Coeficientes", "zernike_coeffs.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, 'w') as f:
            for i, coeff in enumerate(self.zernike_coeffs):
                name = ZERN_NAMES[i] if i < len(ZERN_NAMES) else f"Z{i+1}"
                f.write(f"Z{i+1} - {name}: {coeff:.6f}\n")
