from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class InterferogramResultsDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)

        # Layout principal
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Crear figura de matplotlib
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Inicializar subplots
        self.ax = self.figure.add_subplot(111)

    def update_plot(self, interferogram):
        """Actualiza el plot con el interferograma."""
        self.ax.clear()

        # Mostrar el interferograma
        im = self.ax.imshow(interferogram, cmap='gray', origin='lower')
        self.figure.colorbar(im, ax=self.ax)

        self.ax.set_title('Interferograma')
        self.ax.set_xlabel('X (píxeles)')
        self.ax.set_ylabel('Y (píxeles)')

        # Actualizar el canvas
        self.canvas.draw()