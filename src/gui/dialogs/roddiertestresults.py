import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QCheckBox, QScrollArea,
                             QWidget, QHBoxLayout, QPushButton, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from src.core.interferometry import calculate_interferogram
from src.core.psf import calculate_psf

ZERN_NAMES = [
    "Piston", "Tilt X", "Tilt Y", "Defocus",
    "Astigmatismo 45°", "Astigmatismo 0°", "Coma Y", "Coma X",
    "Trefoil Y", "Trefoil X", "Esférica primaria",
    "Astigmatismo secundario 45°", "Astigmatismo secundario 0°",
    "Tetrafoil Y", "Tetrafoil X", "Coma secundaria Y", "Coma secundaria X",
    "Trefoil secundario Y", "Trefoil secundario X", "Esférica secundaria",
    "Pentafoil Y", "Pentafoil X", "Orden superior"
]

class RoddierTestResultsWindow(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)
        self.setMinimumSize(1600, 800)

        self.zernike_coeffs = None
        self.zernike_base = None
        self.zernike_checks = []
        self.annular_mask = None
        self.interferogram_params = None
        self.telescope_params = None

        # Layout principal
        layout = QVBoxLayout(self)

        # Layout para los gráficos
        plots_layout = QHBoxLayout()
        layout.addLayout(plots_layout)

        # Contenedor para el frente de onda
        wavefront_group = QWidget()
        wavefront_layout = QVBoxLayout(wavefront_group)
        self.wavefront_fig = Figure(figsize=(5, 5), dpi=100)
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)
        self.wavefront_canvas = FigureCanvas(self.wavefront_fig)
        wavefront_layout.addWidget(self.wavefront_canvas)
        plots_layout.addWidget(wavefront_group)

        # Contenedor para el interferograma
        interferogram_group = QWidget()
        interferogram_layout = QVBoxLayout(interferogram_group)
        self.interferogram_fig = Figure(figsize=(5, 5), dpi=100)
        self.interferogram_ax = self.interferogram_fig.add_subplot(111)
        self.interferogram_canvas = FigureCanvas(self.interferogram_fig)
        interferogram_layout.addWidget(self.interferogram_canvas)
        plots_layout.addWidget(interferogram_group)

        # Contenedor para la PSF
        psf_group = QWidget()
        psf_layout = QVBoxLayout(psf_group)
        self.psf_fig = Figure(figsize=(5, 5), dpi=100)
        self.psf_ax = self.psf_fig.add_subplot(111)
        self.psf_canvas = FigureCanvas(self.psf_fig)
        self.psf_canvas.mpl_connect('scroll_event', self._on_psf_scroll)
        psf_layout.addWidget(self.psf_canvas)
        plots_layout.addWidget(psf_group)

        # Layout para la parte inferior (Zernike modes y histograma)
        bottom_layout = QHBoxLayout()
        layout.addLayout(bottom_layout)

        # Contenedor para los modos de Zernike (izquierda)
        zernike_container = QWidget()
        zernike_layout = QVBoxLayout(zernike_container)
        bottom_layout.addWidget(zernike_container, stretch=1)

        # Botones para marcar/desmarcar todos los modos
        select_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Marcar todos")
        select_all_button.clicked.connect(self._select_all_modes)
        deselect_all_button = QPushButton("Desmarcar todos")
        deselect_all_button.clicked.connect(self._deselect_all_modes)
        select_buttons_layout.addWidget(select_all_button)
        select_buttons_layout.addWidget(deselect_all_button)
        zernike_layout.addLayout(select_buttons_layout)

        # Área de scroll para los checkboxes
        self.checkbox_area = QScrollArea()
        self.checkbox_widget = QWidget()
        self.checkbox_layout = QVBoxLayout(self.checkbox_widget)
        self.checkbox_area.setWidgetResizable(True)
        self.checkbox_area.setWidget(self.checkbox_widget)
        zernike_layout.addWidget(self.checkbox_area)

        # Contenedor para el histograma (derecha)
        histogram_container = QWidget()
        histogram_layout = QVBoxLayout(histogram_container)
        bottom_layout.addWidget(histogram_container, stretch=1)

        # Figura para el histograma
        self.histogram_fig = Figure(figsize=(5, 5), dpi=100)
        self.histogram_ax = self.histogram_fig.add_subplot(111)
        self.histogram_canvas = FigureCanvas(self.histogram_fig)
        histogram_layout.addWidget(self.histogram_canvas)

        # Botones inferiores
        button_layout = QHBoxLayout()
        export_button = QPushButton("Exportar Resultados")
        export_button.clicked.connect(self.export_results)
        close_button = QPushButton("Cerrar")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(export_button)
        button_layout.addWidget(close_button)
        layout.addLayout(button_layout)

    def update_plots(self, zernike_coeffs, zernike_base, annular_mask, interferogram_params, telescope_params):
        self.zernike_coeffs = zernike_coeffs
        self.zernike_base = zernike_base
        self.annular_mask = annular_mask
        self.interferogram_params = interferogram_params
        self.telescope_params = telescope_params

        self._create_checkboxes()
        self._update_wavefront_plot()
        self._update_histogram()

    def _create_checkboxes(self):
        for cb in self.zernike_checks:
            self.checkbox_layout.removeWidget(cb)
            cb.deleteLater()
        self.zernike_checks = []

        # Limitar a máximo 23 elementos (0-22)
        max_terms = min(len(self.zernike_coeffs), 23)

        for i, coeff in enumerate(self.zernike_coeffs[:max_terms]):
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

        higher_order_sum = 0

        # Limitar a máximo 23 elementos (0-22)
        max_terms = min(len(self.zernike_checks), 23)

        for i, cb in enumerate(self.zernike_checks[:max_terms]):
            if cb.isChecked():
                if i == 22:  # Último término (22) es la suma de los superiores
                    # Sumar todos los coeficientes por encima del término 22
                    for j in range(22, len(self.zernike_coeffs)):
                        higher_order_sum += self.zernike_coeffs[j] * self.zernike_base[j]
                    active_contrib += higher_order_sum
                else:
                    active_contrib += self.zernike_coeffs[i] * self.zernike_base[i]

        # Aplicar máscara anular si existe
        if hasattr(self, 'annular_mask') and self.annular_mask is not None:
            # Crear una máscara para los valores fuera de la pupila
            mask = self.annular_mask == 0
            # Crear un array enmascarado
            active_contrib = np.ma.masked_array(active_contrib, mask=mask)


        # Limpiar figura y ejes anteriores
        self.wavefront_ax.clear()
        self.wavefront_fig.clf()
        self.wavefront_ax = self.wavefront_fig.add_subplot(111)

        # Crear un mapa de colores personalizado que tenga blanco para valores enmascarados
        cmap = plt.cm.nipy_spectral
        cmap.set_bad('white')  # Establecer el color para valores enmascarados como blanco

        # Rotar la imagen 180 grados antes de mostrarla
        wavefront_for_calc = np.ma.getdata(active_contrib)  # sin máscara, sin rotación
        wavefront_for_display = np.flipud(active_contrib)   # solo para mostrar
        # Visualizar con escalas fijas simétricas
        im = self.wavefront_ax.imshow(
            wavefront_for_display,
            origin='lower',
            cmap=cmap,
            aspect='equal'
        )

        self.wavefront_fig.colorbar(im, ax=self.wavefront_ax)
        self.wavefront_ax.set_title("Suma de modos Zernike seleccionados")
        self.wavefront_canvas.draw()

        # Actualizar el interferograma y la PSF
        self._update_interferogram_plot(wavefront_for_calc)
        self._update_psf_plot(wavefront_for_calc)

    def _update_interferogram_plot(self, wavefront):
        if wavefront is None or self.annular_mask is None or self.interferogram_params is None:
            return

        # Calcular el interferograma
        interferogram = calculate_interferogram(
            wavefront,
            self.interferogram_params['reference_frequency'],
            self.interferogram_params['reference_intensity'],
            self.annular_mask
        )

        # Limpiar figura y ejes anteriores
        self.interferogram_ax.clear()
        self.interferogram_fig.clf()
        self.interferogram_ax = self.interferogram_fig.add_subplot(111)

        # Visualizar el interferograma
        im = self.interferogram_ax.imshow(
            interferogram,
            cmap='gray',
            aspect='equal'
        )

        self.interferogram_ax.set_title("Interferograma")
        self.interferogram_canvas.draw()

    def _update_psf_plot(self, wavefront):
        if wavefront is None or self.annular_mask is None or self.telescope_params is None:
            return

        # Calcular la PSF
        psf, psf_log = calculate_psf(
            wavefront,  # Aplicar la máscara al frente de onda
            self.annular_mask  # Usar la máscara anular como pupila
        )

        # Limpiar figura y ejes anteriores
        self.psf_ax.clear()
        self.psf_fig.clf()
        self.psf_ax = self.psf_fig.add_subplot(111)

        # Visualizar la PSF en escala logarítmica
        im = self.psf_ax.imshow(
            psf_log,  # Ya está en escala logarítmica
            cmap='viridis',
            aspect='equal'
        )

        self.psf_ax.set_title("PSF (escala logarítmica)")
        self.psf_fig.colorbar(im, ax=self.psf_ax)
        self.psf_canvas.draw()

    def _on_psf_scroll(self, event):
        """Manejador del evento de scroll para hacer zoom en la PSF."""
        if event.inaxes != self.psf_ax:
            return

        # Obtener los límites actuales
        xlim = self.psf_ax.get_xlim()
        ylim = self.psf_ax.get_ylim()

        # Calcular el centro del zoom
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        # Factor de zoom (aumentar o disminuir)
        zoom_factor = 1.1 if event.button == 'up' else 0.9

        # Aplicar el zoom manteniendo el centro
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor

        self.psf_ax.set_xlim(x_center - new_width/2, x_center + new_width/2)
        self.psf_ax.set_ylim(y_center - new_height/2, y_center + new_height/2)

        # Redibujar el canvas
        self.psf_canvas.draw()

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

    def _select_all_modes(self):
        """Marca todos los modos de Zernike de manera eficiente."""
        # Actualizar el estado de los checkboxes sin emitir señales
        for cb in self.zernike_checks:
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        # Actualizar el plot una sola vez
        self._update_wavefront_plot()

    def _deselect_all_modes(self):
        """Desmarca todos los modos de Zernike de manera eficiente."""
        # Actualizar el estado de los checkboxes sin emitir señales
        for cb in self.zernike_checks:
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        # Actualizar el plot una sola vez
        self._update_wavefront_plot()

    def _update_histogram(self):
        if self.zernike_coeffs is None:
            return

        # Limitar a máximo 23 elementos (0-22)
        max_terms = min(len(self.zernike_coeffs), 23)
        coeffs = self.zernike_coeffs[:max_terms]
        names = [ZERN_NAMES[i] if i < len(ZERN_NAMES) else f"Z{i+1}" for i in range(max_terms)]

        # Limpiar el histograma anterior
        self.histogram_ax.clear()

        # Crear el histograma
        bars = self.histogram_ax.bar(range(max_terms), coeffs, color='skyblue')

        # Colorear las barras según la magnitud
        for bar in bars:
            magnitude = abs(bar.get_height())
            if magnitude > 0.1:
                bar.set_color('#FF6666')
            elif magnitude > 0.05:
                bar.set_color('#FF9966')
            elif magnitude > 0.01:
                bar.set_color('#FFCC66')

        # Configurar el histograma
        self.histogram_ax.set_xticks(range(max_terms))
        self.histogram_ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        self.histogram_ax.set_ylabel("Coeficiente")
        self.histogram_ax.set_title("Coeficientes de Zernike")
        self.histogram_ax.grid(True, linestyle='--', alpha=0.7)

        # Ajustar el layout para acomodar las etiquetas largas
        self.histogram_fig.subplots_adjust(bottom=0.3)  # Aumentar espacio para las etiquetas
        self.histogram_canvas.draw()
