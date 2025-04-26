from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                            QLabel, QFormLayout, QGroupBox,
                            QLineEdit, QMessageBox, QInputDialog, QFileDialog,
                            QComboBox)
from PyQt5.QtCore import Qt
import json
import os
from pathlib import Path
from src.common.config import get_config_paths

class ConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración")
        self.setModal(True)

        # Get configuration paths
        config_paths = get_config_paths()
        self.config_path = config_paths['telescope_dir']
        self.config_file = Path.home() / '.pyroddier' / 'config.json'

        # Layout principal
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Grupo para rutas
        paths_group = QGroupBox("Rutas")
        paths_layout = QFormLayout()

        # Campo para ruta de imágenes
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        self.image_path_button = QPushButton("Seleccionar")
        self.image_path_button.clicked.connect(lambda: self.select_directory(self.image_path_edit))
        paths_layout.addRow("Ruta de imágenes:", self.image_path_edit)
        paths_layout.addRow("", self.image_path_button)

        # Campo para ruta de resultados
        self.results_path_edit = QLineEdit()
        self.results_path_edit.setReadOnly(True)
        self.results_path_button = QPushButton("Seleccionar")
        self.results_path_button.clicked.connect(lambda: self.select_directory(self.results_path_edit))
        paths_layout.addRow("Ruta de resultados:", self.results_path_edit)
        paths_layout.addRow("", self.results_path_button)

        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)

        # Grupo para parámetros del telescopio
        telescope_group = QGroupBox("Configuraciones de Telescopio")
        telescope_layout = QFormLayout()

        # ComboBox para cargar configuraciones
        self.config_combo = QComboBox()
        self.config_combo.currentIndexChanged.connect(self.load_selected_config)
        telescope_layout.addRow("Cargar configuración:", self.config_combo)

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

        # Botón para guardar configuración
        self.save_config_button = QPushButton("Guardar Configuración")
        self.save_config_button.clicked.connect(self.save_configuration)
        telescope_layout.addRow("", self.save_config_button)

        telescope_group.setLayout(telescope_layout)
        layout.addWidget(telescope_group)

        # Botones de aceptar y cancelar
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Aceptar")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Aplicar estilo
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

        # Cargar configuraciones disponibles
        self.load_configurations()
        # Cargar rutas desde config.json
        self.load_paths()

    def load_paths(self):
        """Carga las rutas desde el archivo config.json."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    if 'image_path' in config:
                        self.image_path_edit.setText(config['image_path'])
                    if 'results_path' in config:
                        self.results_path_edit.setText(config['results_path'])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error al cargar las rutas: {str(e)}")

    def select_directory(self, line_edit):
        """Abre un diálogo para seleccionar un directorio."""
        directory = QFileDialog.getExistingDirectory(self, "Seleccionar Directorio")
        if directory:
            line_edit.setText(directory)

    def load_configurations(self):
        """Carga las configuraciones disponibles en el directorio especificado."""
        if not os.path.exists(self.config_path):
            return

        self.config_combo.clear()
        self.config_combo.addItem("Nueva configuración")

        for file in os.listdir(self.config_path):
            if file.endswith('.json'):
                self.config_combo.addItem(file[:-5])  # Remove .json extension

    def load_selected_config(self, index):
        """Carga la configuración seleccionada del ComboBox."""
        if index == 0:  # "Nueva configuración"
            return

        config_name = self.config_combo.currentText()
        config_file = os.path.join(self.config_path, f"{config_name}.json")

        try:
            with open(config_file, 'r') as f:
                params = json.load(f)
                self.espejo_primario_edit.setText(str(params['espejo_primario']))
                self.espejo_secundario_edit.setText(str(params['espejo_secundario']))
                self.focal_edit.setText(str(params['focal']))
                self.apertura_edit.setText(str(params['apertura']))
                self.tamano_pixel_edit.setText(str(params['tamano_pixel']))
                self.binning_edit.setText(str(params['binning']))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error al cargar la configuración: {str(e)}")

    def save_configuration(self):
        """Guarda la configuración actual del telescopio."""
        name, ok = QInputDialog.getText(self, "Guardar configuración",
                                      "Nombre de la configuración:")
        if ok and name:
            if not name.strip():  # Verificar que el nombre no esté vacío
                QMessageBox.warning(self, "Error", "El nombre de la configuración no puede estar vacío.")
                return

            # Verificar si ya existe una configuración con ese nombre
            file_path = os.path.join(self.config_path, f"{name}.json")
            if os.path.exists(file_path):
                reply = QMessageBox.question(
                    self,
                    "Configuración existente",
                    f"Ya existe una configuración con el nombre '{name}'. ¿Desea sobrescribirla?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # Obtener los parámetros actuales
            params = self.get_telescope_params()
            if params is None:
                return  # La validación ya mostró el mensaje de error

            # Asegurarse de que el directorio existe
            os.makedirs(self.config_path, exist_ok=True)

            # Guardar en archivo
            try:
                with open(file_path, 'w') as f:
                    json.dump(params, f, indent=4)
                QMessageBox.information(self, "Éxito",
                                      "Configuración guardada correctamente.")

                # Desconectar temporalmente el evento para evitar que se cargue la configuración
                self.config_combo.currentIndexChanged.disconnect()

                # Actualizar la lista de configuraciones
                self.load_configurations()

                # Volver a conectar el evento
                self.config_combo.currentIndexChanged.connect(self.load_selected_config)

                # Seleccionar la configuración guardada
                index = self.config_combo.findText(name)
                if index >= 0:
                    self.config_combo.setCurrentIndex(index)
            except Exception as e:
                QMessageBox.warning(self, "Error",
                                  f"No se pudo guardar la configuración: {str(e)}")

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

    def get_config(self):
        """Retorna la configuración actual."""
        # Guardar las rutas en config.json
        try:
            config = {}
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

            config['image_path'] = self.image_path_edit.text()
            config['results_path'] = self.results_path_edit.text()

            # Crear el directorio si no existe
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error al guardar las rutas: {str(e)}")

        return {
            'image_path': self.image_path_edit.text(),
            'results_path': self.results_path_edit.text(),
            'config_path': self.config_path
        }