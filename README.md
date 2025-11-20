# PyRoddier - Análisis de Aberraciones Ópticas con el Test de Roddier

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)

**PyRoddier** es una aplicación de escritorio desarrollada para el análisis de aberraciones ópticas en telescopios mediante el **Test de Roddier**. Este software forma parte del **Trabajo de Fin de Máster del Máster de Astrofísica** desarrollado en colaboración con el **Instituto de Astrofísica de Andalucía (IAA-CSIC)**.

## 📋 Tabla de Contenidos

- [Acerca del Proyecto](#acerca-del-proyecto)
- [El Test de Roddier](#el-test-de-roddier)
- [Características Principales](#características-principales)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Descarga Rápida (Para Usuarios Finales)](#-descarga-rápida-para-usuarios-finales)
- [Instalación para Desarrollo](#-instalación-para-desarrollo)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Algoritmos Implementados](#algoritmos-implementados)
- [Tests y Cobertura](#tests-y-cobertura)
- [Documentación](#documentación)
- [Desarrollo](#desarrollo)
- [Licencia](#licencia)
- [Contacto](#contacto)

## 🔭 Acerca del Proyecto

**PyRoddier** es una herramienta especializada para la caracterización de aberraciones ópticas en sistemas telescópicos mediante el análisis de imágenes intra y extra-focales. La aplicación implementa algoritmos avanzados de procesamiento de imágenes astronómicas y descomposición en polinomios de Zernike para la evaluación cuantitativa de la calidad óptica.

### Contexto Académico
- **Proyecto**: Trabajo de Fin de Máster
- **Programa**: Máster en Astrofísica
- **Institución**: Universidad de Granada en colaboración con IAA-CSIC
- **Autor**: Adrián Hernández Padrón
- **Año**: 2025

## 🌟 El Test de Roddier

El **Test de Roddier** es una técnica de análisis de frente de onda que utiliza imágenes desenfocadas (intra y extra-focales) para determinar las aberraciones ópticas de un sistema telescópico. Esta técnica permite:

- **Medición directa del frente de onda** sin necesidad de sensores especializados
- **Caracterización completa de aberraciones** mediante descomposición en polinomios de Zernike
- **Evaluación de la calidad óptica** del sistema telescópico
- **Diagnóstico de problemas** en la óptica del telescopio

### Principio Físico

El método se basa en la relación entre las aberraciones del frente de onda y los patrones de intensidad observados en imágenes desenfocadas:

```
Δφ(x,y) = k * (I_extra(x,y) - I_intra(x,y)) / (I_extra(x,y) + I_intra(x,y))
```

Donde:
- `Δφ(x,y)` es la fase del frente de onda
- `I_extra` e `I_intra` son las intensidades extra e intra-focales
- `k` es una constante relacionada con la distancia de desenfoque

## ✨ Características Principales

### 🖥️ Interfaz Gráfica Avanzada
- **GUI intuitiva** desarrollada en PyQt5
- **Visualización en tiempo real** de resultados
- **Configuración flexible** de parámetros del telescopio
- **Exportación de resultados** en múltiples formatos

### 🔬 Análisis Científico Completo
- **Procesamiento de imágenes FITS** astronómicas
- **Alineación automática** de imágenes intra/extra-focales
- **Generación de máscaras anulares** adaptativas
- **Cálculo del frente de onda** mediante algoritmo de Roddier
- **Descomposición en polinomios de Zernike** hasta orden 23
- **Cálculo de RMS** del error de frente de onda
- **Generación de PSF teórica** y análisis de interferogramas

### 📊 Visualización y Análisis
- **Mapas 2D del frente de onda** con colormap personalizable
- **Histogramas de coeficientes de Zernike** con codificación por colores
- **Visualización de la PSF** en escala lineal y logarítmica
- **Análisis interactivo** con selección de modos de Zernike
- **Exportación de gráficos** en alta resolución

### ⚙️ Funcionalidades Técnicas
- **Configuración automática** de parámetros ópticos
- **Validación robusta** de datos de entrada
- **Manejo de errores** elegante y informativo
- **Procesamiento optimizado** para imágenes de gran tamaño
- **Guardado de configuraciones** y sesiones de trabajo

## 🛠️ Tecnologías Utilizadas

### Lenguajes y Frameworks
- **Python 3.11+** - Lenguaje principal
- **PyQt5** - Interfaz gráfica de usuario
- **NumPy** - Computación numérica
- **SciPy** - Algoritmos científicos
- **Matplotlib** - Visualización de datos

### Bibliotecas Especializadas
- **Astropy** - Manejo de datos astronómicos (FITS)
- **PIL (Pillow)** - Procesamiento de imágenes
- **PyInstaller** - Generación de ejecutables

### Algoritmos Implementados
- **Transformada de Fourier** para alineación de imágenes
- **Polinomios de Zernike** hasta orden 23
- **Mínimos cuadrados** para ajuste de coeficientes
- **Algoritmo de Roddier** para cálculo de frente de onda
- **Generación de PSF** mediante transformada de Fourier

## 📦 Descarga Rápida (Para Usuarios Finales)

**¿Solo quieres usar la aplicación?** No necesitas instalar Python ni dependencias.

👉 **[Descarga el ejecutable para tu sistema operativo](https://github.com/Adrihp06/PyRoddier/releases/latest)**

- 🪟 **Windows**: Descarga el ZIP, extrae y ejecuta
- 🍎 **macOS**: Descarga el ZIP, extrae y abre la aplicación
- 🐧 **Linux**: Descarga el TAR.GZ, extrae y ejecuta

> **Nota**: Las instrucciones de instalación manual siguientes son **solo para desarrolladores** que quieran modificar el código fuente.

---

## 🚀 Instalación para Desarrollo

### Requisitos del Sistema
- **Python 3.11 o superior**
- **Sistema operativo**:
  - ✅ **macOS 10.14+** (Totalmente soportado, incluyendo Apple Silicon)
  - ✅ **Linux Ubuntu 18+** (Totalmente soportado)
  - ⚠️ **Windows 10+** (Soportado con configuración especial - ver abajo)
- **RAM**: Mínimo 4GB, recomendado 8GB
- **Espacio en disco**: 500MB para instalación completa

> **Nota para usuarios de Windows**: Debido a limitaciones en PyQt5-Qt5, la instalación en Windows requiere
> pasos adicionales. Ver la sección [Instalación en Windows](#instalación-en-windows) más abajo.

### Instalación con uv (Recomendado) ⚡

**[uv](https://github.com/astral-sh/uv)** es un gestor de paquetes ultrarrápido para Python. Recomendamos este método por su velocidad y reproducibilidad.

1. **Instalar uv** (si no lo tienes):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. **Clonar el repositorio**:
```bash
git clone https://github.com/Adrihp06/PyRoddier.git
cd PyRoddier
```

3. **Sincronizar dependencias** (crea automáticamente el entorno virtual):

**macOS y Linux:**
```bash
uv sync --all-extras
```

**Windows:**
```powershell
# Instalación especial para Windows (requiere versiones específicas de PyQt5)
uv sync --all-extras --no-build
uv pip install --force-reinstall "PyQt5==5.15.9" "PyQt5-Qt5==5.15.2" "PyQt5-sip<13,>=12.11"
```

4. **Ejecutar la aplicación**:

**macOS y Linux:**
```bash
uv run python src/main.py
```

**Windows:**
```powershell
# Activar el entorno virtual y ejecutar
.venv\Scripts\python.exe src\main.py
```

### Instalación en Windows (Solo Desarrollo)

> ⚠️ **Usuarios finales**: Si solo quieres usar la aplicación, [descarga el ejecutable](https://github.com/Adrihp06/PyRoddier/releases/latest) en lugar de seguir estas instrucciones.

**Esta sección es solo para desarrolladores que quieran modificar el código en Windows.**

Debido a limitaciones en PyQt5-Qt5 (no hay wheels para Windows en versiones recientes),
los desarrolladores en Windows necesitan instalar versiones específicas.

#### Opción 1: Script Automático (Recomendado)

```powershell
# Clonar el repositorio
git clone https://github.com/Adrihp06/PyRoddier.git
cd PyRoddier

# Ejecutar script de instalación automática
.\install_windows.ps1

# El script instalará todo automáticamente
```

#### Opción 2: Instalación Manual

```powershell
# Después de clonar el repositorio
cd PyRoddier

# Instalar uv si no lo tienes
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Instalar dependencias con versiones compatibles con Windows
uv sync --all-extras --no-build
uv pip install --force-reinstall "PyQt5==5.15.9" "PyQt5-Qt5==5.15.2" "PyQt5-sip<13,>=12.11"

# Ejecutar (NO usar 'uv run' en Windows)
.venv\Scripts\python.exe src\main.py

# Para ejecutar tests
.venv\Scripts\python.exe -m pytest
```

> **⚠️ Importante**: En Windows, NO uses `uv run` después de la instalación manual de PyQt5.
> Usa directamente el Python del entorno virtual (`.venv\Scripts\python.exe`).

**Alternativa**: Usar WSL2 (Windows Subsystem for Linux) para una experiencia sin problemas.
Ver `docs/WINDOWS_INSTALL.md` para más detalles.

### Instalación Tradicional (pip)

1. **Clonar el repositorio**:
```bash
git clone https://github.com/Adrihp06/PyRoddier.git
cd PyRoddier
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Linux/macOS
# o
venv\Scripts\activate     # En Windows
```

3. **Instalar dependencias**:
```bash
pip install -e .
```

4. **Ejecutar la aplicación**:
```bash
python src/main.py
```

### Instalación de Ejecutable (Próximamente)

Se proporcionarán ejecutables precompilados para Windows, macOS y Linux a través de las releases de GitHub.

## 📖 Uso de la Aplicación

### 1. Cargar Imágenes
- **Intra-focal**: Imagen tomada antes del foco
- **Extra-focal**: Imagen tomada después del foco
- Formatos soportados: FITS, TIFF, PNG, JPG

### 2. Configurar Parámetros del Telescopio
- **Apertura**: Diámetro del espejo primario (mm)
- **Distancia focal**: Longitud focal del telescopio (mm)
- **Tamaño de píxel**: Tamaño físico del píxel del detector (μm)
- **Obstrucción central**: Diámetro del espejo secundario (mm)

### 3. Ejecutar el Test de Roddier
- Seleccionar región de interés (ROI)
- Configurar parámetros del algoritmo
- Ejecutar análisis completo

### 4. Analizar Resultados
- **Visualizar frente de onda** reconstruido
- **Examinar coeficientes de Zernike** individuales
- **Evaluar calidad óptica** mediante RMS
- **Exportar resultados** para análisis posterior

### Ejemplo de Uso Básico

```python
# Ejemplo de uso programático de los algoritmos core
from src.core.roddier import calculate_wavefront
from src.core.zernike import fit_zernike, calculate_rms
from src.core.optical_preprocessing import preprocess_roddier

# Preprocesar imágenes
delta_I, mask, center, R_out, dz = preprocess_roddier(
    intra_image, extra_image,
    apertura=900, focal=7200, pixel_scale=15
)

# Calcular frente de onda
wavefront = calculate_wavefront(delta_I, mask, dz_mm=dz)

# Ajustar polinomios de Zernike
coeffs, base = fit_zernike(wavefront, mask, R_out, center, max_order=23)

# Calcular RMS del error
rms_error = calculate_rms(coeffs, exclude_piston=True)
print(f"RMS del frente de onda: {rms_error:.4f} λ")
```

## 📁 Estructura del Proyecto

```
PyRoddier/
├── src/                          # Código fuente principal
│   ├── core/                     # Algoritmos científicos core
│   │   ├── roddier.py           # Implementación del algoritmo de Roddier
│   │   ├── zernike.py           # Polinomios de Zernike y ajuste
│   │   ├── optical_preprocessing.py  # Preprocesamiento de imágenes
│   │   ├── psf.py               # Cálculo de PSF
│   │   ├── interferometry.py    # Análisis de interferogramas
│   │   └── telescope.py         # Gestión de parámetros del telescopio
│   ├── gui/                     # Interfaz gráfica
│   │   ├── main_window.py       # Ventana principal
│   │   └── dialogs/             # Diálogos especializados
│   │       ├── roddiertest.py   # Diálogo de configuración del test
│   │       ├── roddiertestresults.py  # Visualización de resultados
│   │       └── config_dialog.py # Configuración general
│   ├── common/                  # Utilidades comunes
│   │   ├── utils.py            # Funciones de utilidad
│   │   └── config.py           # Gestión de configuración
│   └── main.py                 # Punto de entrada de la aplicación
├── tests/                      # Suite de tests
│   ├── core/                   # Tests de algoritmos core
│   ├── gui/                    # Tests de interfaz gráfica
│   └── utils/                  # Tests de utilidades
├── icons/                      # Iconos de la aplicación
├── images/                     # Imágenes de ejemplo y resultados
├── docs/                       # Documentación del proyecto
│   ├── Presentacion_PyRoddier.pdf    # Presentación del proyecto
│   └── TFM_Astrofisica.pdf           # Memoria completa del TFM
├── requirements.txt            # Dependencias de Python
├── pyroddier.spec             # Configuración de PyInstaller
└── README.md                  # Este archivo
```

## 🧮 Algoritmos Implementados

### 1. Algoritmo de Roddier
**Archivo**: `src/core/roddier.py`

Implementa el algoritmo core para el cálculo del frente de onda a partir de imágenes desenfocadas:

```python
def calculate_wavefront(delta_I_norm, annular_mask, dz_mm, wavelength=556):
    """
    Calcula el frente de onda usando el algoritmo de Roddier
    
    Args:
        delta_I_norm: Diferencia normalizada de intensidades
        annular_mask: Máscara anular de la pupila
        dz_mm: Distancia de desenfoque en mm
        wavelength: Longitud de onda en nm
    
    Returns:
        wavefront: Frente de onda en radianes
    """
```

### 2. Polinomios de Zernike
**Archivo**: `src/core/zernike.py`

Implementación completa de polinomios de Zernike hasta orden 23:

- **Generación de base ortonormal** sobre pupila anular
- **Ajuste por mínimos cuadrados** de coeficientes
- **Cálculo de RMS** excluyendo término de pistón
- **Reconstrucción del frente de onda** a partir de coeficientes

### 3. Preprocesamiento Óptico
**Archivo**: `src/core/optical_preprocessing.py`

Pipeline completo de preprocesamiento:

- **Alineación de imágenes** mediante correlación cruzada
- **Generación automática de máscaras** anulares
- **Estimación de parámetros** de desenfoque
- **Normalización y filtrado** de datos

### 4. Cálculo de PSF
**Archivo**: `src/core/psf.py`

Generación de función de dispersión puntual:

```python
def calculate_psf(wavefront, pupila_mask, wavelength=556):
    """
    Calcula la PSF a partir del frente de onda
    
    Returns:
        PSF: Función de dispersión puntual normalizada
        PSF_log: PSF en escala logarítmica
    """
```

## 🔬 Tests y Cobertura

El proyecto cuenta con una suite comprehensiva de tests unitarios y de integración:

### Cobertura de Tests
- **Módulos Core**: ~90% de cobertura
- **Algoritmos**: Tests exhaustivos con casos edge
- **GUI**: Tests de funcionalidad básica
- **Integración**: Tests end-to-end del pipeline completo

### Ejecutar Tests

```bash
# Ejecutar todos los tests
python -m unittest discover -s tests

# Ejecutar tests específicos
python -m unittest tests.core.test_roddier_calculations
python -m unittest tests.core.test_zernike_complete

# Generar reporte de cobertura (si coverage está instalado)
coverage run -m unittest discover -s tests
coverage report -m
coverage html  # Genera reporte HTML
```

### Tipos de Tests Implementados

1. **Tests Unitarios**: Verificación de funciones individuales
2. **Tests de Integración**: Validación del pipeline completo
3. **Tests de Robustez**: Manejo de casos edge y errores
4. **Tests de Performance**: Verificación de tiempo de ejecución
5. **Tests de GUI**: Validación básica de la interfaz

## 📚 Documentación

### Documentos Académicos
- **`docs/TFM_Astrofisica.pdf`**: Memoria completa del Trabajo de Fin de Máster
- **`docs/Presentacion_PyRoddier.pdf`**: Presentación del proyecto

### Documentación Técnica
- **Docstrings**: Documentación completa en código fuente
- **Comments**: Explicaciones detalladas de algoritmos complejos
- **README**: Guía de usuario y desarrollador (este documento)

### Referencias Científicas

El proyecto se basa en las siguientes publicaciones científicas:

1. **Roddier, F.** (1988). "Curvature sensing and compensation: a new concept in adaptive optics"
2. **Noll, R.J.** (1976). "Zernike polynomials and atmospheric turbulence"
3. **Malacara, D.** (2007). "Optical Shop Testing"

## 🔧 Desarrollo

### Configuración del Entorno de Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install coverage pytest black flake8

# Configurar pre-commit hooks (opcional)
pre-commit install
```

### Estándares de Código
- **Style Guide**: PEP 8
- **Docstrings**: Google Style
- **Type Hints**: Recomendado para funciones core
- **Testing**: Mínimo 80% de cobertura para nuevo código

### Contribuir al Proyecto

1. **Fork** el repositorio
2. **Crear rama** de feature (`git checkout -b feature/nueva-funcionalidad`)
3. **Implementar cambios** siguiendo estándares de código
4. **Añadir tests** para nueva funcionalidad
5. **Verificar** que todos los tests pasan
6. **Commit** con mensaje descriptivo
7. **Push** a la rama (`git push origin feature/nueva-funcionalidad`)
8. **Abrir Pull Request**

### Roadmap de Desarrollo

- [ ] **v2.0**: Soporte para múltiples longitudes de onda
- [ ] **v2.1**: Análisis de aberraciones cromáticas
- [ ] **v2.2**: Integración con sistemas de óptica adaptativa
- [ ] **v2.3**: API REST para procesamiento remoto
- [ ] **v3.0**: Soporte para telescopios segmentados

## 📊 Rendimiento y Especificaciones

### Benchmarks Típicos
- **Imágenes 512x512**: ~2-5 segundos de procesamiento
- **Imágenes 1024x1024**: ~8-15 segundos de procesamiento
- **Imágenes 2048x2048**: ~30-60 segundos de procesamiento

### Limitaciones Conocidas
- **Tamaño máximo de imagen**: Limitado por RAM disponible
- **Precisión numérica**: Dependiente de la calidad de las imágenes de entrada
- **Longitud de onda**: Optimizado para banda V (556 nm)

## 📄 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Adrián Hernández Padrón

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 📧 Contacto y Soporte

### Autor
**Adrián Hernández Padrón**
- 📧 Email: adrianhdezp10@gmail.com
- 🐙 GitHub: [@Adrihp06](https://github.com/Adrihp06)
- 🔗 LinkedIn: [Adrián Hernández Padrón](https://linkedin.com/in/adrianhernandezpadron)

### Institución
**Instituto de Astrofísica de Andalucía (IAA-CSIC)**
- 🌐 Web: [www.iaa.es](https://www.iaa.es)
- 📍 Dirección: Glorieta de la Astronomía s/n, 18008 Granada, España

### Soporte
- **Issues**: [GitHub Issues](https://github.com/Adrihp06/PyRoddier/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/Adrihp06/PyRoddier/discussions)
- **Email**: Para consultas específicas o colaboraciones académicas

---

## 🏆 Agradecimientos

Especial agradecimiento a:
- **Instituto de Astrofísica de Andalucía (IAA-CSIC)** por el apoyo institucional
- **Máster de Astrofísica de la Universidad de Granada** por el marco académico
- **Comunidad científica** por las referencias y algoritmos base
- **Desarrolladores de código abierto** por las herramientas utilizadas

---

<div align="center">

**PyRoddier** - Análisis Profesional de Aberraciones Ópticas para Astronomía

[⭐ Star](https://github.com/Adrihp06/PyRoddier) | [🐛 Report Bug](https://github.com/Adrihp06/PyRoddier/issues) | [✨ Request Feature](https://github.com/Adrihp06/PyRoddier/issues)

</div>
