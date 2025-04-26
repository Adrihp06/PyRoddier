# Copyright (c) 2025 Adrián Hernández Padrón
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
import numpy as np

def calculate_interferogram(wavefront, reference_frequency, reference_intensity, annular_mask):
    """
    Calcula interferograma simulando exactamente la metodología de WinRoddier 3.0.

    Parámetros:
    - wavefront: array (NxN), en unidades de longitud de onda (lambda)
    - reference_frequency: frecuencia espacial (ciclos/pupila)
    - reference_intensity: intensidad relativa del haz de referencia
    - annular_mask: máscara binaria (0 o 1) para la pupila (NxN)

    Retorna:
    - interferograma (NxN)
    """
    N = wavefront.shape[0]
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)

    # Conversión del frente de onda de unidades lambda a fase en radianes
    fase_wavefront = 2 * np.pi * wavefront

    # Aplicar tilt lineal en X (como WinRoddier: franjas en dirección X)
    tilt = 2 * np.pi * reference_frequency * X

    # Fase total (frente de onda aberrado + referencia inclinada)
    fase_total = fase_wavefront + tilt

    # Campos coherentes:
    campo_prueba = np.exp(1j * fase_total)
    campo_referencia = np.sqrt(reference_intensity)  # Amplitud raíz de intensidad

    # Suma de campos coherentes:
    campo_total = campo_prueba + campo_referencia

    # Intensidad final del interferograma (módulo al cuadrado)
    interferograma = np.abs(campo_total)**2

    # Aplicación de la máscara (fuera de la pupila intensidad cero)
    interferograma *= annular_mask

    interferograma -= interferograma.min()
    interferograma /= interferograma.max()

    return interferograma
