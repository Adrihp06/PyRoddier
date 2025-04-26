from dataclasses import dataclass
import json
import os
from typing import Optional

@dataclass
class TelescopeParams:
    """Clase para gestionar los parámetros del telescopio."""
    apertura: float  # en mm
    focal: float  # en mm
    pixel_scale: float  # en arcsec/pixel
    max_order: int = 23  # orden máximo de Zernike por defecto
    threshold: float = 0.5  # threshold por defecto
    binning: int = 1  # binning por defecto

    @classmethod
    def from_dict(cls, data: dict) -> 'TelescopeParams':
        """Crea una instancia de TelescopeParams desde un diccionario."""
        return cls(
            apertura=data.get('apertura', 0.0),
            focal=data.get('focal', 0.0),
            pixel_scale=data.get('pixel_scale', 0.0),
            max_order=data.get('max_order', 23),
            threshold=data.get('threshold', 0.5),
            binning=data.get('binning', 1)
        )

    @classmethod
    def from_json(cls, file_path: str) -> Optional['TelescopeParams']:
        """Crea una instancia de TelescopeParams desde un archivo JSON."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return cls.from_dict(data)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error al cargar configuración: {e}")
            return None

    def to_dict(self) -> dict:
        """Convierte la instancia a un diccionario."""
        return {
            'apertura': self.apertura,
            'focal': self.focal,
            'pixel_scale': self.pixel_scale,
            'max_order': self.max_order,
            'threshold': self.threshold,
            'binning': self.binning
        }

    def save_to_json(self, file_path: str) -> bool:
        """Guarda la configuración en un archivo JSON."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            return True
        except Exception as e:
            print(f"Error al guardar configuración: {e}")
            return False