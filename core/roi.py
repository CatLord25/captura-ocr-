"""ROIs nombradas y persistencia JSON de sus coordenadas en la imagen."""

from dataclasses import asdict, dataclass
import json
import logging
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np

import config
from core.utils import validar_imagen

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ROI:
    """Represente un rectángulo en píxeles de la imagen procesada, por instrumento."""

    nombre: str
    x: int
    y: int
    ancho: int
    alto: int
    instrumento: str = "Balanza"

    def __post_init__(self) -> None:
        """Valide nombres y dimensiones al crear la región."""
        for campo in ("nombre", "instrumento"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"El campo {campo} de una ROI debe ser texto no vacío.")
            object.__setattr__(self, campo, valor.strip())
        for campo in ("x", "y", "ancho", "alto"):
            valor = getattr(self, campo)
            if isinstance(valor, bool) or not isinstance(valor, Integral):
                raise ValueError(f"El campo {campo} de una ROI debe ser un entero.")
            object.__setattr__(self, campo, int(valor))
        if self.x < 0 or self.y < 0 or self.ancho <= 0 or self.alto <= 0:
            raise ValueError("Una ROI debe tener origen no negativo y dimensiones positivas.")


def recortar_roi(imagen: np.ndarray, roi: ROI) -> np.ndarray:
    """Devuelva una copia del recorte y rechace ROIs fuera de la imagen actual."""
    validar_imagen(imagen)
    if not isinstance(roi, ROI):
        raise TypeError("La región debe ser una instancia de ROI.")
    alto, ancho = imagen.shape[:2]
    if roi.x + roi.ancho > ancho or roi.y + roi.alto > alto:
        raise ValueError(
            f"La ROI '{roi.nombre}' excede la imagen de {ancho}×{alto}. "
            "Redefina las ROIs después de cambiar la captura o la perspectiva."
        )
    return imagen[roi.y:roi.y + roi.alto, roi.x:roi.x + roi.ancho].copy()


def guardar_rois(rois: list[ROI], ruta=None) -> None:
    """Guarde las ROIs como lista JSON mediante reemplazo atómico del archivo."""
    destino = Path(ruta if ruta is not None else config.ARCHIVO_ROIS).expanduser()
    regiones = list(rois)
    _validar_lista(regiones)
    temporal = None
    try:
        destino.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=destino.parent, delete=False) as archivo:
            temporal = Path(archivo.name)
            json.dump([asdict(roi) for roi in regiones], archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
        os.replace(temporal, destino)
    except OSError as error:
        LOGGER.exception("No se pudieron guardar las ROIs en %s", destino)
        raise OSError(f"No se pudieron guardar las ROIs en {destino}: {error}") from error
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
    LOGGER.info("Guardadas %d ROIs en %s", len(regiones), destino)


def cargar_rois(ruta=None) -> list[ROI]:
    """Cargue ROIs desde una lista JSON; devuelva [] si no hay archivo o está vacío.

    Rechace contenido corrupto, campos inesperados y nombres duplicados para
    el mismo instrumento. La validación contra una imagen ocurre al recortar.
    """
    origen = Path(ruta if ruta is not None else config.ARCHIVO_ROIS).expanduser()
    try:
        contenido = origen.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return []
    except OSError as error:
        raise OSError(f"No se pudieron leer las ROIs de {origen}: {error}") from error
    if not contenido.strip():
        return []
    try:
        datos = json.loads(contenido)
        if not isinstance(datos, list):
            raise ValueError("El JSON debe contener una lista de ROIs.")
        regiones = [ROI(**dato) for dato in datos]
        _validar_lista(regiones)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Archivo de ROIs inválido en {origen}: {error}") from error
    LOGGER.info("Cargadas %d ROIs de %s", len(regiones), origen)
    return regiones


def _validar_lista(rois: list[ROI]) -> None:
    claves = set()
    for roi in rois:
        if not isinstance(roi, ROI):
            raise TypeError("Todas las regiones deben ser instancias de ROI.")
        clave = (roi.instrumento.casefold(), roi.nombre.casefold())
        if clave in claves:
            raise ValueError(f"Nombre de ROI duplicado para {roi.instrumento}: {roi.nombre}.")
        claves.add(clave)
