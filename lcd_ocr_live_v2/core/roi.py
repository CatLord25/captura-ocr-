"""ROIs nombradas, semántica de medición y persistencia JSON compatible.

El formato JSON conserva los seis campos del proyecto original
(``nombre, x, y, ancho, alto, instrumento``) y añade ``rol`` y ``unidad``
opcionales, de modo que un ``rois.json`` antiguo sigue cargando sin cambios.
"""

from dataclasses import asdict, dataclass
import json
import logging
from numbers import Integral
import os
from pathlib import Path
import tempfile

import numpy as np

import config
from core.utils import limitar_region, validar_imagen

LOG = logging.getLogger(__name__)
ROLES_VALIDOS = frozenset({"", *config.ROLES, "unidad", "texto"})
ROLES_MEDICION = frozenset(config.ROLES)


@dataclass(frozen=True)
class ROI:
    """Rectángulo en píxeles de la imagen procesada, con su rol de medición."""

    nombre: str
    x: int
    y: int
    ancho: int
    alto: int
    instrumento: str = "Balanza"
    rol: str = ""
    unidad: str = ""

    def __post_init__(self) -> None:
        """Valide nombre, instrumento, rol, unidad y geometría."""
        for campo in ("nombre", "instrumento", "unidad"):
            valor = getattr(self, campo)
            if not isinstance(valor, str):
                raise ValueError(f"El campo {campo} de una ROI debe ser texto.")
            object.__setattr__(self, campo, valor.strip())
        if not self.nombre:
            raise ValueError("La ROI necesita un nombre no vacío.")
        if not self.instrumento:
            raise ValueError("La ROI necesita un instrumento.")
        object.__setattr__(self, "instrumento", config.instrumento_canonico(self.instrumento))
        rol = self.rol.strip().casefold() if isinstance(self.rol, str) else self.rol
        if rol not in ROLES_VALIDOS:
            raise ValueError(
                f"Rol desconocido: {self.rol!r}. Use '', peso, temperatura, "
                "humedad, presion, otro, unidad o texto."
            )
        object.__setattr__(self, "rol", rol)
        if len(self.unidad) > 10:
            raise ValueError("La unidad no puede superar 10 caracteres.")
        for campo in ("x", "y", "ancho", "alto"):
            valor = getattr(self, campo)
            if isinstance(valor, bool) or not isinstance(valor, Integral):
                raise ValueError(f"El campo {campo} de una ROI debe ser un entero.")
            object.__setattr__(self, campo, int(valor))
        if self.x < 0 or self.y < 0 or self.ancho <= 0 or self.alto <= 0:
            raise ValueError("Una ROI debe tener origen no negativo y dimensiones positivas.")

    @property
    def clave(self) -> tuple[str, str]:
        """Clave única (instrumento, nombre) para cachés y votación temporal."""
        return self.instrumento, self.nombre


def recortar_roi(imagen: np.ndarray, roi: ROI) -> np.ndarray:
    """Devuelva una copia del recorte y rechace ROIs fuera de la imagen."""
    validar_imagen(imagen)
    if not isinstance(roi, ROI):
        raise TypeError("La región debe ser una instancia de ROI.")
    alto, ancho = imagen.shape[:2]
    if roi.x + roi.ancho > ancho or roi.y + roi.alto > alto:
        raise ValueError(
            f"La ROI '{roi.nombre}' excede la imagen de {ancho}×{alto}. "
            "Redefina las ROIs tras cambiar la captura, la resolución o la perspectiva."
        )
    return imagen[roi.y:roi.y + roi.alto, roi.x:roi.x + roi.ancho].copy()


def recortar_roi_seguro(
    imagen: np.ndarray, roi: ROI
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    """Recorte tolerante para vídeo en vivo: recorta a los límites del frame.

    Devuelve (recorte, caja_efectiva) o (None, None) si la ROI queda fuera.
    La caja efectiva se usa para dibujar el HUD con coordenadas reales.
    """
    validar_imagen(imagen)
    alto, ancho = imagen.shape[:2]
    caja = limitar_region(roi.x, roi.y, roi.ancho, roi.alto, ancho, alto)
    if caja is None:
        return None, None
    x, y, w, h = caja
    if (w, h) != (roi.ancho, roi.alto):
        LOG.debug("ROI %s recortada a %s×%s dentro del frame.", roi.nombre, w, h)
    return imagen[y:y + h, x:x + w].copy(), caja


def guardar_rois(rois: list[ROI], ruta=None) -> Path:
    """Guarde las ROIs como lista JSON mediante reemplazo atómico."""
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
        LOG.exception("No se pudieron guardar las ROIs en %s", destino)
        raise OSError(f"No se pudieron guardar las ROIs en {destino}: {error}") from error
    finally:
        if temporal is not None:
            temporal.unlink(missing_ok=True)
    LOG.info("Guardadas %d ROIs en %s", len(regiones), destino)
    return destino.resolve()


def cargar_rois(ruta=None) -> list[ROI]:
    """Cargue ROIs desde JSON; devuelva [] si no hay archivo o está vacío.

    Acepta el formato original sin ``rol``/``unidad`` y rechaza claves
    desconocidas para detectar archivos corruptos antes de operar.
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
        conocidos = {"nombre", "x", "y", "ancho", "alto", "instrumento", "rol", "unidad"}
        regiones = []
        for dato in datos:
            if not isinstance(dato, dict):
                raise ValueError("Cada ROI debe ser un objeto JSON.")
            extra = set(dato) - conocidos
            if extra:
                raise ValueError(f"Campos desconocidos en ROI: {sorted(extra)}.")
            regiones.append(ROI(**dato))
        _validar_lista(regiones)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Archivo de ROIs inválido en {origen}: {error}") from error
    LOG.info("Cargadas %d ROIs de %s", len(regiones), origen)
    return regiones


def rois_por_instrumento(rois: list[ROI], instrumento: str) -> list[ROI]:
    """Filtre las ROIs del instrumento indicado, conservando el orden."""
    canonico = config.instrumento_canonico(instrumento)
    return [roi for roi in rois if roi.instrumento == canonico]


def _validar_lista(rois: list[ROI]) -> None:
    """Rechace objetos inválidos y nombres duplicados por instrumento."""
    claves = set()
    for roi in rois:
        if not isinstance(roi, ROI):
            raise TypeError("Todas las regiones deben ser instancias de ROI.")
        if roi.clave in claves:
            raise ValueError(f"Nombre de ROI duplicado para {roi.instrumento}: {roi.nombre}.")
        claves.add(roi.clave)
