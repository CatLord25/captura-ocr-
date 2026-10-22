"""Utilidades compartidas: imágenes BGR, iluminación, FPS y colores de HUD.

Contrato de color: todo el proyecto trabaja en **BGR** (orden nativo de
OpenCV) para que ``cv2.imshow``, ``cv2.imencode`` y las máscaras compartan
convención. Las utilidades de Tk convierten a RGB al vuelo.
"""

from collections import deque
from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

import config


def configurar_logging() -> None:
    """Configure consola y reduzca los hilos internos de OpenCV."""
    logging.basicConfig(
        level=config.NIVEL_LOG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        cv2.setNumThreads(int(config.NUM_HILOS_OPENCV))
        cv2.setUseOptimized(True)
    except cv2.error:  # pragma: no cover - depende del build de OpenCV
        logging.getLogger(__name__).warning("OpenCV no aceptó la configuración de hilos.")


def validar_imagen(imagen: np.ndarray) -> np.ndarray:
    """Valide un array uint8 gris, BGR o BGRA y devuelva el mismo array."""
    if not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La imagen está vacía o no es un numpy.ndarray.")
    if imagen.dtype != np.uint8:
        raise ValueError("La imagen debe tener valores uint8 entre 0 y 255.")
    if imagen.ndim == 3 and imagen.shape[2] not in (3, 4):
        raise ValueError("La imagen debe tener 1, 3 (BGR) o 4 (BGRA) canales.")
    if imagen.ndim not in (2, 3):
        raise ValueError("La imagen debe ser bidimensional o BGR/BGRA.")
    return imagen


def a_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierta BGR/BGRA a gris sin modificar la entrada."""
    validar_imagen(imagen)
    if imagen.ndim == 2:
        return imagen.copy()
    codigo = cv2.COLOR_BGR2GRAY if imagen.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
    return cv2.cvtColor(imagen, codigo)


def cargar_imagen(ruta: str | Path) -> np.ndarray:
    """Lea una imagen local como BGR admitiendo rutas Unicode/acentos."""
    archivo = Path(ruta).expanduser()
    try:
        datos = np.frombuffer(archivo.read_bytes(), dtype=np.uint8)
    except OSError as error:
        raise OSError(f"No se pudo leer la imagen {archivo}: {error}") from error
    imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    if imagen is None:
        raise ValueError(f"El archivo no contiene una imagen válida: {archivo}")
    return imagen


def guardar_imagen(imagen: np.ndarray, ruta: str | Path) -> Path:
    """Guarde BGR/gris de forma atómica y devuelva la ruta resuelta."""
    validar_imagen(imagen)
    archivo = Path(ruta).expanduser()
    extension = archivo.suffix.casefold()
    if extension not in config.EXTENSIONES_IMAGEN:
        raise ValueError(f"Formato de imagen no admitido: {extension}")
    parametros = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in {".jpg", ".jpeg"} else []
    correcto, codificada = cv2.imencode(extension, imagen, parametros)
    if not correcto:
        raise OSError(f"No se pudo codificar la imagen para {archivo}.")
    archivo.parent.mkdir(parents=True, exist_ok=True)
    temporal = None
    try:
        with tempfile.NamedTemporaryFile(dir=archivo.parent, delete=False) as salida:
            temporal = Path(salida.name)
            salida.write(codificada.tobytes())
        os.replace(temporal, archivo)
    finally:
        if temporal is not None and temporal.exists():
            temporal.unlink()
    return archivo.resolve()


def normalizar_iluminacion(
    gris: np.ndarray,
    activo: bool = config.CLAHE_ACTIVO,
    clip: float = config.CLAHE_CLIP,
    tile: tuple[int, int] = config.CLAHE_TILE,
) -> np.ndarray:
    """Compense reflejos y viñeteo con CLAHE sobre la luminancia.

    Aplique CLAHE en el canal L de LAB cuando la entrada es BGR, y directamente
    sobre gris cuando ya es monocroma. Un ``clip`` bajo evita amplificar ruido
    del sensor OV5647. Devuelve uint8 de la misma forma que la entrada.
    """
    validar_imagen(gris)
    if not activo:
        return gris.copy() if gris.ndim == 2 else a_grises(gris)
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("CLAHE_CLIP debe ser positivo y finito.")
    ancho_tile, alto_tile = (int(valor) for valor in tile)
    if ancho_tile < 1 or alto_tile < 1:
        raise ValueError("CLAHE_TILE debe contener dos enteros positivos.")
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(ancho_tile, alto_tile))
    if gris.ndim == 3:
        laboratorio = cv2.cvtColor(gris, cv2.COLOR_BGR2LAB)
        luminancia, canal_a, canal_b = cv2.split(laboratorio)
        laboratorio = cv2.merge((clahe.apply(luminancia), canal_a, canal_b))
        return cv2.cvtColor(laboratorio, cv2.COLOR_LAB2BGR)
    return clahe.apply(gris)


def limitar_region(
    x: int, y: int, ancho: int, alto: int, ancho_imagen: int, alto_imagen: int
) -> tuple[int, int, int, int] | None:
    """Recorte una caja a los límites de la imagen; None si queda vacía."""
    x1 = max(0, min(int(x), int(ancho_imagen)))
    y1 = max(0, min(int(y), int(alto_imagen)))
    x2 = max(0, min(int(x) + int(ancho), int(ancho_imagen)))
    y2 = max(0, min(int(y) + int(alto), int(alto_imagen)))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def color_confianza(confianza: float) -> tuple[tuple[int, int, int], str]:
    """Devuelva (BGR, hex Tk) semafórico según la confianza 0..1."""
    if not np.isfinite(confianza):
        confianza = 0.0
    if confianza >= config.UMBRAL_CONFIANZA_VERDE:
        return config.HUD_COLOR_VERDE, "#5cdc78"
    if confianza >= config.UMBRAL_CONFIANZA_AMARILLO:
        return config.HUD_COLOR_AMARILLO, "#fac83c"
    return config.HUD_COLOR_ROJO, "#f05050"


def ahora_iso() -> str:
    """Fecha y hora local con zona, en ISO 8601 con segundos."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


class MedidorFPS:
    """Calcule FPS suavizados con una ventana deslizante de marcas."""

    def __init__(self, ventana: int = 30) -> None:
        """Cree el medidor con ``ventana`` marcas de tiempo."""
        if ventana < 2:
            raise ValueError("La ventana de FPS debe ser al menos 2.")
        self._marcas: deque[float] = deque(maxlen=int(ventana))

    def marcar(self) -> float:
        """Registre un frame y devuelva los FPS actuales."""
        ahora = time.perf_counter()
        self._marcas.append(ahora)
        return self.fps

    @property
    def fps(self) -> float:
        """FPS medios de la ventana; 0 si aún no hay datos suficientes."""
        if len(self._marcas) < 2:
            return 0.0
        intervalo = self._marcas[-1] - self._marcas[0]
        if intervalo <= 0:
            return 0.0
        return (len(self._marcas) - 1) / intervalo


class Latencia:
    """Mida la latencia de extremo a extremo de una lectura de cámara."""

    def __init__(self, ventana: int = 60) -> None:
        """Cree el medidor con una ventana de muestras."""
        self._muestras: deque[float] = deque(maxlen=int(ventana))

    def registrar(self, milisegundos: float) -> float:
        """Añada una medición y devuelva la mediana en ms."""
        if np.isfinite(milisegundos) and milisegundos >= 0:
            self._muestras.append(float(milisegundos))
        return self.ms

    @property
    def ms(self) -> float:
        """Mediana de latencia en milisegundos (0 si no hay muestras)."""
        if not self._muestras:
            return 0.0
        return float(np.median(self._muestras))
