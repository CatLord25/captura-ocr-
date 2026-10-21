"""Utilidades compartidas por la aplicación."""

import logging
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

import config


def configurar_logging() -> None:
    """Configure los mensajes de consola con fecha, nivel y nombre del módulo."""
    logging.basicConfig(
        level=config.NIVEL_LOG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    cv2.setNumThreads(config.NUM_HILOS_OPENCV)


def validar_imagen(imagen: np.ndarray) -> np.ndarray:
    """Valide una imagen uint8, gris o RGB/RGBA, y devuelva el mismo array."""
    if not isinstance(imagen, np.ndarray) or imagen.size == 0:
        raise ValueError("La imagen está vacía o no es un numpy.ndarray.")
    if imagen.dtype != np.uint8:
        raise ValueError("La imagen debe tener valores uint8 entre 0 y 255.")
    if imagen.ndim != 2 and not (
        imagen.ndim == 3 and imagen.shape[2] in (3, 4)
    ):
        raise ValueError("La imagen debe estar en escala de grises, RGB o RGBA.")
    return imagen


def a_grises(imagen: np.ndarray) -> np.ndarray:
    """Convierta una imagen RGB/RGBA a gris sin modificar la entrada."""
    validar_imagen(imagen)
    if imagen.ndim == 2:
        return imagen.copy()
    codigo = cv2.COLOR_RGB2GRAY if imagen.shape[2] == 3 else cv2.COLOR_RGBA2GRAY
    return cv2.cvtColor(imagen, codigo)


def cargar_imagen(ruta: str | Path) -> np.ndarray:
    """Lea una imagen local y devuelva RGB, admitiendo rutas con acentos."""
    archivo = Path(ruta).expanduser()
    try:
        datos = np.frombuffer(archivo.read_bytes(), dtype=np.uint8)
        imagen = cv2.imdecode(datos, cv2.IMREAD_COLOR) if datos.size else None
    except OSError as error:
        raise OSError(f"No se pudo leer la imagen {archivo}: {error}") from error
    if imagen is None:
        raise ValueError(f"El archivo no contiene una imagen válida: {archivo}")
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)


def guardar_imagen(imagen: np.ndarray, ruta: str | Path) -> Path:
    """Guarde RGB o gris de forma atómica y devuelva la ruta del archivo."""
    validar_imagen(imagen)
    archivo = Path(ruta).expanduser()
    extension = archivo.suffix.casefold()
    if extension not in config.EXTENSIONES_IMAGEN:
        raise ValueError(f"Formato de imagen no admitido: {extension}")
    datos = imagen
    if imagen.ndim == 3:
        codigo = cv2.COLOR_RGB2BGR if imagen.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
        datos = cv2.cvtColor(imagen, codigo)
    parametros = [cv2.IMWRITE_JPEG_QUALITY, 95] if extension in {".jpg", ".jpeg"} else []
    correcto, codificada = cv2.imencode(extension, datos, parametros)
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
