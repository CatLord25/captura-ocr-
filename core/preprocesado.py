"""Binarización y operaciones morfológicas independientes de la interfaz."""

import unicodedata

import cv2
import numpy as np

import config
from core.utils import a_grises


def _nombre(valor: str) -> str:
    if not isinstance(valor, str):
        raise ValueError("El método y la operación deben ser nombres de texto.")
    return "".join(c for c in unicodedata.normalize("NFKD", valor.strip().casefold())
                   if not unicodedata.combining(c))


def filtrar_imagen(
    imagen: np.ndarray,
    threshold: int = config.THRESHOLD_BINARIO,
    kernel: int = config.KERNEL_MORFOLOGICO,
    operacion: str = "ninguna",
    invertir: bool = config.INVERTIR_BINARIO,
    metodo: str = "manual",
    iteraciones: int = 1,
) -> np.ndarray:
    """Binarice una imagen RGB/gris y aplique una operación morfológica.

    Admita métodos manual, otsu y adaptativo; operaciones ninguna, erosion,
    dilatacion, apertura y cierre. El resultado es uint8 con valores 0/255.
    En OpenCV la morfología actúa sobre regiones blancas: invertir permite
    seleccionar los dígitos oscuros como primer plano antes de esa operación.
    """
    gris = a_grises(imagen)
    if not isinstance(threshold, (int, np.integer)) or not 0 <= threshold <= 255:
        raise ValueError("El threshold debe ser un entero entre 0 y 255.")
    if not isinstance(kernel, (int, np.integer)) or not 1 <= kernel <= 51:
        raise ValueError("El tamaño de kernel debe ser un entero entre 1 y 51.")
    if not isinstance(iteraciones, (int, np.integer)) or not 1 <= iteraciones <= 20:
        raise ValueError("Las iteraciones deben ser un entero entre 1 y 20.")
    metodo = _nombre(metodo)
    operacion = _nombre(operacion)
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    if metodo == "manual":
        _, binaria = cv2.threshold(gris, int(threshold), 255, tipo)
    elif metodo == "otsu":
        _, binaria = cv2.threshold(gris, 0, 255, tipo | cv2.THRESH_OTSU)
    elif metodo == "adaptativo":
        bloque = max(3, int(getattr(config, "BLOQUE_ADAPTATIVO", 31)))
        bloque += (bloque % 2 == 0)
        constante = float(getattr(config, "CONSTANTE_ADAPTATIVA", 7))
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       tipo, bloque, constante)
    else:
        raise ValueError("Método desconocido: use manual, otsu o adaptativo.")
    operaciones = {"erosion": cv2.MORPH_ERODE, "dilatacion": cv2.MORPH_DILATE,
                   "apertura": cv2.MORPH_OPEN, "cierre": cv2.MORPH_CLOSE}
    if operacion == "ninguna":
        return binaria
    if operacion not in operaciones:
        raise ValueError("Operación desconocida: use ninguna, erosion, dilatacion, apertura o cierre.")
    elemento = cv2.getStructuringElement(cv2.MORPH_RECT, (int(kernel), int(kernel)))
    return cv2.morphologyEx(binaria, operaciones[operacion], elemento,
                            iterations=int(iteraciones))
