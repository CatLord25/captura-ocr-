"""Rectificación del LCD a partir de cuatro esquinas de la imagen original."""

import logging

import cv2
import numpy as np

from core.utils import validar_imagen

LOGGER = logging.getLogger(__name__)


def ordenar_puntos(puntos) -> np.ndarray:
    """Ordene cuatro esquinas convexas como superior izq., der., inferior der., izq.

    Las coordenadas deben estar en píxeles de la imagen original. Rechace
    puntos repetidos, colineales y selecciones donde una esquina esté dentro
    del triángulo formado por las otras tres.
    """
    try:
        esquinas = np.asarray(puntos, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("Las esquinas deben ser cuatro pares de coordenadas.") from error
    if esquinas.shape != (4, 2) or not np.isfinite(esquinas).all():
        raise ValueError("Seleccione exactamente cuatro esquinas válidas del LCD.")
    if len(np.unique(esquinas, axis=0)) != 4:
        raise ValueError("Las cuatro esquinas deben ser distintas.")
    envolvente = cv2.convexHull(esquinas)
    if len(envolvente) != 4 or cv2.contourArea(envolvente) < 4:
        raise ValueError("Las esquinas deben formar un cuadrilátero convexo con área.")
    centro = esquinas.mean(axis=0)
    angulos = np.arctan2(esquinas[:, 1] - centro[1], esquinas[:, 0] - centro[0])
    ordenadas = esquinas[np.argsort(angulos)]
    inicio = min(range(4), key=lambda i: (ordenadas[i].sum(), ordenadas[i, 1]))
    return np.roll(ordenadas, -inicio, axis=0).astype(np.float32)


def corregir_perspectiva(imagen: np.ndarray, puntos, tamano_salida=None) -> np.ndarray:
    """Enderece el LCD mediante una homografía, conservando RGB o escala de grises.

    Acepte las cuatro esquinas en cualquier orden. Deduzca el tamaño de salida
    de las distancias entre lados, o use ``tamano_salida=(ancho, alto)``.
    Lance ValueError si los puntos salen de la imagen o el resultado no es válido.
    """
    validar_imagen(imagen)
    esquinas = ordenar_puntos(puntos)
    alto_origen, ancho_origen = imagen.shape[:2]
    if (esquinas < 0).any() or (esquinas[:, 0] > ancho_origen - 1).any() or (
        esquinas[:, 1] > alto_origen - 1
    ).any():
        raise ValueError("Las esquinas deben quedar dentro de la imagen original.")
    superior_izq, superior_der, inferior_der, inferior_izq = esquinas
    if tamano_salida is None:
        ancho = round(max(np.linalg.norm(superior_der - superior_izq),
                          np.linalg.norm(inferior_der - inferior_izq)))
        alto = round(max(np.linalg.norm(inferior_izq - superior_izq),
                         np.linalg.norm(inferior_der - superior_der)))
    else:
        if len(tamano_salida) != 2 or any(
            not isinstance(valor, (int, np.integer)) for valor in tamano_salida
        ):
            raise ValueError("El tamaño de salida debe ser (ancho, alto) en enteros.")
        ancho, alto = map(int, tamano_salida)
    if ancho < 2 or alto < 2 or ancho * alto > 25_000_000:
        raise ValueError("El LCD rectificado debe medir al menos 2×2 y hasta 25 MP.")
    destino = np.array([[0, 0], [ancho - 1, 0],
                        [ancho - 1, alto - 1], [0, alto - 1]], dtype=np.float32)
    matriz = cv2.getPerspectiveTransform(esquinas, destino)
    corregida = cv2.warpPerspective(imagen, matriz, (ancho, alto), flags=cv2.INTER_LINEAR)
    LOGGER.info("Perspectiva corregida: %d×%d píxeles", ancho, alto)
    return corregida
