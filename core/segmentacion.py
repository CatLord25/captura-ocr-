"""Segmentación de una línea LCD sin separar las piezas de siete segmentos."""

from dataclasses import dataclass
import logging
import unicodedata

import cv2
import numpy as np

import config
from core.utils import a_grises

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segmento:
    """Contenga un recorte de carácter y su caja relativa a la ROI original."""

    imagen: np.ndarray
    x: int
    y: int
    ancho: int
    alto: int


def mascara_primer_plano(imagen: np.ndarray) -> np.ndarray:
    """Obtenga una máscara uint8 con tinta blanca y fondo negro automáticamente.

    Use Otsu si la entrada no está binarizada e infiera la polaridad a partir
    del borde. Devuelva una máscara vacía para imágenes de intensidad uniforme.
    """
    gris = a_grises(imagen)
    if gris.min() == gris.max():
        return np.zeros_like(gris)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    borde = np.concatenate((binaria[0], binaria[-1], binaria[:, 0], binaria[:, -1]))
    blancos = np.count_nonzero(borde) / borde.size
    if blancos > 0.5 or (blancos == 0.5 and np.count_nonzero(binaria) > binaria.size / 2):
        binaria = cv2.bitwise_not(binaria)
    return binaria


def segmentar_caracteres(
    binaria: np.ndarray,
    metodo: str = "contornos",
    area_minima: int | None = None,
    max_hueco: int | None = None,
) -> list[Segmento]:
    """Separe una línea por contornos/proyección, o devuelva la ROI completa.

    Agrupe componentes que se solapen horizontalmente para conservar las
    barras desconectadas de un dígito LCD. Puentee huecos pequeños entre barras,
    y conserve puntos/comas inferiores como caracteres independientes. Los
    recortes devueltos usan tinta blanca sobre negro y dos píxeles de margen,
    sin deformar proporciones. La caja conserva el tamaño original sin margen.
    ``area_minima`` y ``max_hueco`` permiten ajustar ruido y separación en píxeles.
    ``metodo='ROI completa'`` compara un símbolo compuesto, unidad o texto
    entero con su plantilla. Las ROIs numéricas de varias líneas deben dividirse.
    """
    if not isinstance(metodo, str):
        raise ValueError("El método debe ser contornos, proyeccion o ROI completa.")
    metodo = "".join(c for c in unicodedata.normalize("NFKD", metodo.casefold().strip())
                     if not unicodedata.combining(c))
    if metodo not in {"contornos", "proyeccion", "roi completa", "roi_completa"}:
        raise ValueError("Método desconocido: use contornos, proyeccion o ROI completa.")
    area_minima = int(getattr(config, "SEGMENTACION_AREA_MINIMA", 2)) if area_minima is None else area_minima
    if not isinstance(area_minima, (int, np.integer)) or area_minima < 1:
        raise ValueError("El área mínima debe ser un entero positivo.")
    mascara = mascara_primer_plano(binaria)
    if metodo in {"roi completa", "roi_completa"}:
        if cv2.countNonZero(mascara) < area_minima:
            return []
        alto, ancho = mascara.shape
        recorte = cv2.copyMakeBorder(mascara, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
        return [Segmento(recorte, 0, 0, ancho, alto)]
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas = []
    for contorno in contornos:
        x, y, ancho, alto = cv2.boundingRect(contorno)
        if cv2.countNonZero(mascara[y:y + alto, x:x + ancho]) >= area_minima:
            cajas.append((x, y, ancho, alto))
    if not cajas:
        LOGGER.info("No hay caracteres en la ROI después de eliminar ruido.")
        return []
    arriba = min(caja[1] for caja in cajas)
    abajo = max(caja[1] + caja[3] for caja in cajas)
    altura = abajo - arriba
    if max_hueco is None:
        max_hueco = int(getattr(config, "SEGMENTACION_SEPARACION_MAXIMA", 1))
    if not isinstance(max_hueco, (int, np.integer)) or max_hueco < 0:
        raise ValueError("El hueco máximo debe ser un entero no negativo.")
    signos = []
    glifos = []
    for caja in cajas:
        _, y, ancho, alto = caja
        pequeno = ancho < altura * 0.30 and alto < altura * 0.35 and alto <= ancho * 2.5
        inferior = y + alto / 2 > arriba + altura * 0.65
        (signos if pequeno and inferior else glifos).append(caja)
    if metodo == "contornos":
        agrupadas = _agrupar_cajas(glifos, int(max_hueco))
    else:
        mascara_glifos = np.zeros_like(mascara)
        for x, y, ancho, alto in glifos:
            mascara_glifos[y:y + alto, x:x + ancho] = mascara[y:y + alto, x:x + ancho]
        agrupadas = _cajas_proyeccion(mascara_glifos, int(max_hueco))
    # Un fragmento de LCD dentro del glifo ya está incluido en su recorte;
    # no debe generar un segundo carácter por su tamaño o posición inferior.
    signos = [caja for caja in signos if not any(
        _esta_contenida(caja, glifo) for glifo in agrupadas
    )]
    agrupadas.extend(_agrupar_cajas(signos, 0))
    agrupadas.sort(key=lambda caja: (caja[0], caja[1]))
    segmentos = [Segmento(cv2.copyMakeBorder(mascara[y:y + alto, x:x + ancho],
                                           2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0),
                         x, y, ancho, alto)
                 for x, y, ancho, alto in agrupadas]
    LOGGER.info("Segmentación %s: %d caracteres", metodo, len(segmentos))
    return segmentos


def _esta_contenida(caja, contenedor):
    x, y, ancho, alto = caja
    cx, cy, cancho, calto = contenedor
    return (x >= cx and y >= cy and x + ancho <= cx + cancho
            and y + alto <= cy + calto)


def _agrupar_cajas(cajas, max_hueco):
    agrupadas = []
    for caja in sorted(cajas, key=lambda item: (item[0], item[1])):
        if not agrupadas:
            agrupadas.append(caja)
            continue
        x, y, ancho, alto = caja
        prev_x, prev_y, prev_ancho, prev_alto = agrupadas[-1]
        # Ambos extremos son exclusivos; hueco cero une sólo cajas que se tocan.
        if x <= prev_x + prev_ancho + max_hueco:
            derecha = max(prev_x + prev_ancho, x + ancho)
            abajo = max(prev_y + prev_alto, y + alto)
            arriba = min(prev_y, y)
            agrupadas[-1] = (prev_x, arriba, derecha - prev_x, abajo - arriba)
        else:
            agrupadas.append(caja)
    return agrupadas


def _cajas_proyeccion(mascara, max_hueco):
    activas = np.any(mascara != 0, axis=0)
    bordes = np.diff(np.pad(activas.astype(np.int8), (1, 1)))
    inicios = np.flatnonzero(bordes == 1)
    finales = np.flatnonzero(bordes == -1)
    cajas = []
    for izquierda, derecha in zip(inicios, finales):
        filas = np.flatnonzero(np.any(mascara[:, izquierda:derecha] != 0, axis=1))
        cajas.append((int(izquierda), int(filas[0]), int(derecha - izquierda),
                      int(filas[-1] - filas[0] + 1)))
    return _agrupar_cajas(cajas, max_hueco)
