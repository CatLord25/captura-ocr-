"""Segmentación geométrica robusta de líneas LCD y displays de 7 segmentos.

Estrategia (inspirada en ``7seg-ocr`` y ``SegoDec``):

1. Binarizado y puenteo morfológico de la ROI.
2. **Proyección vertical**: cada dígito de siete segmentos ocupa todas sus
   columnas (las barras A/D/G cruzan el ancho), así que los huecos entre
   dígitos son columnas sin tinta. Esto agrupa los siete segmentos de un
   dígito sin necesidad de que se toquen.
3. Agrupación de componentes por cercanía para puntos y comas punteados.
4. Pre-clasificación geométrica (punto/coma, menos, dígito estrecho).
5. División de dígitos pegados por el valle de la proyección vertical.
"""

from dataclasses import dataclass
import logging

import cv2
import numpy as np

import config
from core.preprocesado import PreprocesoROI, perfil_valido, preprocesar

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segmento:
    """Recorte de carácter con su caja en coordenadas de la ROI."""

    imagen: np.ndarray
    x: int
    y: int
    ancho: int
    alto: int
    clase: str = "digito"      # digito | puntuacion | menos
    aspecto: float = 0.0
    estrecho: bool = False


def _cajas_componentes(mascara: np.ndarray, area_minima: int) -> list[tuple[int, int, int, int]]:
    """Cajas de contornos externos con tinta suficiente."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cajas: list[tuple[int, int, int, int]] = []
    for contorno in contornos:
        x, y, ancho, alto = cv2.boundingRect(contorno)
        if ancho < 1 or alto < 1:
            continue
        if cv2.countNonZero(mascara[y:y + alto, x:x + ancho]) >= area_minima:
            cajas.append((int(x), int(y), int(ancho), int(alto)))
    return cajas


def _extension_vertical(cajas) -> float:
    """Altura total de la línea a partir de las cajas detectadas."""
    if not cajas:
        return 1.0
    arriba = min(caja[1] for caja in cajas)
    abajo = max(caja[1] + caja[3] for caja in cajas)
    return float(max(1, abajo - arriba))


def _runs_proyeccion(mascara: np.ndarray, hueco: int) -> list[tuple[int, int]]:
    """Intervalos de columnas con tinta (los huecos separan caracteres)."""
    activas = np.any(mascara > 0, axis=0)
    if not np.any(activas):
        return []
    cambios = np.diff(np.pad(activas.astype(np.int8), (1, 1)))
    inicios = np.flatnonzero(cambios == 1)
    finales = np.flatnonzero(cambios == -1)
    runs: list[list[int]] = []
    for inicio, fin in zip(inicios, finales):
        if runs and inicio - runs[-1][1] <= hueco:
            runs[-1][1] = int(fin)
        else:
            runs.append([int(inicio), int(fin)])
    return [(inicio, fin) for inicio, fin in runs]


def _asignar(runs, cajas):
    """Asigne cada componente al intervalo de columnas con mayor solape."""
    grupos: list[list[tuple[int, int, int, int]]] = [[] for _ in runs]
    for caja in cajas:
        x1, x2 = caja[0], caja[0] + caja[2]
        mejor, solape_maximo = None, 0
        for indice, (inicio, fin) in enumerate(runs):
            solape = min(x2, fin) - max(x1, inicio)
            if solape > solape_maximo:
                mejor, solape_maximo = indice, solape
        if mejor is not None:
            grupos[mejor].append(caja)
        else:
            # Componente fuera de toda proyección: se ignora (ruido de borde).
            continue
    return grupos


def _solape_vertical(a, b) -> float:
    """Fracción de solape vertical respecto a la caja más baja."""
    arriba = max(a[1], b[1])
    abajo = min(a[1] + a[3], b[1] + b[3])
    if abajo <= arriba:
        return 0.0
    return (abajo - arriba) / max(1.0, float(min(a[3], b[3])))


def _hueco_horizontal(a, b) -> int:
    """Separación horizontal (0 si se tocan o solapan)."""
    return max(0, b[0] - (a[0] + a[2]), a[0] - (b[0] + b[2]))





def _unir(cajas):
    """Caja envolvente de una lista de cajas."""
    x1 = min(caja[0] for caja in cajas)
    y1 = min(caja[1] for caja in cajas)
    x2 = max(caja[0] + caja[2] for caja in cajas)
    y2 = max(caja[1] + caja[3] for caja in cajas)
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _agrupar_puntuacion(
    cajas: list[tuple[int, int, int, int]], altura_linea: float
) -> list[tuple[int, int, int, int]]:
    """Una los puntos dispersos de una coma o un punto decimal."""
    if not cajas:
        return []
    tolerancia = max(2, int(round(altura_linea * 0.05)))
    grupos = sorted(cajas, key=lambda caja: (caja[0], caja[1]))
    cambiado = True
    while cambiado:
        cambiado = False
        resultado: list[tuple[int, int, int, int]] = []
        for caja in grupos:
            fusionada = False
            for indice, grupo in enumerate(resultado):
                if _hueco_horizontal(grupo, caja) <= tolerancia * 2:
                    resultado[indice] = _unir([grupo, caja])
                    fusionada = True
                    cambiado = True
                    break
            if not fusionada:
                resultado.append(caja)
        grupos = resultado
    return grupos


def _cortes_valle(submascara: np.ndarray, piezas: int) -> list[int]:
    """Posiciones de corte en los mínimos de la proyección vertical."""
    alto, ancho = submascara.shape
    proyeccion = np.count_nonzero(submascara, axis=0)
    ancho_pieza = ancho / piezas
    radio = max(2, int(round(ancho_pieza * config.SEGMENTACION_VALLE_VENTANA)))
    cortes: list[int] = []
    for indice in range(1, piezas):
        ideal = int(round(indice * ancho_pieza))
        izquierda = max(1, ideal - radio)
        derecha = min(ancho - 1, ideal + radio)
        if derecha <= izquierda:
            continue
        ventana = proyeccion[izquierda:derecha]
        minimo = int(ventana.min())
        candidatos = np.flatnonzero(ventana <= minimo + 1) + izquierda
        corte = int(candidatos[np.argmin(np.abs(candidatos - ideal))])
        if cortes and corte <= cortes[-1] + 1:
            continue
        cortes.append(corte)
    return cortes


def _dividir_pegados(
    cajas: list[tuple[int, int, int, int]],
    mascara: np.ndarray,
    altura_digito: float,
    ancho_tipico: float,
    area_minima: int,
) -> list[tuple[int, int, int, int]]:
    """Divida bloques anchos (dos cifras pegadas) por valles de proyección."""
    salida: list[tuple[int, int, int, int]] = []
    for x, y, ancho, alto in cajas:
        aspecto = ancho / max(1, alto)
        if (alto < altura_digito * 0.55 or aspecto <= config.GEOM_DIGITO_AR_MAX
                or ancho <= ancho_tipico * 1.30):
            salida.append((x, y, ancho, alto))
            continue
        piezas = int(np.clip(round(ancho / max(1.0, ancho_tipico)), 2, 6))
        recorte = mascara[y:y + alto, x:x + ancho]
        cortes = _cortes_valle(recorte, piezas)
        if len(cortes) != piezas - 1:
            salida.append((x, y, ancho, alto))
            continue
        limites = [0, *cortes, ancho]
        for inicio, fin in zip(limites[:-1], limites[1:]):
            pieza = recorte[:, inicio:fin]
            puntos = cv2.findNonZero(pieza)
            if puntos is None or cv2.countNonZero(pieza) < area_minima:
                continue
            px, py, pw, ph = cv2.boundingRect(puntos)
            salida.append((x + inicio + px, y + py, max(1, pw), max(1, ph)))
    return salida


def _ancho_tipico(cajas, altura_digito: float) -> float:
    """Ancho medio de dígito para estimar cuántas cifras hay en un bloque."""
    anchos = [
        caja[2] for caja in cajas
        if caja[3] >= altura_digito * 0.55 and 0.30 <= caja[2] / max(1, caja[3]) <= 1.05
    ]
    if anchos:
        return float(np.median(anchos))
    return max(4.0, altura_digito * 0.62)


def _clasificar(caja, altura_roi: float, altura_digito: float) -> tuple[str, bool]:
    """Pre-clasifique por posición y aspecto: puntuación, menos o dígito."""
    x, y, ancho, alto = caja
    aspecto = ancho / max(1, alto)
    centro_y = y + alto / 2
    puntuacion = (
        alto <= max(altura_digito * 0.42, altura_roi * 0.12)
        and alto <= altura_roi * 0.45
        and ancho <= alto * 1.8
        and centro_y > altura_roi * config.GEOM_PUNTO_Y_MIN
    )
    if puntuacion:
        return "puntuacion", aspecto < config.AR_ESTRECHO
    menos = (
        config.GEOM_MENOS_Y_MIN * altura_roi < centro_y < config.GEOM_MENOS_Y_MAX * altura_roi
        and aspecto >= config.GEOM_MENOS_AR_MIN
        and alto <= altura_digito * config.GEOM_MENOS_ALTO_MAX
    )
    if menos:
        return "menos", False
    estrecho = aspecto < config.AR_ESTRECHO and alto >= altura_roi * config.GEOM_DIGITO_ALTO_MIN
    return "digito", estrecho


def segmentar_preproceso(
    pre: PreprocesoROI,
    area_minima: int | None = None,
    max_hueco: int | None = None,
    dividir_pegados: bool = True,
) -> list[Segmento]:
    """Segmente una ROI ya preprocesada y devuelva caracteres ordenados."""
    if not isinstance(pre, PreprocesoROI):
        raise TypeError("Se esperaba un PreprocesoROI de core.preprocesado.")
    minimo = int(config.SEGMENTACION_AREA_MINIMA if area_minima is None else area_minima)
    if minimo < 1:
        raise ValueError("El área mínima debe ser positiva.")
    hueco = int(config.SEGMENTACION_SEPARACION_MAXIMA if max_hueco is None else max_hueco)
    if hueco < 0:
        raise ValueError("El hueco máximo debe ser no negativo.")
    mascara = pre.puente
    alto_roi, ancho_roi = mascara.shape
    cajas = _cajas_componentes(mascara, minimo)
    if not cajas:
        LOG.debug("Sin componentes tras el binarizado.")
        return []
    altura_linea = _extension_vertical(cajas)
    runs = _runs_proyeccion(mascara, hueco)
    if not runs:
        return []
    grupos = _asignar(runs, cajas)
    cajas_digito: list[tuple[int, int, int, int]] = []
    cajas_puntuacion: list[tuple[int, int, int, int]] = []
    for grupo in grupos:
        if not grupo:
            continue
        extension = _extension_vertical(grupo)
        base = max(caja[1] + caja[3] for caja in grupo)
        caja = _unir(grupo)
        # Un intervalo bajo, compacto y pegado a la base es una coma o un
        # punto aislado. Un signo menos es bajo pero mucho más ancho que alto
        # (aspecto > 1.7) y debe seguir el camino de la clasificación normal.
        es_puntuacion = (
            extension <= altura_linea * 0.45
            and base > alto_roi * config.GEOM_PUNTO_Y_MIN
            and caja[2] <= max(1.0, caja[3]) * 1.8
        )
        if es_puntuacion:
            cajas_puntuacion.append(caja)
        else:
            cajas_digito.append(caja)
    if not cajas_digito and not cajas_puntuacion:
        return []
    altura_digito = max((caja[3] for caja in cajas_digito), default=altura_linea)
    ancho_tipico = _ancho_tipico(cajas_digito, altura_digito)
    if dividir_pegados and cajas_digito:
        cajas_digito = _dividir_pegados(
            cajas_digito, mascara, altura_digito, ancho_tipico, minimo
        )
    if cajas_puntuacion:
        cajas_puntuacion = _agrupar_puntuacion(cajas_puntuacion, altura_linea)
    finales = [*cajas_digito, *cajas_puntuacion]
    segmentos: list[Segmento] = []
    for caja in finales:
        clase, estrecho = _clasificar(caja, float(alto_roi), altura_digito)
        x, y, ancho, alto = caja
        margen = 2
        x0, y0 = max(0, x - margen), max(0, y - margen)
        x1 = min(ancho_roi, x + ancho + margen)
        y1 = min(alto_roi, y + alto + margen)
        segmentos.append(Segmento(
            imagen=pre.gris[y0:y1, x0:x1].copy(),
            x=int(x), y=int(y), ancho=int(ancho), alto=int(alto),
            clase=clase, aspecto=round(ancho / max(1, alto), 4), estrecho=estrecho,
        ))
    segmentos.sort(key=lambda segmento: (segmento.x, segmento.y))
    LOG.debug("Segmentación %s: %d caracteres %s", pre.perfil, len(segmentos),
              [segmento.clase for segmento in segmentos])
    return segmentos


def segmentar_imagen(
    imagen: np.ndarray,
    perfil: str = "7seg",
    usar_clahe: bool = config.CLAHE_ACTIVO,
    metodo: str = "auto",
    invertir: bool | None = None,
    area_minima: int | None = None,
    max_hueco: int | None = None,
    dividir_pegados: bool = True,
) -> list[Segmento]:
    """Preprocese y segmente una ROI completa en una sola llamada."""
    pre = preprocesar(imagen, perfil=perfil_valido(perfil), usar_clahe=usar_clahe,
                      metodo=metodo, invertir=invertir, area_minima=area_minima)
    return segmentar_preproceso(pre, area_minima, max_hueco, dividir_pegados)


def agrupar_por_linea(segmentos: list[Segmento]) -> list[list[Segmento]]:
    """Separe segmentos en líneas por solape vertical (ROIs de varias filas)."""
    if not segmentos:
        return []
    lineas: list[list[Segmento]] = []
    for segmento in sorted(segmentos, key=lambda item: item.y):
        colocado = False
        for linea in lineas:
            referencia = linea[0]
            if _solape_vertical(
                (segmento.x, segmento.y, segmento.ancho, segmento.alto),
                (referencia.x, referencia.y, referencia.ancho, referencia.alto),
            ) >= 0.45:
                linea.append(segmento)
                colocado = True
                break
        if not colocado:
            lineas.append([segmento])
    return [sorted(linea, key=lambda item: item.x) for linea in lineas]
