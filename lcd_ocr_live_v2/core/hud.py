"""HUD y dashboard dibujados íntegramente con OpenCV.

- Cajas de ROI con color semafórico según confianza (verde ≥ 80 %, amarillo
  60–79 %, rojo < 60 %).
- Sub-cajas de cada dígito y banner translúcido con lectura, unidad y %.
- Estadísticas de FPS, latencia, estado CSV y muestras.
- Panel lateral con displays digitales de siete segmentos y miniatura de la
  ROI binarizada de cada instrumento.
"""

import logging
import time

import cv2
import numpy as np

import config
from core.instrumentos import Lectura
from core.utils import color_confianza

LOG = logging.getLogger(__name__)

_SEGMENTOS = {
    "A": ((0.16, 0.06), (0.84, 0.06)),
    "B": ((0.88, 0.12), (0.88, 0.48)),
    "C": ((0.88, 0.55), (0.88, 0.92)),
    "D": ((0.16, 0.97), (0.84, 0.97)),
    "E": ((0.12, 0.55), (0.12, 0.92)),
    "F": ((0.12, 0.12), (0.12, 0.48)),
    "G": ((0.16, 0.51), (0.84, 0.51)),
}
_DIGITOS_7SEG = {
    "0": "ABCDEF", "1": "BC", "2": "ABGED", "3": "ABGCD", "4": "FGBC",
    "5": "AFGCD", "6": "AFGEDC", "7": "ABC", "8": "ABCDEFG", "9": "ABCDFG",
    "-": "G", " ": "",
}


def _escala(alto: int) -> float:
    """Factor de escala respecto al diseño base de 720 px."""
    return max(0.5, alto / 720.0)


def _texto(img, texto, origen, escala, color, grosor=1, sombra=True, centrado=False):
    """Dibuje texto con sombra para máximo contraste sobre vídeo."""
    fuente = cv2.FONT_HERSHEY_DUPLEX
    (ancho, alto), base = cv2.getTextSize(texto, fuente, escala, grosor)
    x, y = int(origen[0]), int(origen[1])
    if centrado:
        x -= ancho // 2
    if sombra:
        cv2.putText(img, texto, (x + 1, y + 1), fuente, escala, (0, 0, 0),
                    grosor + 1, cv2.LINE_AA)
    cv2.putText(img, texto, (x, y), fuente, escala, color, grosor, cv2.LINE_AA)
    return ancho, alto + base


def _caja_translucida(img, x1, y1, x2, y2, color, alfa=0.55):
    """Rellene un rectángulo translúcido usando ``addWeighted``."""
    alto, ancho = img.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(ancho, int(x2)), min(alto, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    region = img[y1:y2, x1:x2]
    capa = np.full(region.shape, color, dtype=np.uint8)
    cv2.addWeighted(capa, alfa, region, 1.0 - alfa, 0, region)
    img[y1:y2, x1:x2] = region


def _esquinas(img, x1, y1, x2, y2, color, grosor=2, largo=14):
    """Dibuje solo las esquinas de la caja (estilo HUD técnico)."""
    for (x, y, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                           (x2, y2, -1, -1), (x1, y2, 1, -1)):
        cv2.line(img, (x, y), (x + dx * largo, y), color, grosor, cv2.LINE_AA)
        cv2.line(img, (x, y), (x, y + dy * largo), color, grosor, cv2.LINE_AA)


def dibujar_digito_7seg(
    img: np.ndarray, digito: str, x: int, y: int, alto: int,
    color, grosor: int | None = None, inclinacion: float = 0.06,
) -> int:
    """Dibuje un dígito tipo siete segmentos y devuelva su ancho."""
    ancho = max(6, int(alto * 0.58))
    grosor = grosor or max(2, int(alto * 0.12))
    segmentos = _DIGITOS_7SEG.get(digito)
    if segmentos is None:
        return ancho
    for nombre in segmentos:
        (px1, py1), (px2, py2) = _SEGMENTOS[nombre]
        # Inclinación leve hacia la derecha en la parte superior (fuente itálica).
        desplazamiento1 = inclinacion * (1.0 - py1) * ancho
        desplazamiento2 = inclinacion * (1.0 - py2) * ancho
        p1 = (int(x + px1 * ancho + desplazamiento1), int(y + py1 * alto))
        p2 = (int(x + px2 * ancho + desplazamiento2), int(y + py2 * alto))
        cv2.line(img, p1, p2, color, grosor, cv2.LINE_AA)
    return ancho


def ancho_texto_7seg(texto: str, alto: int, espacio: float = 0.14) -> int:
    """Ancho aproximado de un texto dibujado con segmentos."""
    ancho_digito = alto * 0.58
    total = 0.0
    for caracter in texto:
        if caracter in ".,":
            total += ancho_digito * 0.30
        else:
            total += ancho_digito * (1 + espacio)
    return int(max(0.0, total - ancho_digito * espacio))


def dibujar_texto_7seg(
    img: np.ndarray, texto: str, x: int, y: int, alto: int, color,
    grosor: int | None = None,
) -> int:
    """Dibuje dígitos, signos y separadores; devuelva el ancho consumido."""
    ancho_digito = alto * 0.58
    grosor = grosor or max(2, int(alto * 0.12))
    cursor = float(x)
    inicio = cursor
    for caracter in texto:
        if caracter in ".,":
            radio = max(2, int(alto * 0.09))
            cv2.circle(img, (int(cursor + radio), int(y + alto - radio)),
                       radio, color, -1, cv2.LINE_AA)
            cursor += radio * 2 + ancho_digito * 0.12
            continue
        if caracter == " ":
            cursor += ancho_digito * 0.6
            continue
        if caracter in _DIGITOS_7SEG:
            dibujar_digito_7seg(img, caracter, int(cursor), y, alto, color, grosor)
        cursor += ancho_digito * 1.14
    return int(cursor - inicio)


def _miniatura(vista: np.ndarray | None, ancho: int, alto: int) -> np.ndarray:
    """Miniatura BGR de la máscara binaria de una ROI."""
    lienzo = np.full((alto, ancho, 3), 24, dtype=np.uint8)
    if vista is None or vista.size == 0:
        return lienzo
    imagen = vista
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    # Recorte al contenido activo para que los dígitos llenen la miniatura.
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    activo = cv2.findNonZero((gris > 0).astype(np.uint8) * 255)
    if activo is not None:
        bx, by, bw, bh = cv2.boundingRect(activo)
        margen = 2
        imagen = imagen[max(0, by - margen):by + bh + margen,
                        max(0, bx - margen):bx + bw + margen]
    alto_o, ancho_o = imagen.shape[:2]
    escala = min(ancho / max(1, ancho_o), alto / max(1, alto_o))
    nuevo = (max(1, int(ancho_o * escala)), max(1, int(alto_o * escala)))
    reducida = cv2.resize(imagen, nuevo, interpolation=cv2.INTER_NEAREST)
    x = (ancho - nuevo[0]) // 2
    y = (alto - nuevo[1]) // 2
    lienzo[y:y + nuevo[1], x:x + nuevo[0]] = reducida
    return lienzo


def dibujar_rois(frame: np.ndarray, lecturas: list[Lectura]) -> np.ndarray:
    """Dibuje cajas, sub-cajas de dígitos y banners de lectura sobre el frame."""
    salida = frame.copy()
    alto_frame, ancho_frame = salida.shape[:2]
    escala = _escala(alto_frame)
    for lectura in lecturas:
        x, y, ancho, alto = lectura.caja_roi
        if ancho <= 0 or alto <= 0:
            continue
        color, _ = color_confianza(lectura.confianza)
        cv2.rectangle(salida, (x, y), (x + ancho, y + alto), color, max(1, int(2 * escala)))
        _esquinas(salida, x, y, x + ancho, y + alto, color, max(1, int(2 * escala)),
                  int(12 * escala))
        for (dx, dy, dw, dh, confianza) in lectura.cajas:
            color_digito, _ = color_confianza(confianza)
            cv2.rectangle(
                salida,
                (x + int(dx), y + int(dy)),
                (x + int(dx + dw), y + int(dy + dh)),
                color_digito, max(1, int(escala)),
            )
        etiqueta = f"{lectura.etiqueta_rol.upper()}  {lectura.valor_mostrado}"
        sufijo = f"  {lectura.confianza:.0%}"
        if not lectura.valido:
            etiqueta = f"{lectura.etiqueta_rol.upper()}  SIN LECTURA"
        texto = etiqueta + ("" if not lectura.valido else sufijo)
        fuente = cv2.FONT_HERSHEY_DUPLEX
        tamano = 0.58 * escala
        (ancho_texto, alto_texto), base = cv2.getTextSize(texto, fuente, tamano, 1)
        banner_alto = alto_texto + base + int(10 * escala)
        banner_ancho = ancho_texto + int(18 * escala)
        banner_y = y - banner_alto - int(2 * escala)
        if banner_y < 0:
            banner_y = y + int(2 * escala)
        _caja_translucida(
            salida, x, banner_y, x + banner_ancho, banner_y + banner_alto,
            (12, 10, 8), config.HUD_BANNER_ALFA,
        )
        cv2.rectangle(salida, (x, banner_y), (x + banner_ancho, banner_y + banner_alto),
                      color, max(1, int(escala)))
        cv2.putText(salida, texto, (x + int(9 * escala), banner_y + alto_texto + int(4 * escala)),
                    fuente, tamano, color, 1, cv2.LINE_AA)
        if not lectura.estable and lectura.valido:
            _texto(salida, "estabilizando…", (x + int(4 * escala), y + alto + int(14 * escala)),
                   max(0.4, 0.42 * escala), config.HUD_COLOR_TEXTO_TENUE, 1)
    return salida


def dibujar_estado(
    frame: np.ndarray, fps: float, latencia_ms: float, estado_csv: str,
    muestras: int, pausado: bool = False, ocr_ms: float = 0.0,
) -> np.ndarray:
    """Superponga el bloque de estadísticas en la esquina superior izquierda."""
    salida = frame
    alto = salida.shape[0]
    escala = _escala(alto)
    lineas = [
        ("FPS", f"{fps:5.1f}"),
        ("Latencia", f"{latencia_ms:5.0f} ms"),
        ("OCR", f"{ocr_ms:5.0f} ms"),
        ("CSV", estado_csv),
        ("Muestras", str(muestras)),
    ]
    fuente = cv2.FONT_HERSHEY_DUPLEX
    tamano = 0.46 * escala
    ancho_clave = max(cv2.getTextSize(clave, fuente, tamano, 1)[0][0] for clave, _ in lineas)
    ancho_valor = max(cv2.getTextSize(valor, fuente, tamano, 1)[0][0] for _, valor in lineas)
    ancho_caja = int(ancho_clave + ancho_valor + 36 * escala)
    alto_caja = int((len(lineas) * 22 + 16) * escala)
    _caja_translucida(salida, int(10 * escala), int(10 * escala),
                      int(10 * escala) + ancho_caja, int(10 * escala) + alto_caja,
                      (16, 14, 12), 0.62)
    cv2.rectangle(salida, (int(10 * escala), int(10 * escala)),
                  (int(10 * escala) + ancho_caja, int(10 * escala) + alto_caja),
                  config.HUD_COLOR_ACENTO, max(1, int(escala)))
    y = int(10 * escala) + int(8 * escala)
    for indice, (clave, valor) in enumerate(lineas):
        y += int(20 * escala)
        color = config.HUD_COLOR_TEXTO
        if clave == "CSV":
            minuscula = valor.lower()
            color = (config.HUD_COLOR_VERDE if minuscula.startswith(("grab", "on", "activ"))
                     else config.HUD_COLOR_AMARILLO if "paus" in minuscula
                     or "off" in minuscula else config.HUD_COLOR_ROJO)
        _texto(salida, clave, (int(20 * escala), y), tamano, config.HUD_COLOR_TEXTO_TENUE, 1)
        _texto(salida, valor, (int(20 * escala) + int(ancho_clave) + int(14 * escala), y),
               tamano, color, 1)
    if pausado:
        texto = "MONITOREO EN PAUSA"
        (ancho, alto_t), _ = cv2.getTextSize(texto, fuente, 0.75 * escala, 2)
        x = (salida.shape[1] - ancho) // 2
        y = int(30 * escala)
        _caja_translucida(salida, x - int(12 * escala), y - alto_t,
                          x + ancho + int(12 * escala), y + int(10 * escala),
                          (10, 10, 10), 0.6)
        cv2.putText(salida, texto, (x, y), fuente, 0.75 * escala,
                    config.HUD_COLOR_AMARILLO, 2, cv2.LINE_AA)
    return salida


def _tarjeta(
    panel: np.ndarray, lectura: Lectura, x: int, y: int, ancho: int, alto: int,
    escala: float,
) -> None:
    """Dibuje una tarjeta digital con valor, unidad, confianza y miniatura."""
    color, _ = color_confianza(lectura.confianza)
    if not lectura.valido:
        color = config.HUD_COLOR_ROJO
    cv2.rectangle(panel, (x, y), (x + ancho, y + alto), config.HUD_COLOR_TARJETA, -1)
    cv2.rectangle(panel, (x, y), (x + 4, y + alto), color, -1)
    cv2.rectangle(panel, (x, y), (x + ancho, y + alto), (52, 46, 40), 1)
    _texto(panel, lectura.etiqueta_rol.upper(), (x + int(16 * escala), y + int(24 * escala)),
           max(0.45, 0.44 * escala), config.HUD_COLOR_TEXTO_TENUE, 1)
    alto_valor = int(min(alto * 0.38, 96 * escala))
    valor = lectura.texto if lectura.valido else "---"
    ancho_valor = ancho_texto_7seg(valor, alto_valor)
    if ancho_valor > ancho - int(20 * escala):
        factor = (ancho - int(20 * escala)) / max(1, ancho_valor)
        alto_valor = max(12, int(alto_valor * factor))
        ancho_valor = ancho_texto_7seg(valor, alto_valor)
    dibujar_texto_7seg(
        panel, valor, x + int(16 * escala), y + int(34 * escala), alto_valor, color,
    )
    unidad = lectura.unidad
    if unidad:
        cv2.putText(panel, unidad,
                    (x + int(16 * escala) + ancho_valor + int(8 * escala),
                     y + int(34 * escala) + alto_valor),
                    cv2.FONT_HERSHEY_DUPLEX, max(0.5, 0.58 * escala),
                    config.HUD_COLOR_TEXTO, 1, cv2.LINE_AA)
    # Barra de confianza.
    barra_y = y + alto - int(30 * escala)
    barra_x = x + int(16 * escala)
    barra_ancho = ancho - int(140 * escala)
    cv2.rectangle(panel, (barra_x, barra_y),
                  (barra_x + barra_ancho, barra_y + int(7 * escala)), (60, 54, 48), -1)
    cv2.rectangle(panel, (barra_x, barra_y),
                  (barra_x + int(barra_ancho * max(0.0, min(1.0, lectura.confianza))),
                   barra_y + int(7 * escala)), color, -1)
    _texto(panel, f"{lectura.confianza:.0%}", (barra_x, barra_y - int(6 * escala)),
           max(0.4, 0.38 * escala), color, 1)
    # Miniatura de la ROI binarizada, con la proporción real del recorte.
    if lectura.vista is not None and lectura.vista.size:
        alto_vista, ancho_vista = lectura.vista.shape[:2]
        relacion = max(0.2, ancho_vista / max(1, alto_vista))
        miniatura_ancho = int(118 * escala)
        miniatura_alto = int(min(alto - 52 * escala, miniatura_ancho / relacion))
        if miniatura_alto >= 12:
            miniatura = _miniatura(lectura.vista, miniatura_ancho, miniatura_alto)
            px = x + ancho - miniatura_ancho - int(10 * escala)
            py = y + (alto - miniatura_alto) // 2
            panel[py:py + miniatura_alto, px:px + miniatura_ancho] = miniatura
            cv2.rectangle(panel, (px - 1, py - 1),
                          (px + miniatura_ancho, py + miniatura_alto), (72, 66, 60), 1)


def dibujar_dashboard(
    alto: int, lecturas: list[Lectura], estado_csv: str = "",
    muestras: int = 0, pausado: bool = False,
) -> np.ndarray:
    """Panel lateral oscuro con displays digitales de alto contraste."""
    escala = _escala(alto)
    ancho = max(360, int(config.HUD_ANCHO_PANEL * escala))
    panel = np.full((alto, ancho, 3), config.HUD_COLOR_FONDO, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (2, alto), (60, 52, 46), -1)
    _texto(panel, "LCD OCR LIVE", (int(20 * escala), int(34 * escala)),
           max(0.5, 0.55 * escala), config.HUD_COLOR_ACENTO, 1)
    _texto(panel, "v2 · RPi 4B + OV5647", (int(20 * escala), int(56 * escala)),
           max(0.38, 0.34 * escala), config.HUD_COLOR_TEXTO_TENUE, 1)
    marca = time.strftime("%H:%M:%S")
    ancho_marca, _ = cv2.getTextSize(marca, cv2.FONT_HERSHEY_DUPLEX,
                                     max(0.42, 0.40 * escala), 1)
    _texto(panel, marca, (ancho - ancho_marca[0] - int(18 * escala), int(34 * escala)),
           max(0.42, 0.40 * escala), config.HUD_COLOR_TEXTO, 1)
    cv2.line(panel, (int(16 * escala), int(70 * escala)),
             (ancho - int(16 * escala), int(70 * escala)), (58, 50, 44), 1)
    cabecera = int(84 * escala)
    pie = int(48 * escala)
    cantidad = max(1, len(lecturas))
    disponible = alto - cabecera - pie - int(10 * escala)
    hueco = int(8 * escala)
    alto_tarjeta = max(56, min(int(210 * escala),
                               (disponible - hueco * (cantidad - 1)) // cantidad))
    total = cantidad * alto_tarjeta + hueco * (cantidad - 1)
    inicio = cabecera + max(0, (disponible - total) // 2)
    for indice, lectura in enumerate(lecturas):
        y = inicio + indice * (alto_tarjeta + hueco)
        _tarjeta(panel, lectura, int(14 * escala), y,
                 ancho - int(28 * escala), alto_tarjeta, escala)
    if not lecturas:
        _texto(panel, "Sin ROIs configuradas", (int(20 * escala), int(alto / 2)),
               max(0.5, 0.5 * escala), config.HUD_COLOR_TEXTO_TENUE, 1)
    estado_color = (config.HUD_COLOR_VERDE if estado_csv.lower().startswith(("grab", "on"))
                    else config.HUD_COLOR_AMARILLO if "paus" in estado_csv.lower()
                    else config.HUD_COLOR_ROJO)
    cv2.circle(panel, (int(22 * escala), alto - int(26 * escala)),
               int(5 * escala), estado_color, -1, cv2.LINE_AA)
    _texto(panel, estado_csv or "CSV: —", (int(34 * escala), alto - int(20 * escala)),
           max(0.4, 0.38 * escala), config.HUD_COLOR_TEXTO, 1)
    _texto(panel, f"{muestras} muestras", (ancho - int(130 * escala), alto - int(20 * escala)),
           max(0.4, 0.38 * escala), config.HUD_COLOR_TEXTO_TENUE, 1)
    return panel


def componer_vista(frame: np.ndarray, panel: np.ndarray, separacion: int = 2) -> np.ndarray:
    """Una el frame (escalado a la altura del panel) con el dashboard lateral."""
    alto_panel = panel.shape[0]
    alto_frame, ancho_frame = frame.shape[:2]
    if alto_frame != alto_panel:
        ancho = max(1, int(round(ancho_frame * alto_panel / alto_frame)))
        frame = cv2.resize(frame, (ancho, alto_panel), interpolation=cv2.INTER_AREA)
    separador = np.full((alto_panel, separacion, 3), (70, 62, 56), dtype=np.uint8)
    return np.hstack([frame, separador, panel])


def dibujar_hud(
    frame: np.ndarray,
    lecturas: list[Lectura],
    fps: float,
    latencia_ms: float,
    estado_csv: str,
    muestras: int,
    pausado: bool = False,
    ocr_ms: float = 0.0,
) -> np.ndarray:
    """Pipeline de dibujo del HUD: ROIs, banners, sub-cajas y estadísticas."""
    salida = dibujar_rois(frame, lecturas)
    salida = dibujar_estado(salida, fps, latencia_ms, estado_csv, muestras,
                            pausado, ocr_ms)
    return salida
