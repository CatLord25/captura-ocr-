"""Generador de escenas LCD/7seg sintéticas a partir del dataset del proyecto.

Sirve para tres cosas: probar el HUD sin cámara, validar el OCR de extremo a
extremo y hacer demos en PC. Compone los glifos reales del dataset sobre un
panel con gradiente, viñeteado y ruido, como si se mirara el instrumento.
"""

from dataclasses import dataclass
import logging
import time
from pathlib import Path

import cv2
import numpy as np

import config
from core.dataset import cargar_banco
from core.roi import ROI

LOG = logging.getLogger(__name__)

VALORES = {
    "Balanza": ["0.000", "1.245", "12.340", "0.500", "99.995", "-0.125"],
    "termohigrobarometro": [
        ("23.4", "56.7", "1013.2"),
        ("24.1", "55.9", "1012.8"),
        ("19.8", "61.2", "1010.5"),
        ("-5.0", "48.0", "1009.0"),
        ("31.2", "70.4", "1008.3"),
    ],
}
ROLES_TERMOHIGRO = ("temperatura", "humedad", "presion")
UNIDADES_SIMULADAS = {
    "temperatura": ("grados_celsius.png", "°C"),
    "humedad": ("humedad_relativa.png", "%RH"),
    "presion": (None, "hPa"),
    "peso": (None, "kg"),
}


def _mascara(imagen: np.ndarray) -> np.ndarray:
    """Extraiga la tinta del recorte del dataset como máscara 0/255."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY) if imagen.ndim == 3 else imagen
    suave = cv2.GaussianBlur(gris, (3, 3), 0)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate((binaria[0], binaria[-1], binaria[:, 0], binaria[:, -1]))
    if np.count_nonzero(borde) > borde.size / 2:
        binaria = cv2.bitwise_not(binaria)
    cantidad, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if cantidad <= 1:
        return np.zeros_like(binaria)
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    conservar = np.concatenate(([False], areas >= max(4, 0.03 * areas.max())))
    mascara = np.where(conservar[etiquetas], 255, 0).astype(np.uint8)
    puntos = cv2.findNonZero(mascara)
    if puntos is None:
        return np.zeros_like(binaria)
    x, y, w, h = cv2.boundingRect(puntos)
    return mascara[y:y + h, x:x + w]


@dataclass
class _Campo:
    rol: str
    textos: tuple[str, ...]
    x: int
    y: int
    ancho: int
    alto: int


class GeneradorEscenas:
    """Componga frames sintéticos con valores que cambian en el tiempo."""

    def __init__(
        self,
        instrumento: str = "termohigrobarometro",
        resolucion: tuple[int, int] = (1280, 720),
        intervalo_cambio: float = 2.0,
        semilla: int = 7,
    ) -> None:
        """Cargue el dataset y precalcule la disposición del panel."""
        self.instrumento = config.instrumento_canonico(instrumento)
        self.ancho, self.alto = (int(valor) for valor in resolucion)
        self.intervalo_cambio = float(intervalo_cambio)
        self._semilla = int(semilla)
        self._banco = cargar_banco(self.instrumento)
        self._mascaras = self._cargar_mascaras()
        self._inicio = time.monotonic()
        self._valores = VALORES[self.instrumento]
        self._campos: list[_Campo] = []
        self._panel = (0, 0, 0, 0)
        self._unidades: list[tuple[str, np.ndarray | None, str]] = []
        self._cache_texto: dict[tuple[str, int], np.ndarray] = {}
        self._cache_fondo: np.ndarray | None = None
        self._cache_panel: np.ndarray | None = None
        self._disponer()

    # ------------------------------------------------------------- inicialización
    def _cargar_mascaras(self) -> dict[str, np.ndarray]:
        """Máscaras por etiqueta del banco (dígitos y símbolos)."""
        mascaras: dict[str, np.ndarray] = {}
        for plantilla in self._banco.plantillas:
            imagen = cv2.imdecode(np.fromfile(plantilla.archivo, np.uint8),
                                  cv2.IMREAD_GRAYSCALE)
            if imagen is None:
                continue
            mascaras.setdefault(plantilla.etiqueta, _mascara(imagen))
        return mascaras

    def _mascara_texto(self, texto: str, altura: int) -> np.ndarray:
        """Componga un texto con los glifos reales del dataset (con caché)."""
        clave = (texto, int(altura))
        if clave in self._cache_texto:
            return self._cache_texto[clave]
        piezas = []
        for caracter in texto:
            if caracter == " ":
                piezas.append(None)
                continue
            mascara = self._mascaras.get(caracter)
            if mascara is None:
                raise KeyError(f"El dataset no contiene el glifo {caracter!r}.")
            if caracter in ".,":
                escala = altura * 0.22 / mascara.shape[0]
            elif caracter == "-":
                escala = altura * 0.10 / mascara.shape[0]
            else:
                escala = altura / mascara.shape[0]
            ancho = max(2, round(mascara.shape[1] * escala))
            alto = max(2, round(mascara.shape[0] * escala))
            redimensionada = cv2.resize(mascara, (ancho, alto), interpolation=cv2.INTER_AREA)
            _, redimensionada = cv2.threshold(redimensionada, 127, 255, cv2.THRESH_BINARY)
            piezas.append(redimensionada)
        separacion = max(3, round(altura * 0.10))
        ancho_total = sum(
            (pieza.shape[1] if pieza is not None else separacion * 2) + separacion
            for pieza in piezas
        )
        lienzo = np.zeros((altura, max(1, ancho_total)), dtype=np.uint8)
        x = 0
        for caracter, pieza in zip(texto, piezas):
            if pieza is None:
                x += separacion * 2
                continue
            # El signo menos va centrado en altura; dígitos y puntuación, a la base.
            if caracter == "-":
                y = (altura - pieza.shape[0]) // 2
            else:
                y = altura - pieza.shape[0] if pieza.shape[0] < altura else 0
            lienzo[y:y + pieza.shape[0], x:x + pieza.shape[1]] = np.maximum(
                lienzo[y:y + pieza.shape[0], x:x + pieza.shape[1]], pieza
            )
            x += pieza.shape[1] + separacion
        recorte = cv2.findNonZero(lienzo)
        if recorte is None:
            resultado = np.zeros((altura, 1), np.uint8)
        else:
            bx, by, bw, bh = cv2.boundingRect(recorte)
            resultado = lienzo[by:by + bh, bx:bx + bw]
        resultado.flags.writeable = False
        self._cache_texto[clave] = resultado
        return resultado

    def _disponer(self) -> None:
        """Calcule posiciones estables de campos, unidades y panel."""
        if self.instrumento == "Balanza":
            roles = ("peso",)
            textos_por_rol = {"peso": tuple(self._valores)}
            altura = max(72, int(self.alto * 0.15))
            inicio_y = int(self.alto * 0.42)
            paso = int(altura * 2.2)
            margen_x = int(self.ancho * 0.14)
        else:
            roles = ROLES_TERMOHIGRO
            textos_por_rol = {
                rol: tuple(valores[indice] for valores in self._valores)
                for indice, rol in enumerate(roles)
            }
            altura = max(48, int(self.alto * 0.085))
            inicio_y = int(self.alto * 0.20)
            paso = int(altura * 2.1)
            margen_x = int(self.ancho * 0.12)
        x_campo = margen_x + int(self.ancho * 0.16)
        ancho_maximo = 0
        for indice, rol in enumerate(roles):
            muestras = textos_por_rol[rol]
            dimensiones = [self._mascara_texto(texto, altura).shape
                           for texto in muestras]
            ancho_campo = max(ancho for _, ancho in dimensiones) + 12
            alto_campo = max(alto_ for alto_, _ in dimensiones) + 8
            campo = _Campo(
                rol=rol, textos=muestras, x=x_campo,
                y=inicio_y + indice * paso,
                ancho=ancho_campo, alto=alto_campo,
            )
            self._campos.append(campo)
            ancho_maximo = max(ancho_maximo, x_campo + ancho_campo)
            archivo_unidad, etiqueta_unidad = UNIDADES_SIMULADAS[rol]
            mascara_unidad = None
            if archivo_unidad is not None:
                for plantilla in self._banco.plantillas:
                    if plantilla.archivo.name == archivo_unidad:
                        mascara_unidad = self._mascaras.get(plantilla.etiqueta)
                        break
            self._unidades.append((rol, mascara_unidad, etiqueta_unidad))
            if mascara_unidad is not None:
                ancho_unidad = int(mascara_unidad.shape[1] *
                                   (altura * 0.45 / mascara_unidad.shape[0]))
                ancho_maximo = max(ancho_maximo, x_campo + ancho_campo + 18 + ancho_unidad)
            else:
                ancho_maximo = max(ancho_maximo, x_campo + ancho_campo + 90)
        ultimo = self._campos[-1]
        panel_x = max(10, margen_x - int(self.ancho * 0.05))
        panel_y = max(10, inicio_y - int(altura * 1.2))
        panel_ancho = min(
            self.ancho - panel_x - 10,
            max(120, ancho_maximo + int(self.ancho * 0.05) - panel_x),
        )
        panel_alto = min(
            self.alto - panel_y - 10,
            (ultimo.y + ultimo.alto + int(altura * 0.35)) - panel_y,
        )
        self._panel = (panel_x, panel_y, panel_ancho, panel_alto)

    @property
    def cantidad_muestras(self) -> int:
        """Número de combinaciones de valores de la escena."""
        return len(self._valores)

    def valores(self, momento: float | None = None) -> dict[str, str]:
        """Texto que la escena muestra en cada campo para un instante dado."""
        ahora = time.monotonic() if momento is None else float(momento)
        indice = int(max(0, ahora - self._inicio) / max(0.1, self.intervalo_cambio))
        return {
            campo.rol: campo.textos[indice % len(campo.textos)]
            for campo in self._campos
        }

    def frame_muestra(self, indice: int) -> np.ndarray:
        """Renderice una muestra concreta de la tabla de valores."""
        total = len(self._valores)
        return self.frame(self._inicio + (int(indice) % total) * self.intervalo_cambio)

    def valores_muestra(self, indice: int) -> dict[str, str]:
        """Valores esperados de una muestra concreta (para verificación)."""
        total = len(self._valores)
        posicion = int(indice) % total
        return {campo.rol: campo.textos[posicion % len(campo.textos)]
                for campo in self._campos}

    def rois(self) -> list[ROI]:
        """ROIs de los campos numéricos dibujados (coordenadas del frame)."""
        return [
            ROI(
                nombre=campo.rol, x=campo.x + 2, y=campo.y,
                ancho=campo.ancho - 4, alto=campo.alto,
                instrumento=self.instrumento, rol=campo.rol,
            )
            for campo in self._campos
        ]

    # ------------------------------------------------------------------ render
    def _fondo(self) -> np.ndarray:
        """Fondo oscuro con gradiente y viñeteado (cacheado)."""
        if self._cache_fondo is not None:
            return self._cache_fondo.copy()
        xx = np.linspace(0, 1, self.ancho, dtype=np.float32)
        yy = np.linspace(0, 1, self.alto, dtype=np.float32)[:, None]
        base = 26 + 24 * xx[None, :] + 10 * (1 - yy)
        distancia = ((xx[None, :] - 0.5) ** 2 + (yy - 0.5) ** 2)
        base *= (1 - 0.55 * np.clip(distancia * 2.2, 0, 1))
        fondo = np.dstack([base * 0.85, base * 0.92, base]).astype(np.uint8)
        self._cache_fondo = fondo
        return fondo.copy()

    def _panel_base(self) -> np.ndarray:
        """Panel claro del instrumento con leve gradiente (cacheado)."""
        if self._cache_panel is not None:
            return self._cache_panel
        x, y, ancho, alto = self._panel
        xx = np.linspace(0, 1, ancho, dtype=np.float32)
        base = 192 + 30 * xx
        panel = np.empty((alto, ancho, 3), dtype=np.float32)
        panel[..., 0] = base * 0.98
        panel[..., 1] = base
        panel[..., 2] = base * 0.96
        yy = np.linspace(-1, 1, alto, dtype=np.float32)[:, None, None]
        panel *= (1 - 0.06 * yy ** 2)
        self._cache_panel = np.clip(panel, 0, 255).astype(np.uint8)
        return self._cache_panel

    def frame(self, momento: float | None = None) -> np.ndarray:
        """Genere un frame BGR con los valores vigentes."""
        ahora = time.monotonic() if momento is None else float(momento)
        indice = int(max(0, ahora - self._inicio) / max(0.1, self.intervalo_cambio))
        frame = self._fondo()
        panel = self._panel_base()
        x, y, ancho, alto = self._panel
        # Sombra y borde del panel.
        sombra = frame.copy()
        cv2.rectangle(sombra, (x + 8, y + 8), (x + ancho + 8, y + alto + 8), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.55, sombra, 0.45, 0)
        frame[y:y + alto, x:x + ancho] = panel
        cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (70, 74, 78), 2)
        for posicion, campo in enumerate(self._campos):
            texto = campo.textos[indice % len(campo.textos)]
            mascara = self._mascara_texto(texto, int(campo.alto * 0.86))
            ox = campo.x + campo.ancho - mascara.shape[1] - 4
            oy = campo.y + (campo.alto - mascara.shape[0]) // 2
            self._pintar(frame, mascara, ox, oy)
            etiqueta = {"temperatura": "TEMP", "humedad": "HUM", "presion": "PRES",
                        "peso": "PESO"}.get(campo.rol, campo.rol.upper()[:8])
            cv2.putText(frame, etiqueta, (x + 24, campo.y + campo.alto - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 64, 68), 1, cv2.LINE_AA)
            _, mascara_unidad, etiqueta_unidad = self._unidades[posicion]
            ux = campo.x + campo.ancho + 12
            if mascara_unidad is not None:
                altura_unidad = int(campo.alto * 0.45)
                escala = altura_unidad / mascara_unidad.shape[0]
                ancho_unidad = max(2, round(mascara_unidad.shape[1] * escala))
                redimensionada = cv2.resize(
                    mascara_unidad, (ancho_unidad, altura_unidad),
                    interpolation=cv2.INTER_AREA,
                )
                _, redimensionada = cv2.threshold(redimensionada, 127, 255, cv2.THRESH_BINARY)
                self._pintar(frame, redimensionada, ux,
                             campo.y + campo.alto - altura_unidad - 2)
            else:
                cv2.putText(frame, etiqueta_unidad, (ux, campo.y + campo.alto - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 44, 48), 2, cv2.LINE_AA)
        # Reflejo especular fuera de las ROIs y ruido de sensor.
        cv2.ellipse(frame, (x + ancho - 40, y + 30), (70, 26), -18, 0, 360,
                    (235, 240, 245), -1, cv2.LINE_AA)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        generador = np.random.default_rng(self._semilla + indice)
        # Ruido a 1/4 de resolución y reescalado: mismo aspecto, 16× más rápido.
        pequeño = generador.normal(
            0, 2.4, (max(2, self.alto // 4), max(2, self.ancho // 4), 3)
        ).astype(np.float32)
        ruido = cv2.resize(pequeño, (self.ancho, self.alto), interpolation=cv2.INTER_LINEAR)
        brillo = 1.0 + 0.02 * np.sin(ahora * 2.1)
        return np.clip(frame.astype(np.float32) * brillo + ruido, 0, 255).astype(np.uint8)

    @staticmethod
    def _pintar(frame: np.ndarray, mascara: np.ndarray, x: int, y: int) -> None:
        """Componga una máscara de tinta oscura con antialiasing."""
        alto, ancho = mascara.shape
        if alto <= 0 or ancho <= 0:
            return
        if y < 0 or x < 0 or y + alto > frame.shape[0] or x + ancho > frame.shape[1]:
            return
        alpha = cv2.GaussianBlur(mascara, (3, 3), 0).astype(np.float32) / 255.0
        region = frame[y:y + alto, x:x + ancho].astype(np.float32)
        tinta = np.array([34, 38, 40], np.float32)
        frame[y:y + alto, x:x + ancho] = (
            region * (1 - alpha[..., None]) + tinta * alpha[..., None]
        ).astype(np.uint8)
