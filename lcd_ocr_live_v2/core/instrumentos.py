"""Semántica de instrumentos: rol de cada ROI, unidades y composición.

Convierte los caracteres reconocidos (``"1" "2" "." "3"``) en una lectura con
significado físico: texto, número, unidad y rol (Peso, Temperatura, Humedad,
Presión). La asignación de rol usa, por orden: el campo ``rol`` de la ROI, el
nombre de la ROI y, como último recurso, el orden de las ROIs del
termohigrobarómetro.
"""

from dataclasses import dataclass
import re

import numpy as np

import config
from core.dataset import PUNTUACION, UNIDADES
from core.roi import ROI

ROLES_ESPECIALES = {"unidad": "Unidad", "texto": "Texto"}
_ORDEN_TERMOHIGRO = ("temperatura", "humedad", "presion")


@dataclass(frozen=True)
class Lectura:
    """Medición interpretada de una ROI."""

    roi: str
    instrumento: str
    rol: str
    texto: str
    numero: float | None
    unidad: str
    confianza: float
    confianza_minima: float
    caracteres: tuple = ()
    cajas: tuple[tuple[int, int, int, int, float], ...] = ()
    caja_roi: tuple[int, int, int, int] = (0, 0, 0, 0)
    estable: bool = False
    muestras: int = 0
    vista: np.ndarray | None = None
    error: str = ""

    @property
    def valido(self) -> bool:
        """True si hay texto reconocido sin caracteres dudosos."""
        return bool(self.texto) and "?" not in self.texto and "□" not in self.texto

    @property
    def etiqueta_rol(self) -> str:
        """Nombre visible del rol (Peso, Temperatura, …)."""
        return ROLES_ESPECIALES.get(self.rol) or config.ROLES.get(self.rol, "Valor")

    @property
    def valor_mostrado(self) -> str:
        """Texto listo para HUD/CSV: valor y unidad."""
        if not self.texto:
            return "—"
        return f"{self.texto} {self.unidad}".strip()


def _normalizar(texto: str) -> str:
    return config.normalizar_nombre(texto)


def rol_de_roi(roi: ROI, indice: int = 0) -> str:
    """Deduzca el rol de medición de una ROI."""
    if roi.rol:
        return roi.rol
    nombre = _normalizar(roi.nombre)
    for rol, claves in config.CLAVES_ROL:
        if any(clave in nombre for clave in claves):
            return rol
    if roi.instrumento == "Balanza":
        return "peso"
    return _ORDEN_TERMOHIGRO[indice % len(_ORDEN_TERMOHIGRO)]


def unidad_por_defecto(rol: str) -> str:
    """Unidad esperada para un rol de medición."""
    return config.UNIDAD_POR_ROL.get(rol, "")


def parsear_numero(texto: str) -> float | None:
    """Convierta ``"12,34"``, ``"1.234,5"`` o ``"-0.5"`` en número.

    Usa el último separador decimal encontrado y descarta los separadores de
    miles. Devuelve None si no hay dígitos o el texto contiene ``?``.
    """
    if not isinstance(texto, str) or not texto.strip() or "?" in texto:
        return None
    limpio = re.sub(r"[^0-9,.\-]", "", texto)
    if not any(caracter.isdigit() for caracter in limpio):
        return None
    signo = -1.0 if limpio.startswith("-") else 1.0
    limpio = limpio.lstrip("-")
    indice = max(limpio.rfind("."), limpio.rfind(","))
    if indice >= 0:
        entero = re.sub(r"[.,]", "", limpio[:indice]) or "0"
        decimal = re.sub(r"[.,]", "", limpio[indice + 1:])
    else:
        entero, decimal = limpio, ""
    try:
        valor = float(f"{entero}.{decimal}" if decimal else entero)
    except ValueError:
        return None
    return signo * valor


def texto_numerico(resultados) -> str:
    """Una las etiquetas numéricas en orden, excluyendo unidades."""
    partes = [resultado.etiqueta for resultado in resultados
              if resultado.etiqueta not in UNIDADES and resultado.grupo != "unidad"]
    return "".join(partes)


def unidad_detectada(resultados) -> str:
    """Primera unidad reconocida entre los caracteres de la ROI."""
    for resultado in resultados:
        if resultado.etiqueta in UNIDADES:
            return resultado.etiqueta
    return ""


def componer_lectura(
    roi: ROI,
    resultados,
    instrumento: str,
    indice: int = 0,
    cajas: tuple[tuple[int, int, int, int, float], ...] = (),
    vista: np.ndarray | None = None,
    error: str = "",
) -> Lectura:
    """Componga la lectura física a partir de los caracteres reconocidos."""
    canonico = config.instrumento_canonico(instrumento)
    rol = rol_de_roi(roi, indice)
    resultados = tuple(resultados)
    if rol in {"unidad", "texto"}:
        texto = resultados[0].etiqueta if resultados else ""
        confianza = resultados[0].confianza if resultados else 0.0
        return Lectura(
            roi=roi.nombre, instrumento=canonico, rol=rol, texto=texto,
            numero=None, unidad="", confianza=confianza,
            confianza_minima=confianza, caracteres=resultados, cajas=cajas,
            caja_roi=(roi.x, roi.y, roi.ancho, roi.alto), estable=True,
            muestras=1, vista=vista, error=error,
        )
    texto = texto_numerico(resultados)
    confianzas = [resultado.confianza for resultado in resultados
                  if resultado.grupo != "unidad"]
    confianza = float(np.mean(confianzas)) if confianzas else 0.0
    minima = float(min(confianzas)) if confianzas else 0.0
    unidad = roi.unidad or unidad_detectada(resultados) or unidad_por_defecto(rol)
    return Lectura(
        roi=roi.nombre, instrumento=canonico, rol=rol, texto=texto,
        numero=parsear_numero(texto), unidad=unidad,
        confianza=confianza, confianza_minima=minima,
        caracteres=resultados, cajas=cajas,
        caja_roi=(roi.x, roi.y, roi.ancho, roi.alto),
        estable=False, muestras=1, vista=vista, error=error,
    )


def es_puntuacion(etiqueta: str) -> bool:
    """Indique si la etiqueta es un separador decimal."""
    return etiqueta in PUNTUACION
