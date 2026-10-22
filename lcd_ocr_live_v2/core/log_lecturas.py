"""Registro CSV de lecturas, compatible con el ``lecturas.csv`` original.

Cabecera: ``timestamp,instrumento,roi,valor,confianzas``.
"""

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import threading

import config

LOG = logging.getLogger(__name__)
CABECERA = tuple(config.CSV_CABECERA)
_CERROJO = threading.Lock()


def _serializar_numero(valor):
    if hasattr(valor, "item"):
        return valor.item()
    raise TypeError(f"Confianza no serializable: {type(valor).__name__}.")


def _ruta(ruta) -> Path:
    return Path(ruta if ruta is not None else config.ARCHIVO_LECTURAS).expanduser()


def registrar_lectura(instrumento: str, roi: str, valor: str, confianzas, ruta=None) -> dict:
    """Añada una fila con fecha ISO local; cree carpeta y cabecera si faltan.

    Valide una cabecera preexistente y use CSV estándar para preservar comas y
    símbolos. Devuelva la fila registrada; lance OSError si no se puede escribir.
    """
    destino = _ruta(ruta)
    if not isinstance(instrumento, str) or not instrumento.strip():
        raise ValueError("Debe indicar el instrumento de la lectura.")
    if not isinstance(roi, str) or not roi.strip():
        raise ValueError("Debe indicar el nombre de la ROI de la lectura.")
    try:
        confianza_json = json.dumps(
            list(confianzas), ensure_ascii=False, allow_nan=False,
            default=_serializar_numero, separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"Confianzas no serializables: {error}") from error
    fila = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "instrumento": instrumento.strip(),
        "roi": roi.strip(),
        "valor": str(valor),
        "confianzas": confianza_json,
    }
    try:
        with _CERROJO:
            destino.parent.mkdir(parents=True, exist_ok=True)
            with destino.open("a+", encoding="utf-8", newline="") as archivo:
                archivo.seek(0)
                cabecera = next(csv.reader(archivo), None)
                if cabecera is not None and tuple(cabecera) != CABECERA:
                    raise ValueError(
                        f"Cabecera CSV incompatible en {destino}. "
                        f"Se esperaba: {','.join(CABECERA)}."
                    )
                archivo.seek(0, 2)
                escritor = csv.DictWriter(archivo, fieldnames=CABECERA)
                if cabecera is None:
                    escritor.writeheader()
                escritor.writerow(fila)
                archivo.flush()
    except OSError as error:
        LOG.exception("No se pudo registrar la lectura en %s", destino)
        raise OSError(f"No se pudo registrar la lectura en {destino}: {error}") from error
    LOG.debug("Lectura %s / %s: %s", instrumento, roi, valor)
    return fila


def registrar_lecturas(lecturas, ruta=None) -> int:
    """Registre una lista de ``Lectura`` válidas y devuelva cuántas se guardaron.

    Una lectura con ``?`` o sin texto no se escribe: el CSV conserva solo
    valores publicados por el filtro temporal.
    """
    guardadas = 0
    for lectura in lecturas:
        if not getattr(lectura, "valido", False):
            continue
        registrar_lectura(
            lectura.instrumento, lectura.roi, lectura.texto,
            [caracter.confianza for caracter in lectura.caracteres], ruta,
        )
        guardadas += 1
    return guardadas


def contar_registros(ruta=None) -> int:
    """Cuente filas de datos del CSV (sin cabecera); 0 si no existe."""
    destino = _ruta(ruta)
    try:
        with _CERROJO, destino.open("r", encoding="utf-8", newline="") as archivo:
            lector = csv.reader(archivo)
            next(lector, None)
            return sum(1 for fila in lector if any(campo.strip() for campo in fila))
    except FileNotFoundError:
        return 0
    except OSError:
        LOG.warning("No se pudo contar las lecturas de %s", destino)
        return 0


def leer_ultimas(cantidad: int = 20, ruta=None) -> list[dict]:
    """Devuelva las últimas filas del CSV como diccionarios."""
    if cantidad < 1:
        raise ValueError("La cantidad debe ser positiva.")
    destino = _ruta(ruta)
    try:
        with _CERROJO, destino.open("r", encoding="utf-8", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
    except FileNotFoundError:
        return []
    except OSError as error:
        raise OSError(f"No se pudo leer {destino}: {error}") from error
    return filas[-int(cantidad):]
