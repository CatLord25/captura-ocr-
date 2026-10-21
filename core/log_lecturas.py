"""Registro CSV de lecturas y confidencias por carácter."""

import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import threading

import config

LOGGER = logging.getLogger(__name__)
CABECERA = ("timestamp", "instrumento", "roi", "valor", "confianzas")
_CERROJO = threading.Lock()


def _serializar_numero(valor):
    if hasattr(valor, "item"):
        return valor.item()
    raise TypeError(f"Confianza no serializable: {type(valor).__name__}.")


def registrar_lectura(instrumento: str, roi: str, valor: str, confianzas, ruta=None) -> dict:
    """Añada una lectura CSV con fecha ISO local y sus confidencias como JSON.

    Cree directorios y cabecera cuando hagan falta. Valide una cabecera ya
    existente y serialice con CSV estándar para preservar comas y símbolos.
    Devuelva la fila registrada; lance OSError si el archivo no se puede escribir.
    """
    destino = Path(ruta if ruta is not None else config.ARCHIVO_LECTURAS).expanduser()
    if not isinstance(instrumento, str) or not instrumento.strip():
        raise ValueError("Debe indicar el instrumento de la lectura.")
    if not isinstance(roi, str) or not roi.strip():
        raise ValueError("Debe indicar el nombre de la ROI de la lectura.")
    confianza_json = json.dumps(confianzas, ensure_ascii=False, allow_nan=False,
                                default=_serializar_numero, separators=(",", ":"))
    fila = {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "instrumento": instrumento.strip(), "roi": roi.strip(),
            "valor": str(valor), "confianzas": confianza_json}
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
        LOGGER.exception("No se pudo registrar la lectura en %s", destino)
        raise OSError(f"No se pudo registrar la lectura en {destino}: {error}") from error
    LOGGER.info("Lectura %s / %s: %s", instrumento, roi, valor)
    return fila
