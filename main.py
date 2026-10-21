"""Punto de entrada de la aplicación modular LCD OCR."""

import logging
import sys

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Inicie la interfaz y devuelva cero si la aplicación termina correctamente."""
    if sys.version_info < (3, 11):
        logging.basicConfig(level=logging.ERROR)
        LOGGER.error("Se requiere Python 3.11 o posterior.")
        return 1
    try:
        from core.utils import configurar_logging
        configurar_logging()
        from tkinter import TclError
        from ui.app import AplicacionLCD
    except ImportError as error:
        logging.basicConfig(level=logging.ERROR)
        LOGGER.exception(
            "Falta una dependencia (%s). En Windows instale requirements.txt "
            "y Python con Tcl/Tk; en la Pi instale los paquetes apt del README "
            "y active el venv con --system-site-packages.", error.name
        )
        return 1

    try:
        aplicacion = AplicacionLCD()
        LOGGER.info("LCD OCR: ventana creada con cinco pestañas.")
        aplicacion.mainloop()
    except TclError:
        LOGGER.exception(
            "No se pudo iniciar o mantener la interfaz Tkinter. "
            "Ejecute la aplicación desde una sesión de escritorio con pantalla."
        )
        return 1
    LOGGER.info("Aplicación cerrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
