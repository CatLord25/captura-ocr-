"""Punto de entrada de la aplicación modular LCD OCR LIVE v2."""

import logging
import sys

LOGGER = logging.getLogger(__name__)


def main() -> int:
    """Inicie la interfaz y devuelva cero si la aplicación termina bien."""
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
            "Falta una dependencia (%s). En Windows instale requirements.txt; "
            "en la Raspberry Pi use los paquetes apt del README y un venv con "
            "--system-site-packages.", error.name,
        )
        return 1

    aplicacion = None
    try:
        aplicacion = AplicacionLCD()
        LOGGER.info("LCD OCR LIVE v2: ventana con seis pestañas.")
        aplicacion.mainloop()
    except TclError:
        LOGGER.exception(
            "No se pudo iniciar o mantener la interfaz Tkinter. "
            "Ejecute la aplicación desde una sesión de escritorio con pantalla."
        )
        return 1
    finally:
        if aplicacion is not None:
            try:
                aplicacion.destroy()
            except Exception:
                pass
    LOGGER.info("Aplicación cerrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
