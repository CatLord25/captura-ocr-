"""Ventana Tkinter y estado compartido del flujo de calibración + vivo."""

import logging
import tkinter as tk
from tkinter import ttk

import config
from ui.tab1_captura import CapturaTab
from ui.tab2_perspectiva import PerspectivaTab
from ui.tab3_filtrado import FiltradoTab
from ui.tab4_rois import ROIsTab
from ui.tab5_reconocimiento import ReconocimientoTab
from ui.tab6_live_hud import LiveHUDTab

LOG = logging.getLogger(__name__)
NOMBRES_PESTANAS = (
    "1 · Captura",
    "2 · Perspectiva",
    "3 · Filtrado",
    "4 · ROIs",
    "5 · Reconocimiento",
    "6 · En vivo (HUD)",
)


class AplicacionLCD(tk.Tk):
    """Coordine las pestañas sin introducir lógica de visión en la ventana."""

    def __init__(self):
        """Construya la aplicación sin abrir cámara ni cargar datasets."""
        super().__init__()
        self.title(config.VENTANA_TITULO)
        self.geometry(config.VENTANA_GEOMETRIA)
        self.minsize(*config.VENTANA_MINIMA)
        self.imagen_capturada = None
        self.imagen_corregida = None
        self.imagen_binaria = None
        self.rois = []
        self.rois_pendientes = None
        self.resultados = []
        self.rectificador = None
        self.filtro_pendiente = False
        self.revision = 0
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=6, pady=6)
        self.estado = tk.StringVar(value="Capture o abra una imagen para empezar.")
        ttk.Label(self, textvariable=self.estado, padding=(10, 3)).pack(fill="x")
        tipos = (CapturaTab, PerspectivaTab, FiltradoTab, ROIsTab,
                 ReconocimientoTab, LiveHUDTab)
        self.pestanas = {}
        for nombre, tipo in zip(NOMBRES_PESTANAS, tipos):
            tab = tipo(self.notebook, self)
            self.notebook.add(tab, text=nombre)
            self.pestanas[nombre] = tab
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    def establecer_captura(self, imagen):
        """Guarde una captura y reinicie las etapas dependientes."""
        self.imagen_capturada = imagen
        self.imagen_corregida = None
        self.imagen_binaria = None
        self.rois = []
        self.resultados = []
        self.rectificador = None
        self.estado.set("Imagen disponible. Seleccione las cuatro esquinas o defina ROIs.")
        self.notificar_cambio()

    def establecer_corregida(self, imagen):
        """Guarde la perspectiva y reinicie binarización, ROIs y resultados."""
        self.imagen_corregida = imagen
        self.imagen_binaria = None
        self.rois = []
        self.resultados = []
        self.estado.set("Perspectiva aplicada. Ajuste el filtrado y dibuje las ROIs.")
        self.notificar_cambio()

    def establecer_binaria(self, imagen):
        """Guarde la imagen binaria y descarte resultados obsoletos."""
        if self.imagen_binaria is not None and self.imagen_binaria.shape != imagen.shape:
            self.rois = []
        self.imagen_binaria = imagen
        self.resultados = []
        self.notificar_cambio()

    def establecer_procesada(self, imagen, binaria=None):
        """Cargue una imagen ya procesada conservando sus píxeles binarios."""
        self.imagen_capturada = imagen
        self.imagen_corregida = imagen
        self.imagen_binaria = binaria
        self.rois = []
        self.resultados = []
        self.estado.set("Imagen procesada cargada. Dibuje o cargue ROIs antes de reconocer.")
        self.notificar_cambio()

    def notificar_cambio(self):
        """Actualice las pestañas e incremente la revisión del estado."""
        self.revision += 1
        for tab in self.pestanas.values():
            actualizar = getattr(tab, "actualizar", None)
            if actualizar:
                actualizar()

    def cerrar(self):
        """Detenga trabajadores, libere cámara y cierre Tkinter."""
        for tab in self.pestanas.values():
            cerrar = getattr(tab, "cerrar", None)
            if cerrar:
                try:
                    cerrar()
                except Exception:
                    LOG.exception("Error al cerrar una pestaña")
        self.destroy()
