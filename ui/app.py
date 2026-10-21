"""Ventana Tkinter y estado compartido del flujo de reconocimiento."""

import logging
import tkinter as tk
from tkinter import ttk

import config
from ui.tab1_captura import CapturaTab
from ui.tab2_perspectiva import PerspectivaTab
from ui.tab3_filtrado import FiltradoTab
from ui.tab4_rois import ROIsTab
from ui.tab5_reconocimiento import ReconocimientoTab


NOMBRES_PESTANAS = (
    "Captura de Imagen", "Corrección de Perspectiva",
    "Filtrado y Binarización", "Definición de ROIs",
    "Segmentación y Reconocimiento",
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
        self.resultados = []
        self.filtro_pendiente = False
        self.revision = 0
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)
        self.estado = tk.StringVar(value="Capture o abra una imagen para empezar.")
        ttk.Label(self, textvariable=self.estado, padding=(10, 3)).pack(fill="x")
        self.pestanas = {}
        tipos = (CapturaTab, PerspectivaTab, FiltradoTab, ROIsTab, ReconocimientoTab)
        for nombre, tipo in zip(NOMBRES_PESTANAS, tipos):
            tab = tipo(self.notebook, self)
            self.notebook.add(tab, text=nombre)
            self.pestanas[nombre] = tab
        self.protocol("WM_DELETE_WINDOW", self.cerrar)

    def establecer_captura(self, imagen):
        """Guarde una nueva captura RGB y reinicie las etapas dependientes."""
        self.imagen_capturada = imagen
        self.imagen_corregida = None
        self.imagen_binaria = None
        self.rois = []
        self.resultados = []
        self.estado.set("Imagen disponible. Seleccione las cuatro esquinas del LCD.")
        self.notificar_cambio()

    def establecer_corregida(self, imagen):
        """Guarde la perspectiva RGB y reinicie binarización, ROIs y resultados."""
        self.imagen_corregida = imagen
        self.imagen_binaria = None
        self.rois = []
        self.resultados = []
        self.estado.set("Perspectiva aplicada. Ajuste el filtrado y dibuje las ROIs.")
        self.notificar_cambio()

    def establecer_binaria(self, imagen):
        """Guarde la imagen binaria y descarte resultados de la versión anterior."""
        if self.imagen_binaria is not None and self.imagen_binaria.shape != imagen.shape:
            self.rois = []
        self.imagen_binaria = imagen
        self.resultados = []
        self.notificar_cambio()

    def establecer_procesada(self, rgb, binaria):
        """Cargue una imagen procesada conservando sus píxeles binarios."""
        self.imagen_capturada = rgb
        self.imagen_corregida = rgb
        self.imagen_binaria = binaria
        self.rois = []
        self.resultados = []
        self.estado.set("Imagen procesada cargada. Dibuje o cargue ROIs antes de reconocer.")
        self.notificar_cambio()

    def notificar_cambio(self):
        """Actualice pestañas y cambie la revisión para evitar resultados obsoletos."""
        self.revision += 1
        for tab in self.pestanas.values():
            actualizar = getattr(tab, "actualizar", None)
            if actualizar:
                actualizar()

    def cerrar(self):
        """Detenga tareas pendientes, libere la cámara y cierre Tkinter."""
        for tab in self.pestanas.values():
            cerrar = getattr(tab, "cerrar", None)
            if cerrar:
                try:
                    cerrar()
                except Exception:
                    logging.getLogger(__name__).exception("Error al cerrar una pestaña")
        self.destroy()
