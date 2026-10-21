"""Controles de filtrado con ejecución en segundo plano y preview en vivo."""

import queue
import threading
import tkinter as tk
from tkinter import filedialog, ttk

import config
from core.preprocesado import filtrar_imagen
from core.utils import guardar_imagen
from ui.widgets import VisorImagen, mostrar_error


class FiltradoTab(ttk.Frame):
    """Ajuste la binarización sin bloquear la interfaz durante el cálculo."""

    def __init__(self, parent, app):
        """Prepare sliders, operaciones y una cola de resultados."""
        super().__init__(parent, padding=10)
        self.app = app
        self._fuente = None
        self._binaria = None
        self._ticket = 0
        self._debounce = None
        self._ocupado = False
        self._repetir = False
        self._cerrado = False
        self._cola = queue.Queue()
        self.threshold = tk.IntVar(value=config.THRESHOLD_BINARIO)
        self.kernel = tk.IntVar(value=config.KERNEL_MORFOLOGICO)
        self.iteraciones = tk.IntVar(value=1)
        self.invertir = tk.BooleanVar(value=config.INVERTIR_BINARIO)
        self.metodo = tk.StringVar(value="manual")
        self.operacion = tk.StringVar(value="ninguna")
        controles = ttk.Frame(self)
        controles.pack(fill="x")
        ttk.Label(controles, text="Threshold").grid(row=0, column=0, sticky="w")
        ttk.Scale(controles, from_=0, to=255, variable=self.threshold,
                  command=lambda _: self._programar()).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Label(controles, textvariable=self.threshold, width=4).grid(row=0, column=2)
        ttk.Label(controles, text="Método").grid(row=0, column=3, padx=(10, 4))
        ttk.Combobox(controles, textvariable=self.metodo, state="readonly",
                     values=("manual", "otsu", "adaptativo"), width=13).grid(row=0, column=4)
        ttk.Checkbutton(controles, text="Invertir", variable=self.invertir).grid(row=0, column=5, padx=8)
        ttk.Label(controles, text="Kernel impar").grid(row=1, column=0, pady=8)
        ttk.Spinbox(controles, from_=1, to=31, increment=2,
                    textvariable=self.kernel, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(controles, text="Operación").grid(row=1, column=2)
        ttk.Combobox(controles, textvariable=self.operacion, state="readonly",
                     values=("ninguna", "erosion", "dilatacion", "apertura", "cierre"),
                     width=13).grid(row=1, column=3, columnspan=2)
        ttk.Label(controles, text="Iteraciones").grid(row=1, column=5)
        ttk.Spinbox(controles, from_=1, to=10, textvariable=self.iteraciones,
                    width=4).grid(row=1, column=6)
        ttk.Button(controles, text="Guardar binaria…", command=self._guardar).grid(row=0, column=6, padx=5)
        controles.columnconfigure(1, weight=1)
        self.mensaje = tk.StringVar(value="El filtrado usa el LCD corregido o la captura original.")
        ttk.Label(self, textvariable=self.mensaje).pack(fill="x", pady=(0, 7))
        self.visor = VisorImagen(self)
        self.visor.pack(fill="both", expand=True)
        for variable in (self.kernel, self.iteraciones, self.invertir, self.metodo, self.operacion):
            variable.trace_add("write", lambda *_: self._programar())
        self._poll = self.after(60, self._recibir)

    def actualizar(self):
        """Recalcule si cambia la fuente o muestre una binaria cargada."""
        binaria = self.app.imagen_binaria
        if binaria is not self._binaria:
            self._binaria = binaria
            if binaria is not None and (self._ocupado or self._debounce):
                self._ticket += 1
                self._repetir = False
                self.app.filtro_pendiente = False
                if self._debounce:
                    self.after_cancel(self._debounce)
                    self._debounce = None
        fuente = self.app.imagen_corregida
        if fuente is None:
            fuente = self.app.imagen_capturada
        if fuente is not self._fuente:
            self._fuente = fuente
            self._ticket += 1
            self.visor.mostrar_imagen(self.app.imagen_binaria)
            if self.app.imagen_binaria is None:
                self._programar()
            else:
                self._repetir = False
                self.app.filtro_pendiente = False
                if self._debounce:
                    self.after_cancel(self._debounce)
                    self._debounce = None
        elif self.visor.imagen is not self.app.imagen_binaria:
            self.visor.mostrar_imagen(self.app.imagen_binaria)

    def cerrar(self):
        """Cancele callbacks; los trabajadores no acceden a Tkinter."""
        self._cerrado = True
        if self._debounce:
            self.after_cancel(self._debounce)
        self.after_cancel(self._poll)

    def _programar(self):
        if self._cerrado:
            return
        self._ticket += 1
        self.app.filtro_pendiente = self._fuente is not None
        self.app.resultados = []
        self.app.notificar_cambio()
        if self._debounce:
            self.after_cancel(self._debounce)
        self._debounce = self.after(getattr(config, "RETARDO_FILTRO_MS", 180), self._filtrar)

    def _filtrar(self):
        if self._debounce:
            self.after_cancel(self._debounce)
        self._debounce = None
        if self._fuente is None:
            self.app.filtro_pendiente = False
            return
        if self._ocupado:
            self._repetir = True
            return
        try:
            parametros = dict(threshold=int(float(self.threshold.get())),
                              kernel=self.kernel.get(), operacion=self.operacion.get(),
                              invertir=self.invertir.get(), metodo=self.metodo.get(),
                              iteraciones=self.iteraciones.get())
            if parametros["kernel"] % 2 == 0 or parametros["kernel"] < 1:
                raise ValueError("El kernel debe ser positivo e impar.")
        except (tk.TclError, ValueError) as error:
            self.app.filtro_pendiente = False
            self.mensaje.set(str(error))
            return
        self._ocupado = True
        self._repetir = False
        self.mensaje.set("Procesando…")
        ticket, fuente = self._ticket, self._fuente
        def trabajo():
            try:
                resultado = filtrar_imagen(fuente, **parametros)
                self._cola.put((ticket, resultado, None))
            except Exception as error:
                self._cola.put((ticket, None, error))
        threading.Thread(target=trabajo, daemon=True).start()

    def _recibir(self):
        if self._cerrado:
            return
        try:
            ticket, resultado, error = self._cola.get_nowait()
        except queue.Empty:
            pass
        else:
            self._ocupado = False
            if ticket == self._ticket:
                self.app.filtro_pendiente = False
                if error:
                    self.mensaje.set(f"Error de filtrado: {error}")
                else:
                    self.app.establecer_binaria(resultado)
                    self.mensaje.set("Preview actualizado. Ajuste para separar los caracteres del fondo.")
            if self._repetir:
                self._filtrar()
        self._poll = self.after(60, self._recibir)

    def _guardar(self):
        if self.app.imagen_binaria is None:
            mostrar_error("Sin binaria", "Primero capture o abra una imagen y ajuste el filtrado.")
            return
        ruta = filedialog.asksaveasfilename(
            parent=self, initialdir=config.DIRECTORIO_CAPTURAS,
            initialfile="lcd_binario.png", defaultextension=".png",
            filetypes=[("PNG", "*.png")])
        if ruta:
            try:
                guardar_imagen(self.app.imagen_binaria, ruta)
                self.mensaje.set(f"Guardada: {ruta}")
            except Exception as error:
                mostrar_error("No se pudo guardar", error)
