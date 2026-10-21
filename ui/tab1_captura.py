"""Captura CSI y apertura de imágenes locales sin bloquear el hilo de Tkinter."""

import logging
from queue import Empty, Queue
from threading import Event, Thread
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import config
from core.camara import CamaraLCD
from core.utils import cargar_imagen, guardar_imagen
from ui.widgets import VisorImagen


LOGGER = logging.getLogger(__name__)


class CapturaTab(ttk.Frame):
    """Controles de cámara, preview y carga local para Windows o Raspberry Pi."""

    def __init__(self, parent, app) -> None:
        """Prepare los controles y un trabajador único para todas las capturas."""
        super().__init__(parent, padding=10)
        self.app = app
        self._camara = CamaraLCD()
        self._comandos = Queue()
        self._eventos = Queue()
        self._fin = Event()
        self._ultimo_preview = None
        self._ocupada = False
        self._cerrado = False
        self._timer = None
        barra = ttk.Frame(self)
        barra.pack(fill=tk.X)
        self.boton_iniciar = ttk.Button(barra, text="Iniciar cámara", command=self.iniciar)
        self.boton_iniciar.pack(side=tk.LEFT, padx=3)
        self.boton_capturar = ttk.Button(barra, text="Capturar", command=self.capturar)
        self.boton_capturar.pack(side=tk.LEFT, padx=3)
        self.boton_detener = ttk.Button(barra, text="Detener cámara", command=self.detener)
        self.boton_detener.pack(side=tk.LEFT, padx=3)
        self.boton_abrir = ttk.Button(barra, text="Abrir imagen", command=self.abrir_imagen)
        self.boton_abrir.pack(side=tk.LEFT, padx=3)
        self.estado = tk.StringVar(value="Inicie la cámara CSI o abra una imagen local.")
        ttk.Label(self, textvariable=self.estado, wraplength=950).pack(fill=tk.X, pady=6)
        self.visor = VisorImagen(self)
        self.visor.pack(fill=tk.BOTH, expand=True)
        # Al cerrar, una captura en curso debe terminar y ejecutar finally.
        self._trabajador = Thread(target=self._trabajar, name="camara-lcd", daemon=False)
        self._trabajador.start()
        self._actualizar_botones()
        self._timer = self.after(config.INTERVALO_PREVIEW_MS, self._recibir)

    def iniciar(self) -> None:
        """Solicite abrir Picamera2 desde el trabajador de cámara."""
        self._enviar("iniciar", "Iniciando cámara CSI…")

    def capturar(self) -> None:
        """Solicite una captura de alta resolución y su guardado a disco."""
        self._enviar("capturar", "Capturando y guardando imagen…")

    def detener(self) -> None:
        """Solicite detener el preview y liberar la cámara."""
        self._ultimo_preview = None
        self._enviar("detener", "Deteniendo cámara…")

    def abrir_imagen(self) -> None:
        """Seleccione una imagen local para probar todo el flujo sin cámara CSI."""
        ruta = filedialog.askopenfilename(
            parent=self,
            title="Abrir captura del LCD",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")],
        )
        if ruta:
            self._enviar("archivo", "Abriendo imagen…", ruta)

    def actualizar(self) -> None:
        """Muestre la captura vigente cuando no hay un preview activo."""
        if (self.app.imagen_capturada is not None and not self._camara.activa
                and self.visor.imagen is not self.app.imagen_capturada):
            self.visor.mostrar_imagen(self.app.imagen_capturada)

    def cerrar(self) -> None:
        """Cancele el polling de Tk y solicite liberar el hardware al cerrar."""
        self._cerrado = True
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        self._fin.set()
        self._comandos.put(("salir", None))
        self._trabajador.join(timeout=2)
        if self._trabajador.is_alive():
            LOGGER.info("Esperando que termine la captura para liberar la cámara.")

    def _enviar(self, comando, mensaje, argumento=None):
        if self._ocupada:
            return
        self._ocupada = True
        self.estado.set(mensaje)
        self._actualizar_botones()
        self._comandos.put((comando, argumento))

    def _actualizar_botones(self):
        libre = not self._ocupada
        self.boton_iniciar.configure(state="normal" if libre and not self._camara.activa else "disabled")
        self.boton_capturar.configure(state="normal" if libre and self._camara.activa else "disabled")
        self.boton_detener.configure(state="normal" if libre and self._camara.activa else "disabled")
        self.boton_abrir.configure(state="normal" if libre else "disabled")

    def _trabajar(self):
        siguiente_preview = 0.0
        try:
            while not self._fin.is_set():
                try:
                    comando, argumento = self._comandos.get(timeout=0.04)
                except Empty:
                    comando = None
                try:
                    if comando == "salir":
                        break
                    if comando == "iniciar":
                        self._camara.iniciar()
                        self._eventos.put(("estado", "Cámara activa. Capture para continuar a perspectiva."))
                    elif comando == "capturar":
                        imagen = self._camara.capturar()
                        self._eventos.put(("captura", imagen))
                    elif comando == "detener":
                        self._camara.cerrar()
                        self._ultimo_preview = None
                        self._eventos.put(("estado", "Cámara detenida."))
                    elif comando == "archivo":
                        self._camara.cerrar()
                        self._ultimo_preview = None
                        imagen = cargar_imagen(argumento)
                        guardar_imagen(imagen, config.CAPTURA_ACTUAL)
                        self._eventos.put(("captura", imagen))
                    ahora = time.monotonic()
                    if self._camara.activa and ahora >= siguiente_preview:
                        self._ultimo_preview = self._camara.obtener_preview()
                        siguiente_preview = ahora + config.INTERVALO_PREVIEW_MS / 1000
                except Exception as error:
                    LOGGER.exception("Error durante una operación de cámara o archivo.")
                    try:
                        self._camara.cerrar()
                    except Exception:
                        LOGGER.exception("No se pudo cerrar la cámara tras el fallo.")
                    self._ultimo_preview = None
                    self._eventos.put(("error", str(error)))
        finally:
            try:
                self._camara.cerrar()
            except Exception:
                LOGGER.exception("Error al liberar la cámara al cerrar la aplicación.")

    def _recibir(self):
        if self._cerrado:
            return
        try:
            while True:
                tipo, dato = self._eventos.get_nowait()
                self._ocupada = False
                if tipo == "captura":
                    self.app.establecer_captura(dato)
                    self.visor.mostrar_imagen(dato)
                    self.estado.set(f"Captura {dato.shape[1]}×{dato.shape[0]} guardada en {config.CAPTURA_ACTUAL}")
                elif tipo == "error":
                    self.estado.set(dato)
                    messagebox.showerror("Captura", dato, parent=self)
                else:
                    self.estado.set(dato)
                self._actualizar_botones()
        except Empty:
            pass
        if self._ultimo_preview is not None:
            self.visor.mostrar_imagen(self._ultimo_preview)
            self._ultimo_preview = None
        self._timer = self.after(config.INTERVALO_PREVIEW_MS, self._recibir)
