"""Pestaña 1: captura OV5647 / webcam, preview en vivo y escenas de prueba."""

import logging
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, ttk

import config
from core.camara import CameraStream, ErrorCamara
from core.simulador import GeneradorEscenas
from core.utils import cargar_imagen, guardar_imagen
from ui.widgets import VisorImagen, mostrar_error

LOG = logging.getLogger(__name__)
RESOLUCIONES = ("1280×720", "960×720", "640×480", "1920×1080", "2592×1944")
BACKENDS = ("auto", "v4l2", "gstreamer", "picamera2", "dshow", "msmf", "any")


class CapturaTab(ttk.Frame):
    """Controles de cámara, preview y carga local para Raspberry Pi o PC."""

    def __init__(self, parent, app):
        """Prepare controles, cámara y bucle de preview."""
        super().__init__(parent, padding=10)
        self.app = app
        self._camara: CameraStream | None = None
        self._eventos: Queue = Queue()
        self._abriendo = False
        self._cerrado = False
        self._timer = None
        self._ultimo_mostrado = 0.0

        barra = ttk.Frame(self)
        barra.pack(fill=tk.X)
        ttk.Label(barra, text="Cámara").pack(side=tk.LEFT)
        self.indice = tk.IntVar(value=config.CAMARA_INDICE)
        ttk.Spinbox(barra, from_=0, to=4, width=3, textvariable=self.indice).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(barra, text="Resolución").pack(side=tk.LEFT)
        self.resolucion = tk.StringVar(value="1280×720")
        ttk.Combobox(barra, textvariable=self.resolucion, values=RESOLUCIONES,
                     state="readonly", width=11).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(barra, text="Backend").pack(side=tk.LEFT)
        self.backend = tk.StringVar(value=config.CAMARA_BACKEND)
        ttk.Combobox(barra, textvariable=self.backend, values=BACKENDS,
                     state="readonly", width=10).pack(side=tk.LEFT, padx=(4, 10))
        self.boton_iniciar = ttk.Button(barra, text="Iniciar cámara", command=self.iniciar)
        self.boton_iniciar.pack(side=tk.LEFT, padx=3)
        self.boton_capturar = ttk.Button(barra, text="Capturar", command=self.capturar,
                                         state="disabled")
        self.boton_capturar.pack(side=tk.LEFT, padx=3)
        self.boton_detener = ttk.Button(barra, text="Detener", command=self.detener,
                                        state="disabled")
        self.boton_detener.pack(side=tk.LEFT, padx=3)
        ttk.Button(barra, text="Abrir imagen", command=self.abrir_imagen).pack(side=tk.LEFT, padx=3)
        ttk.Button(barra, text="Escena sintética", command=self.escena_sintetica).pack(
            side=tk.LEFT, padx=3)

        sensores = ttk.Frame(self)
        sensores.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(sensores, text="Sensor OV5647:").pack(side=tk.LEFT)
        ttk.Button(sensores, text="Modo día", command=lambda: self.modo_sensor(False)).pack(
            side=tk.LEFT, padx=3)
        ttk.Button(sensores, text="Modo noche (IR)", command=lambda: self.modo_sensor(True)).pack(
            side=tk.LEFT, padx=3)
        ttk.Label(sensores, text="(evita quemar los dígitos con LED/IR)").pack(side=tk.LEFT, padx=8)

        self.estado = tk.StringVar(value="Inicie la cámara o abra una imagen local.")
        ttk.Label(self, textvariable=self.estado, wraplength=1050).pack(fill=tk.X, pady=6)
        self.visor = VisorImagen(self)
        self.visor.pack(fill=tk.BOTH, expand=True)
        self._timer = self.after(40, self._recibir)

    # -------------------------------------------------------------- acciones
    def _resolucion(self) -> tuple[int, int]:
        ancho, alto = self.resolucion.get().replace("x", "×").split("×")
        return int(ancho), int(alto)

    def iniciar(self) -> None:
        """Abra la cámara en segundo plano (no bloquea la interfaz)."""
        if self._abriendo or (self._camara is not None and self._camara.activa):
            return
        self._abriendo = True
        self.boton_iniciar.configure(state="disabled")
        self.estado.set("Abriendo la cámara…")
        indice = int(self.indice.get())
        resolucion = self._resolucion()
        backend = self.backend.get()

        def abrir():
            try:
                camara = CameraStream(indice, resolucion, config.CAMARA_FPS, backend)
                camara.iniciar()
                self._eventos.put(("camara", camara))
            except Exception as error:
                self._eventos.put(("error", str(error)))

        Thread(target=abrir, daemon=True, name="abrir-camara").start()

    def capturar(self) -> None:
        """Capture el frame actual, guárdelo y páselo al flujo de calibración."""
        camara = self._camara
        try:
            if camara is not None and camara.activa:
                imagen = camara.capturar()
            else:
                imagen = cargar_imagen(config.CAPTURA_ACTUAL)
                self.estado.set(f"Imagen cargada desde {config.CAPTURA_ACTUAL}")
            self.app.establecer_captura(imagen)
            self.visor.mostrar_imagen(imagen)
            if camara is not None and camara.activa:
                self.estado.set(
                    f"Captura {imagen.shape[1]}×{imagen.shape[0]} guardada en "
                    f"{config.CAPTURA_ACTUAL}"
                )
        except Exception as error:
            mostrar_error("No se pudo capturar", error, self)

    def detener(self) -> None:
        """Cierre la cámara y limpie el preview."""
        if self._camara is not None:
            self._camara.cerrar()
        self._camara = None
        self._actualizar_botones()
        self.estado.set("Cámara detenida.")

    def abrir_imagen(self) -> None:
        """Seleccione una imagen local para probar el flujo sin hardware."""
        ruta = filedialog.askopenfilename(
            parent=self, title="Abrir captura del LCD",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")],
        )
        if not ruta:
            return
        try:
            self.detener()
            imagen = cargar_imagen(ruta)
            guardar_imagen(imagen, config.CAPTURA_ACTUAL)
            self.app.establecer_captura(imagen)
            self.visor.mostrar_imagen(imagen)
            self.estado.set(f"Imagen {imagen.shape[1]}×{imagen.shape[0]}: {ruta}")
        except Exception as error:
            mostrar_error("No se pudo abrir la imagen", error, self)

    def escena_sintetica(self) -> None:
        """Cargue una escena generada con los glifos reales del dataset."""
        try:
            self.detener()
            generador = GeneradorEscenas("termohigrobarometro")
            imagen = generador.frame(momento=0)
            self.app.establecer_captura(imagen)
            self.app.rois_pendientes = generador.rois()
            self.app.notificar_cambio()
            self.visor.mostrar_imagen(imagen)
            self.estado.set(
                "Escena sintética cargada con ROIs de demostración (pestañas 4 y 5)."
            )
        except Exception as error:
            mostrar_error("No se pudo generar la escena", error, self)

    def modo_sensor(self, nocturno: bool) -> None:
        """Aplique el preset día/noche del OV5647 en un hilo aparte."""
        camara = self._camara
        if camara is None or not camara.activa:
            self.estado.set("Inicie la cámara antes de ajustar el sensor.")
            return

        def ajustar():
            try:
                resultado = camara.modo_nocturno(nocturno)
                self._eventos.put((
                    "mensaje",
                    f"Modo {'nocturno' if nocturno else 'diurno'}: "
                    f"{'aplicado' if resultado.soportado else 'el driver ignoró parte de los ajustes'}",
                ))
            except ErrorCamara as error:
                self._eventos.put(("error", str(error)))

        Thread(target=ajustar, daemon=True, name="sensor").start()

    # ---------------------------------------------------------------- bucle
    def _recibir(self) -> None:
        if self._cerrado:
            return
        try:
            while True:
                tipo, dato = self._eventos.get_nowait()
                self._abriendo = False
                if tipo == "camara":
                    self._camara = dato
                    self.estado.set(
                        f"Cámara activa ({dato.backend_usado}) a {dato.resolucion_real[0]}×"
                        f"{dato.resolucion_real[1]}. Pulse Capturar."
                    )
                elif tipo == "error":
                    self.estado.set(dato)
                    mostrar_error("Cámara", dato, self)
                else:
                    self.estado.set(dato)
                self._actualizar_botones()
        except Empty:
            pass
        camara = self._camara
        if camara is not None and camara.activa:
            frame = camara.leer()
            if frame is not None:
                self.visor.mostrar_imagen(frame)
                self.estado.set(
                    f"{camara.backend_usado} · {camara.resolucion_real[0]}×"
                    f"{camara.resolucion_real[1]} · {camara.fps:.1f} FPS · "
                    f"latencia {camara.latencia_ms:.0f} ms"
                )
        self._timer = self.after(60, self._recibir)

    def _actualizar_botones(self) -> None:
        activa = self._camara is not None and self._camara.activa
        self.boton_iniciar.configure(state="disabled" if (activa or self._abriendo) else "normal")
        self.boton_capturar.configure(state="normal" if activa else "disabled")
        self.boton_detener.configure(state="normal" if activa else "disabled")

    def actualizar(self) -> None:
        """Muestre la captura vigente cuando no hay preview de cámara."""
        if (self.app.imagen_capturada is not None
                and (self._camara is None or not self._camara.activa)
                and self.visor.imagen is not self.app.imagen_capturada):
            self.visor.mostrar_imagen(self.app.imagen_capturada)

    def cerrar(self) -> None:
        """Detenga el sondeo y libere la cámara."""
        self._cerrado = True
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        if self._camara is not None:
            self._camara.cerrar()
            self._camara = None
