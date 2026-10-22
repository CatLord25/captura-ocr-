"""Pestaña 6: monitoreo en vivo con HUD y dashboard de alto contraste."""

import time
import tkinter as tk
from tkinter import ttk

import config
from core.camara import ErrorCamara, FuenteImagen, FuenteSintetica, CameraStream
from core.monitor import MonitorEnVivo
from core.perspectiva import Rectificador
from core.roi import cargar_rois
from ui.widgets import VisorImagen, mostrar_error

FUENTES = (
    "Cámara",
    "Escena sintética: termohigrobarómetro",
    "Escena sintética: Balanza",
    "Imagen actual",
)
RESOLUCIONES = ("1280×720", "960×720", "640×480")


class LiveHUDTab(ttk.Frame):
    """Vista de vídeo rectificado con HUD, dashboard y controles rápidos."""

    def __init__(self, parent, app):
        """Construya controles y visor; el monitor se crea al iniciar."""
        super().__init__(parent, padding=8)
        self.app = app
        self._monitor: MonitorEnVivo | None = None
        self._cerrado = False
        self._ultima_marca = 0.0
        self._timer = None
        self._cargando = False

        barra = ttk.Frame(self)
        barra.pack(fill="x")
        ttk.Label(barra, text="Fuente").pack(side="left")
        self.fuente = tk.StringVar(value=FUENTES[0])
        combo_fuente = ttk.Combobox(barra, textvariable=self.fuente, values=FUENTES,
                                    state="readonly", width=30)
        combo_fuente.pack(side="left", padx=(4, 8))
        ttk.Label(barra, text="Resolución").pack(side="left")
        self.resolucion = tk.StringVar(value="1280×720")
        ttk.Combobox(barra, textvariable=self.resolucion, values=RESOLUCIONES,
                     state="readonly", width=10).pack(side="left", padx=(4, 8))
        self.boton_iniciar = ttk.Button(barra, text="Iniciar monitoreo",
                                        command=self.iniciar, width=18)
        self.boton_iniciar.pack(side="left", padx=3)
        self.boton_pausa = ttk.Button(barra, text="Pausar", command=self.alternar,
                                      state="disabled", width=11)
        self.boton_pausa.pack(side="left", padx=3)
        self.boton_detener = ttk.Button(barra, text="Detener", command=self.detener,
                                        state="disabled", width=10)
        self.boton_detener.pack(side="left", padx=3)
        ttk.Button(barra, text="Recargar ROIs", command=self._recargar_rois).pack(
            side="left", padx=8)
        ttk.Button(barra, text="Captura HUD…", command=self._captura).pack(side="right")

        barra2 = ttk.Frame(self)
        barra2.pack(fill="x", pady=(6, 0))
        ttk.Label(barra2, text="Guardar CSV cada").pack(side="left")
        self.intervalo = tk.StringVar(value=config.INTERVALOS_GUARDADO[1][0])
        combo_intervalo = ttk.Combobox(
            barra2, textvariable=self.intervalo, state="readonly", width=12,
            values=[nombre for nombre, _ in config.INTERVALOS_GUARDADO],
        )
        combo_intervalo.pack(side="left", padx=4)
        combo_intervalo.bind("<<ComboboxSelected>>", lambda _: self._aplicar_opciones())
        self.filtro_temporal = tk.BooleanVar(value=config.LIVE_FILTRO_TEMPORAL_POR_DEFECTO)
        ttk.Checkbutton(barra2, text="Filtro temporal (moda N=5)",
                        variable=self.filtro_temporal,
                        command=self._aplicar_opciones).pack(side="left", padx=10)
        self.guardar_csv = tk.BooleanVar(value=config.LIVE_GUARDAR_POR_DEFECTO)
        ttk.Checkbutton(barra2, text="Grabar CSV", variable=self.guardar_csv,
                        command=self._aplicar_opciones).pack(side="left", padx=6)
        self.roi_completa = tk.BooleanVar(value=False)
        ttk.Checkbutton(barra2, text="Tratar ROIs como unidad/texto",
                        variable=self.roi_completa,
                        command=self._aplicar_opciones).pack(side="left", padx=6)

        self.estado = tk.StringVar(value="Inicie el monitoreo en vivo.")
        ttk.Label(self, textvariable=self.estado, wraplength=1100).pack(fill="x", pady=5)
        self.visor = VisorImagen(self, ancho=960, alto=430)
        self.visor.pack(fill="both", expand=True)
        pie = ttk.Frame(self)
        pie.pack(fill="x", pady=(4, 0))
        self.metricas = tk.StringVar(value="FPS — · Latencia — · OCR — · CSV — · Muestras —")
        ttk.Label(pie, textvariable=self.metricas, font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )
        self.estado_csv = tk.StringVar(value="")
        ttk.Label(pie, textvariable=self.estado_csv).pack(side="right")
        self._timer = self.after(60, self._recibir)

    # ---------------------------------------------------------------- acciones
    def _resolucion(self):
        ancho, alto = self.resolucion.get().replace("x", "×").split("×")
        return int(ancho), int(alto)

    def _crear_fuente(self):
        seleccion = self.fuente.get()
        if seleccion == "Cámara":
            fuente = CameraStream(backend=config.CAMARA_BACKEND,
                                  resolucion=self._resolucion())
            rois = cargar_rois()
            rectificador = self.app.rectificador or Rectificador.cargar()
            return fuente, rois, rectificador, "camera"
        if seleccion.startswith("Escena sintética"):
            instrumento = "Balanza" if seleccion.endswith("Balanza") else "termohigrobarometro"
            fuente = FuenteSintetica(instrumento, resolucion=self._resolucion())
            return fuente, fuente.rois(), None, "sintetica"
        fuente = FuenteImagen(config.CAPTURA_ACTUAL)
        rois = list(self.app.rois) or cargar_rois()
        rectificador = self.app.rectificador
        if rectificador is not None:
            esperado = rectificador.tamano
            if (fuente.resolucion_real != rectificador.resolucion_origen
                    and fuente.resolucion_real != esperado):
                rectificador = None
        return fuente, rois, rectificador, "imagen"

    def iniciar(self):
        """Cree la fuente y arranque el monitor en vivo."""
        if self._monitor is not None:
            return
        try:
            fuente, rois, rectificador, clase = self._crear_fuente()
            if not rois:
                raise ValueError(
                    "No hay ROIs definidas. Dibújelas en la pestaña 4 y pulse "
                    "«Recargar ROIs», o use una escena sintética."
                )
            guardar = bool(self.guardar_csv.get())
            monitor = MonitorEnVivo(
                fuente, rois, rectificador=rectificador,
                intervalo=config.intervalo_guardado(self.intervalo.get()),
                guardar=guardar,
                filtro_temporal=bool(self.filtro_temporal.get()),
                roi_completa=bool(self.roi_completa.get()),
            )
            monitor.iniciar()
            self._monitor = monitor
            self.boton_iniciar.configure(state="disabled")
            self.boton_pausa.configure(state="normal", text="Pausar")
            self.boton_detener.configure(state="normal")
            self.estado.set("Monitoreo en vivo iniciado.")
        except Exception as error:
            mostrar_error("No se pudo iniciar el monitoreo", error, self)

    def alternar(self):
        """Pause o reanude el OCR y el CSV."""
        if self._monitor is None:
            return
        self._monitor.alternar()
        pausado = self._monitor.pausado
        self.boton_pausa.configure(text="Reanudar" if pausado else "Pausar")
        self.estado.set("Monitoreo en pausa." if pausado else "Monitoreo en vivo.")

    def detener(self):
        """Detenga el monitor y libere la fuente."""
        monitor, self._monitor = self._monitor, None
        if monitor is not None:
            monitor.cerrar()
        self.boton_iniciar.configure(state="normal")
        self.boton_pausa.configure(state="disabled", text="Pausar")
        self.boton_detener.configure(state="disabled")
        self.estado.set("Monitoreo detenido.")

    def _aplicar_opciones(self):
        monitor = self._monitor
        if monitor is None:
            return
        try:
            monitor.configurar(
                intervalo=config.intervalo_guardado(self.intervalo.get()),
                guardar=bool(self.guardar_csv.get()),
                filtro_temporal=bool(self.filtro_temporal.get()),
                roi_completa=bool(self.roi_completa.get()),
            )
        except Exception as error:
            mostrar_error("Opciones de monitoreo", error, self)

    def _recargar_rois(self):
        """Relea data/rois.json y aplíquelo si el monitor está activo."""
        try:
            rois = list(self.app.rois) or cargar_rois()
            if not rois:
                raise ValueError("data/rois.json está vacío. Defina ROIs en la pestaña 4.")
            if self._monitor is not None:
                self._monitor.actualizar_rois(rois)
            self.estado.set(f"{len(rois)} ROIs cargadas.")
        except Exception as error:
            mostrar_error("No se pudieron recargar las ROIs", error, self)

    def _captura(self):
        """Guarde la vista HUD actual como PNG."""
        monitor = self._monitor
        estado = monitor.estado if monitor is not None else None
        if estado is None or estado.vista is None:
            self.estado.set("No hay vista en vivo para capturar.")
            return
        from core.utils import guardar_imagen

        ruta = config.DIRECTORIO_CAPTURAS / f"hud_{int(time.time())}.png"
        try:
            destino = guardar_imagen(estado.vista, ruta)
            self.estado.set(f"HUD guardado en {destino}")
        except Exception as error:
            mostrar_error("No se pudo guardar el HUD", error, self)

    # ------------------------------------------------------------------ bucle
    def _recibir(self):
        if self._cerrado:
            return
        monitor = self._monitor
        if monitor is not None:
            estado = monitor.estado
            if estado.vista is not None and estado.marca != self._ultima_marca:
                self._ultima_marca = estado.marca
                self.visor.mostrar_imagen(estado.vista)
            self.metricas.set(
                f"FPS {estado.fps:4.1f} · Latencia {estado.latencia_ms:4.0f} ms · "
                f"OCR {estado.ocr_ms:3.0f} ms · {estado.estado_csv} · "
                f"{estado.muestras} muestras"
            )
            self.estado_csv.set("EN PAUSA" if estado.pausado else "")
            if estado.error:
                self.estado.set(estado.error)
            elif estado.mensaje:
                self.estado.set(estado.mensaje)
        self._timer = self.after(60, self._recibir)

    def cerrar(self):
        """Detenga el monitor y cancele el sondeo."""
        self._cerrado = True
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        if self._monitor is not None:
            self._monitor.cerrar()
            self._monitor = None
