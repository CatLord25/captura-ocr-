"""Lanzador directo del monitoreo en vivo con HUD de OpenCV.

Uso típico en la Raspberry Pi (sin Tkinter):

    python main_live_hud.py --fuente camara --escala 0.8
    python main_live_hud.py --fuente sintetica:termohigrobarometro
    python main_live_hud.py --headless --intervalo 5s

Teclas en la ventana: q/ESC salir · espacio pausar · t filtro temporal ·
g CSV · 1-5 intervalo · p captura del HUD.
"""

import argparse
import logging
import sys
import time

import cv2

import config
from core.camara import ErrorCamara, abrir_fuente
from core.monitor import MonitorEnVivo
from core.perspectiva import Rectificador
from core.roi import cargar_rois
from core.utils import configurar_logging, guardar_imagen

LOG = logging.getLogger(__name__)
VENTANA = "LCD OCR LIVE v2"


def _resolucion(texto: str) -> tuple[int, int]:
    """Acepte '1280x720' o '1280×720'."""
    partes = texto.lower().replace("×", "x").split("x")
    if len(partes) != 2:
        raise argparse.ArgumentTypeError("Resolución con formato ANCHOxALTO.")
    try:
        ancho, alto = int(partes[0]), int(partes[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("Resolución numérica ANCHOxALTO.") from error
    if ancho < 160 or alto < 120:
        raise argparse.ArgumentTypeError("Resolución demasiado pequeña.")
    return ancho, alto


def _intervalo(texto: str) -> float:
    """Acepte '1s', '5', '2.5' o el nombre del selector."""
    limpio = texto.strip().casefold()
    for etiqueta, segundos in config.INTERVALOS_GUARDADO:
        if limpio == etiqueta.casefold():
            return segundos
    try:
        return float(limpio.rstrip("s"))
    except ValueError as error:
        nombres = ", ".join(etiqueta for etiqueta, _ in config.INTERVALOS_GUARDADO)
        raise argparse.ArgumentTypeError(f"Intervalo inválido; use {nombres} o segundos.") from error


def _argumentos(argv=None):
    parser = argparse.ArgumentParser(
        description="LCD OCR LIVE v2 — HUD en vivo para balanza y termohigrobarómetro."
    )
    parser.add_argument("--fuente", default="camara",
                        help="camara | imagen:RUTA | video:RUTA | sintetica[:instrumento]")
    parser.add_argument("--indice", type=int, default=config.CAMARA_INDICE,
                        help="Índice de cámara (0 por defecto).")
    parser.add_argument("--resolucion", type=_resolucion, default=config.CAMARA_RESOLUCION,
                        help="Resolución de captura, p. ej. 1280x720.")
    parser.add_argument("--fps", type=int, default=config.CAMARA_FPS)
    parser.add_argument("--backend", default=config.CAMARA_BACKEND,
                        help="auto, v4l2, gstreamer, picamera2, dshow, msmf, any")
    parser.add_argument("--rois", default=str(config.ARCHIVO_ROIS),
                        help="Archivo JSON de ROIs (por defecto data/rois.json).")
    parser.add_argument("--sin-perspectiva", action="store_true",
                        help="Ignore data/perspectiva.json y use el frame original.")
    parser.add_argument("--intervalo", type=_intervalo,
                        default=config.INTERVALO_GUARDADO_POR_DEFECTO,
                        help="Cadencia de guardado CSV: 'Tiempo real', 1s, 5s, 10s, 60s.")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--guardar", dest="guardar", action="store_true", default=None,
                       help="Fuerce el registro CSV (por defecto, el de config.py).")
    grupo.add_argument("--sin-guardar", dest="guardar", action="store_false",
                       help="Desactive el registro CSV.")
    parser.add_argument("--sin-filtro", dest="filtro", action="store_false", default=None,
                        help="Desactive el filtro temporal de moda.")
    parser.add_argument("--escala", type=float, default=0.8,
                        help="Escala de la ventana (1.0 = tamaño nativo).")
    parser.add_argument("--pantalla-completa", action="store_true")
    parser.add_argument("--captura-inicial", metavar="RUTA",
                        help="Guarde una captura inmediata y continúe.")
    parser.add_argument("--headless", action="store_true",
                        help="Sin ventana: imprime lecturas y guarda CSV (SSH).")
    parser.add_argument("--duracion", type=float, default=0.0,
                        help="Segundos de ejecución en modo headless (0 = indefinido).")
    return parser.parse_args(argv)


def _resumen_teclas() -> str:
    return "q/ESC salir · espacio pausa · t filtro · g CSV · 1-5 intervalo · p captura"


def _guardar_captura(monitor: MonitorEnVivo) -> None:
    estado = monitor.estado
    if estado.vista is None:
        print("No hay vista para capturar.", flush=True)
        return
    ruta = config.DIRECTORIO_CAPTURAS / f"hud_{int(time.time())}.png"
    destino = guardar_imagen(estado.vista, ruta)
    print(f"HUD guardado en {destino}", flush=True)


def _headless(monitor: MonitorEnVivo, duracion: float) -> int:
    print("Modo headless: Ctrl+C para terminar.", flush=True)
    inicio = time.monotonic()
    ultimo = 0.0
    try:
        while True:
            time.sleep(0.1)
            ahora = time.monotonic()
            if ahora - ultimo < 2.0:
                continue
            ultimo = ahora
            estado = monitor.estado
            valores = " | ".join(
                f"{lectura.roi}={lectura.texto}{lectura.unidad}" for lectura in estado.lecturas
            )
            print(
                f"{time.strftime('%H:%M:%S')} FPS={estado.fps:4.1f} "
                f"lat={estado.latencia_ms:4.0f}ms ocr={estado.ocr_ms:3.0f}ms "
                f"csv={estado.estado_csv} filas={estado.muestras} {valores}",
                flush=True,
            )
            if estado.error:
                print(f"ERROR: {estado.error}", flush=True)
            if duracion > 0 and ahora - inicio >= duracion:
                break
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.", flush=True)
    return 0


def main(argv=None) -> int:
    """Punto de entrada del HUD directo."""
    argumentos = _argumentos(argv)
    configurar_logging()
    try:
        fuente = abrir_fuente(
            argumentos.fuente, indice=argumentos.indice,
            resolucion=argumentos.resolucion, fps=argumentos.fps,
            backend=argumentos.backend,
        )
    except ErrorCamara as error:
        print(f"No se pudo crear la fuente: {error}", file=sys.stderr)
        return 2
    rois = fuente.rois() if argumentos.fuente.casefold().startswith("sintetica") \
        else cargar_rois(argumentos.rois)
    if not rois:
        print(
            "No hay ROIs definidas.\n"
            f"Defínalas en main.py (pestaña 4) o en {argumentos.rois}, o use "
            "--fuente sintetica:termohigrobarometro.",
            file=sys.stderr,
        )
        return 2
    rectificador = None
    if not argumentos.sin_perspectiva:
        try:
            rectificador = Rectificador.cargar()
        except ValueError as error:
            print(f"Aviso: {error}. Se usará el frame sin rectificar.", file=sys.stderr)
    monitor = MonitorEnVivo(
        fuente, rois, rectificador=rectificador,
        intervalo=argumentos.intervalo,
        guardar=config.LIVE_GUARDAR_POR_DEFECTO if argumentos.guardar is None
        else bool(argumentos.guardar),
        filtro_temporal=config.LIVE_FILTRO_TEMPORAL_POR_DEFECTO if argumentos.filtro is None
        else bool(argumentos.filtro),
        ruta_csv=config.ARCHIVO_LECTURAS,
        componer_hud=not argumentos.headless,
    )
    print(
        f"Fuente: {argumentos.fuente} · ROIs: {len(rois)} · "
        f"perspectiva: {'sí' if rectificador else 'no'} · "
        f"CSV: {config.ARCHIVO_LECTURAS}",
        flush=True,
    )
    monitor.iniciar()
    try:
        if argumentos.headless:
            return _headless(monitor, argumentos.duracion)
        try:
            cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
            if argumentos.pantalla_completa:
                cv2.setWindowProperty(VENTANA, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN)
        except cv2.error as error:
            print(
                f"No hay servidor gráfico para abrir la ventana ({error}). "
                "Use --headless o ejecute sobre el escritorio de la Raspberry Pi.",
                file=sys.stderr,
            )
            return 3
        print(_resumen_teclas(), flush=True)
        ultimo_dibujo = 0.0
        estado = monitor.estado
        while True:
            estado = monitor.estado
            if estado.vista is not None and estado.marca != ultimo_dibujo:
                ultimo_dibujo = estado.marca
                vista = estado.vista
                if argumentos.escala and argumentos.escala != 1.0:
                    vista = cv2.resize(
                        vista, None, fx=argumentos.escala, fy=argumentos.escala,
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(VENTANA, vista)
            tecla = cv2.waitKey(15) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                monitor.alternar()
                print("Pausado." if monitor.pausado else "Reanudado.", flush=True)
            elif tecla == ord("t"):
                activo = monitor.alternar_filtro()
                print(f"Filtro temporal {'activado' if activo else 'desactivado'}.",
                      flush=True)
            elif tecla == ord("g"):
                activo = monitor.alternar_guardado()
                print(f"CSV {'activado' if activo else 'desactivado'}.", flush=True)
            elif ord("1") <= tecla <= ord("5"):
                indice = tecla - ord("1")
                etiqueta, segundos = config.INTERVALOS_GUARDADO[indice]
                monitor.configurar(intervalo=segundos)
                print(f"Intervalo CSV: {etiqueta}.", flush=True)
            elif tecla == ord("p"):
                _guardar_captura(monitor)
            if estado.error:
                print(f"ERROR: {estado.error}", flush=True)
    except KeyboardInterrupt:
        print("\nInterrumpido.", flush=True)
    finally:
        monitor.cerrar()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
