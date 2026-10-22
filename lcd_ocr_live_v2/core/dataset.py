"""Plantillas canónicas 32×48, aumentación sintética y bancos por instrumento.

La normalización es **la misma** para las plantillas del dataset y para los
recortes en vivo: binarizado Otsu, polaridad automática, eliminación de ruido,
conservación de proporción (con relleno negro lateral para el dígito ``1``),
reducción a ``TAMANO_PLANTILLA`` y puenteo morfológico en el espacio canónico.
"""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from threading import RLock

import cv2
import numpy as np

import config
from core.utils import a_grises

LOG = logging.getLogger(__name__)

_ETIQUETAS = {
    "coma": ",",
    "punto": ".",
    "signo_menos": "-",
    "porcentaje": "%",
    "porciento": "%",
    "grados_celsius": "°C",
    "grados_fahrenheit": "°F",
    "humedad_relativa": "%RH",
    "texto_data_logger": "Data Logger",
    "texto_humidity": "Humidity",
    "texto_pressure": "Pressure",
    "texto_temperature": "Temperature",
}
DIGITOS = frozenset("0123456789")
PUNTUACION = frozenset({",", "."})
UNIDADES = frozenset({"%", "%RH", "°C", "°F"})
TEXTOS = frozenset({"Data Logger", "Humidity", "Pressure", "Temperature"})
GRUPOS = {
    "digito": DIGITOS,
    "puntuacion": PUNTUACION,
    "menos": frozenset({"-"}),
    "unidad": UNIDADES,
    "texto": TEXTOS,
    "auto": DIGITOS | PUNTUACION | frozenset({"-"}) | UNIDADES | TEXTOS,
}

_CACHE: dict[tuple, "BancoPlantillas"] = {}
_LOCK = RLock()


@dataclass(frozen=True)
class Plantilla:
    """Una muestra etiquetada del dataset y sus variantes aumentadas."""

    etiqueta: str
    imagen: np.ndarray
    archivo: Path
    aumentadas: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class BancoPlantillas:
    """Plantillas de un instrumento listas para template matching y KNN."""

    instrumento: str
    perfil: str
    ruta: Path
    plantillas: tuple[Plantilla, ...] = field(default_factory=tuple)

    def etiquetas(self) -> tuple[str, ...]:
        """Etiquetas únicas ordenadas."""
        return tuple(sorted({plantilla.etiqueta for plantilla in self.plantillas}))

    def por_grupo(self, grupo: str = "auto") -> tuple[Plantilla, ...]:
        """Filtre plantillas por grupo semántico (digito, unidad...)."""
        if grupo not in GRUPOS:
            raise ValueError(f"Grupo de etiquetas desconocido: {grupo!r}.")
        if grupo == "auto":
            return self.plantillas
        permitidas = GRUPOS[grupo]
        return tuple(p for p in self.plantillas if p.etiqueta in permitidas)

    def muestras_aumentadas(self, grupo: str = "auto") -> tuple[np.ndarray, np.ndarray]:
        """Devuelva X (n, alto*ancho) float32/255 e y (n,) de etiquetas."""
        elegidas = self.por_grupo(grupo)
        if not elegidas:
            raise ValueError(f"No hay plantillas del grupo {grupo!r} en {self.instrumento}.")
        filas: list[np.ndarray] = []
        etiquetas: list[str] = []
        for plantilla in elegidas:
            variantes = plantilla.aumentadas or (plantilla.imagen,)
            for variante in variantes:
                filas.append(variante.reshape(-1).astype(np.float32) / 255.0)
                etiquetas.append(plantilla.etiqueta)
        return np.stack(filas), np.asarray(etiquetas, dtype=object)


def etiqueta_de_archivo(archivo: str | Path) -> str:
    """Obtenga la clase del nombre del recorte; rechace nombres sin mapeo.

    Acepta ``digito_0.png``…``digito_9.png``, los símbolos del dataset y las
    muestras extra ``digito_3__02.png``. También acepta ``0.png``…``9.png``.
    """
    nombre = Path(archivo).stem.casefold()
    nombre = re.sub(r"__\d+$", "", nombre)
    coincidencia = re.fullmatch(r"(?:digito_)?([0-9])", nombre)
    if coincidencia:
        return coincidencia.group(1)
    if nombre in _ETIQUETAS:
        return _ETIQUETAS[nombre]
    raise ValueError(
        f"Nombre de recorte sin etiqueta conocida: {Path(archivo).name!r}. "
        "Use digito_0…digito_9, coma, punto, signo_menos, porcentaje, "
        "grados_celsius, grados_fahrenheit, humedad_relativa o texto_"
        "data_logger/humidity/pressure/temperature."
    )


def grupo_de_etiqueta(etiqueta: str) -> str:
    """Clasifique una etiqueta en digito/puntuacion/menos/unidad/texto."""
    for grupo, valores in GRUPOS.items():
        if grupo != "auto" and etiqueta in valores:
            return grupo
    raise ValueError(f"Etiqueta sin grupo conocido: {etiqueta!r}.")


def _polaridad(mascara: np.ndarray) -> np.ndarray:
    """Fije tinta blanca: invierta si el borde es mayoritariamente blanco."""
    borde = np.concatenate((mascara[0], mascara[-1], mascara[:, 0], mascara[:, -1]))
    blancos = float(np.count_nonzero(borde)) / max(1, borde.size)
    return cv2.bitwise_not(mascara) if blancos > 0.5 else mascara


def _desplazar(imagen: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Traslade una imagen binaria canónica sin envolver los bordes."""
    alto, ancho = imagen.shape
    salida = np.zeros_like(imagen)
    x1, x2 = max(0, dx), min(ancho, ancho + dx)
    y1, y2 = max(0, dy), min(alto, alto + dy)
    if x2 <= x1 or y2 <= y1:
        return salida
    salida[y1:y2, x1:x2] = imagen[y1 - dy:y2 - dy, x1 - dx:x2 - dx]
    return salida


def _puente_canonico(binaria: np.ndarray, perfil: str) -> np.ndarray:
    """Funda segmentos próximos en el espacio canónico 32×48.

    En siete segmentos se usa una elipse vertical (3×5); en LCD una (2×3)
    para unir trazos discontinuos sin deformar el glifo.
    """
    ancho, alto = config.PUENTE_7SEG if perfil == "7seg" else config.PUENTE_LCD
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(ancho), int(alto)))
    return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, nucleo)


def normalizar_caracter(
    imagen: np.ndarray,
    tamano: tuple[int, int] | None = None,
    perfil: str = "lcd",
    puentear: bool = True,
) -> np.ndarray:
    """Convierta un recorte en una plantilla binaria canónica con proporción.

    - Elimina componentes menores al 3 % del mayor (ruido del sensor).
    - Si el glifo es estrecho (``AR < 0.38``), añade relleno negro lateral con
      ``cv2.copyMakeBorder`` antes de escalar: un ``1`` nunca se convierte en
      una barra blanca ancha que se confunda con ``0``/``8``.
    - Escala con ``INTER_AREA`` conservando proporción y centra en negro.
    """
    ancho, alto = (int(valor) for valor in (tamano or config.TAMANO_PLANTILLA))
    if min(ancho, alto) < 8:
        raise ValueError("TAMANO_PLANTILLA debe tener al menos 8×8 píxeles.")
    gris = a_grises(imagen)
    if float(np.ptp(gris)) < 3:
        raise ValueError("El recorte no tiene contraste suficiente para OCR.")
    suave = cv2.GaussianBlur(gris, (int(config.DESENFOQUE_KERNEL),) * 2, 0)
    _, binaria = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = _polaridad(binaria)
    cantidad, componentes, estadisticas, _ = cv2.connectedComponentsWithStats(binaria, 8)
    if cantidad <= 1:
        raise ValueError("No se encontró primer plano en el recorte.")
    areas = estadisticas[1:, cv2.CC_STAT_AREA]
    minimo = max(1, int(np.ceil(areas.max() * config.NORMALIZACION_AREA_RELATIVA_MINIMA)))
    conservar = np.concatenate(([False], areas >= minimo))
    mascara = np.where(conservar[componentes], 255, 0).astype(np.uint8)
    puntos = cv2.findNonZero(mascara)
    if puntos is None:
        raise ValueError("No se encontró un carácter después de quitar ruido.")
    x, y, w, h = cv2.boundingRect(puntos)
    contenido = mascara[y:y + h, x:x + w]
    # Relleno lateral explícito para glifos estrechos (dígito 1).
    if w / max(1, h) < config.AR_ESTRECHO:
        objetivo = max(w, int(round(h * config.AR_ESTRECHO_OBJETIVO)))
        relleno = objetivo - w
        izquierda = relleno // 2
        contenido = cv2.copyMakeBorder(
            contenido, 0, 0, izquierda, relleno - izquierda,
            cv2.BORDER_CONSTANT, value=0,
        )
    alto_contenido, ancho_contenido = contenido.shape
    margen = max(1, round(min(ancho, alto) * config.NORMALIZACION_MARGEN_RELATIVO))
    escala = min(
        (ancho - 2 * margen) / ancho_contenido,
        (alto - 2 * margen) / alto_contenido,
    )
    destino = (
        max(1, round(ancho_contenido * escala)),
        max(1, round(alto_contenido * escala)),
    )
    reducido = cv2.resize(contenido, destino, interpolation=cv2.INTER_AREA)
    _, reducido = cv2.threshold(reducido, 127, 255, cv2.THRESH_BINARY)
    salida = np.zeros((alto, ancho), dtype=np.uint8)
    izquierda = (ancho - destino[0]) // 2
    superior = (alto - destino[1]) // 2
    salida[superior:superior + destino[1], izquierda:izquierda + destino[0]] = reducido
    if puentear:
        salida = _puente_canonico(salida, perfil)
        _, salida = cv2.threshold(salida, 127, 255, cv2.THRESH_BINARY)
    return salida


def aumentar_plantilla(imagen: np.ndarray, perfil: str = "lcd") -> tuple[np.ndarray, ...]:
    """Genere variantes con desplazamientos, erosión y dilatación 3×3.

    Todo se calcula en memoria al arrancar (< 10 ms por banco) y alimenta al
    KNN nativo de OpenCV sin escribir archivos en el dataset original.
    """
    variantes: list[np.ndarray] = [imagen]
    desplazamientos = [(dx, 0) for dx in config.AUMENTO_DESPLAZAMIENTOS if dx]
    desplazamientos += [(0, dy) for dy in config.AUMENTO_DESPLAZAMIENTOS if dy]
    for dx, dy in desplazamientos:
        variantes.append(_desplazar(imagen, dx, dy))
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    variantes.append(cv2.erode(imagen, nucleo, iterations=1))
    variantes.append(cv2.dilate(imagen, nucleo, iterations=1))
    unicas: list[np.ndarray] = []
    vistos: set[bytes] = set()
    for variante in variantes:
        variante = np.ascontiguousarray(np.where(variante > 0, 255, 0).astype(np.uint8))
        clave = variante.tobytes()
        if clave in vistos or cv2.countNonZero(variante) == 0:
            continue
        vistos.add(clave)
        variante.flags.writeable = False
        unicas.append(variante)
    return tuple(unicas)


def cargar_banco(instrumento: str) -> BancoPlantillas:
    """Cargue y normalice todas las muestras del instrumento (con caché)."""
    canonico = config.instrumento_canonico(instrumento)
    perfil = config.perfil_de_instrumento(canonico)
    ruta = config.resolver_ruta_dataset(canonico)
    tamano = tuple(config.TAMANO_PLANTILLA)
    clave = (str(ruta), tamano, perfil)
    with _LOCK:
        if clave in _CACHE:
            return _CACHE[clave]
        archivos = sorted(
            (archivo for archivo in ruta.rglob("*")
             if archivo.is_file() and archivo.suffix.casefold() in config.EXTENSIONES_IMAGEN),
            key=lambda archivo: str(archivo.relative_to(ruta)).casefold(),
        )
        if not archivos:
            raise FileNotFoundError(f"No hay recortes de imagen en {ruta}.")
        plantillas: list[Plantilla] = []
        for archivo in archivos:
            etiqueta = etiqueta_de_archivo(archivo)
            try:
                datos = np.fromfile(archivo, dtype=np.uint8)
                imagen = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
                if imagen is None:
                    raise ValueError("OpenCV no pudo decodificar la imagen")
                normalizada = normalizar_caracter(imagen, tamano, perfil)
            except (OSError, ValueError, cv2.error) as error:
                raise ValueError(f"Recorte inválido {archivo}: {error}") from error
            normalizada.flags.writeable = False
            plantillas.append(Plantilla(
                etiqueta=etiqueta,
                imagen=normalizada,
                archivo=archivo,
                aumentadas=aumentar_plantilla(normalizada, perfil),
            ))
        banco = BancoPlantillas(canonico, perfil, ruta, tuple(plantillas))
        _CACHE[clave] = banco
        conteo: dict[str, int] = {}
        for plantilla in plantillas:
            conteo[plantilla.etiqueta] = conteo.get(plantilla.etiqueta, 0) + 1
        LOG.info(
            "Banco %s (%s): %d recortes, %d clases, %d muestras KNN; ruta=%s",
            canonico, perfil, len(plantillas), len(conteo),
            sum(len(p.aumentadas) for p in plantillas), ruta,
        )
        for etiqueta, cantidad in sorted(conteo.items()):
            LOG.debug("  Clase %r: %d recorte(s)", etiqueta, cantidad)
        return banco


def limpiar_cache() -> None:
    """Vacíe los bancos cargados para releer cambios en disco."""
    with _LOCK:
        _CACHE.clear()


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verifique los recortes y sus clases.")
    parser.add_argument("instrumento", nargs="?", help="Balanza o Termohigrobarómetro.")
    argumentos = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    instrumentos = ([argumentos.instrumento] if argumentos.instrumento
                    else ["Balanza", "Termohigrobarómetro"])
    try:
        for instrumento in instrumentos:
            banco = cargar_banco(instrumento)
            print(f"\n{banco.instrumento} [{banco.perfil}] — {banco.ruta}")
            for plantilla in banco.plantillas:
                print(f"  {plantilla.archivo.name:26s} -> {plantilla.etiqueta!r:18s}"
                      f" {plantilla.imagen.shape[1]}×{plantilla.imagen.shape[0]}"
                      f"  variantes={len(plantilla.aumentadas)}")
    except (FileNotFoundError, ValueError, OSError) as error:
        LOG.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    _main()
