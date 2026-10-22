"""Motor de reconocimiento triple en OpenCV puro.

1. ``cv2.matchTemplate`` con ``TM_CCOEFF_NORMED`` y ``TM_SQDIFF_NORMED``.
2. ``cv2.ml.KNearest_create`` sobre variantes aumentadas en memoria, con
   respaldo KNN en numpy para builds de OpenCV 5 sin módulo ``ml``.
3. Desempate topológico de siete segmentos (firmas A..G aprendidas).
"""

from dataclasses import dataclass, field
import logging
from threading import RLock

import cv2
import numpy as np

import config
from core.dataset import BancoPlantillas, cargar_banco, grupo_de_etiqueta, normalizar_caracter
from core.siete_segmentos import (
    DecodificadorSieteSegmentos,
    medir_firma,
    medir_presencia,
)
from core.utils import a_grises

LOG = logging.getLogger(__name__)
_MOTORES: dict[str, "MotorReconocimiento"] = {}
_LOCK = RLock()
_AVISO_ML = False


@dataclass(frozen=True)
class ResultadoCaracter:
    """Etiqueta aceptada, confianza, candidato y evidencia por nivel."""

    etiqueta: str
    confianza: float
    candidato: str
    metodo: str
    grupo: str = "auto"
    desglose: tuple[tuple[str, float], ...] = ()
    detalle: dict = field(default_factory=dict)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    """Raíz del error cuadrático medio entre dos vectores normalizados."""
    return float(np.sqrt(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)))


class ModeloKnn:
    """KNN sobre muestras aumentadas; usa ``cv2.ml`` cuando está disponible."""

    def __init__(self, muestras: list[tuple[str, np.ndarray]], k: int) -> None:
        """Entrene el modelo en memoria y calcule la escala de distancias."""
        if not muestras:
            raise ValueError("No hay muestras para entrenar el KNN.")
        self._etiquetas = sorted({etiqueta for etiqueta, _ in muestras})
        self._indices = {etiqueta: indice for indice, etiqueta in enumerate(self._etiquetas)}
        self._X = np.stack(
            [imagen.reshape(-1).astype(np.float32) / 255.0 for _, imagen in muestras]
        )
        self._y = np.asarray([etiqueta for etiqueta, _ in muestras], dtype=object)
        conteo = {etiqueta: int(np.sum(self._y == etiqueta)) for etiqueta in self._etiquetas}
        self.k = max(1, min(int(k), min(conteo.values())))
        self._dim = self._X.shape[1]
        self._escala = self._escala_interclase(muestras)
        self._nativo = self._entrenar_nativo()

    @staticmethod
    def _escala_interclase(muestras: list[tuple[str, np.ndarray]]) -> float:
        """Distancia RMS típica entre clases distintas (calibración local)."""
        vistas: dict[str, np.ndarray] = {}
        for etiqueta, imagen in muestras:
            vistas.setdefault(etiqueta, imagen.astype(np.float32) / 255.0)
        vectores = list(vistas.values())
        distancias = [
            _rms(vectores[i], vectores[j])
            for i in range(len(vectores)) for j in range(i + 1, len(vectores))
        ]
        if not distancias:
            return 0.15
        return max(0.02, float(np.median(distancias)))

    def _entrenar_nativo(self):
        """Entrene ``cv2.ml.KNearest``; devuelva None si el build no lo trae."""
        global _AVISO_ML
        if not hasattr(cv2, "ml") or not hasattr(cv2.ml, "KNearest_create"):
            if not _AVISO_ML:
                LOG.warning(
                    "OpenCV %s no incluye cv2.ml.KNearest; se usa KNN numpy equivalente.",
                    cv2.__version__,
                )
                _AVISO_ML = True
            return None
        try:
            respuestas = np.asarray(
                [self._indices[etiqueta] for etiqueta in self._y], dtype=np.float32
            ).reshape(-1, 1)
            modelo = cv2.ml.KNearest_create()
            modelo.setIsClassifier(True)
            modelo.train(self._X, cv2.ml.ROW_SAMPLE, respuestas)
            LOG.debug("KNN nativo cv2.ml entrenado con %s muestras.", len(self._y))
            return modelo
        except cv2.error as error:  # pragma: no cover - depende del build
            LOG.warning("No se pudo entrenar cv2.ml.KNearest (%s); se usa numpy.", error)
            return None

    @property
    def usa_cv2_ml(self) -> bool:
        """Indique si el modelo nativo de OpenCV está activo."""
        return self._nativo is not None

    def etiqueta_nativa(self, vector: np.ndarray) -> str | None:
        """Predicción del KNN nativo de OpenCV, si está disponible."""
        if self._nativo is None:
            return None
        consulta = vector.reshape(1, -1).astype(np.float32) / 255.0
        _, resultados, _, _ = self._nativo.findNearest(consulta, self.k)
        indice = int(resultados[0, 0])
        if 0 <= indice < len(self._etiquetas):
            return self._etiquetas[indice]
        return None

    def puntajes(self, vector: np.ndarray) -> dict[str, float]:
        """Puntaje 0..1 por etiqueta según la muestra más cercana de cada una."""
        consulta = vector.reshape(-1).astype(np.float32) / 255.0
        distancias = np.sqrt(np.mean((self._X - consulta) ** 2, axis=1))
        minimos: dict[str, float] = {}
        for etiqueta, distancia in zip(self._y, distancias):
            actual = minimos.get(etiqueta)
            if actual is None or distancia < actual:
                minimos[etiqueta] = float(distancia)
        return {
            etiqueta: float(np.exp(-((distancia / self._escala) ** 2) * np.log(2.0)))
            for etiqueta, distancia in minimos.items()
        }


class MotorReconocimiento:
    """Combine plantillas, KNN y topología para un instrumento."""

    def __init__(self, instrumento: str, banco: BancoPlantillas | None = None) -> None:
        """Cargue el banco del instrumento y prepare los modelos bajo demanda."""
        self.instrumento = config.instrumento_canonico(instrumento)
        self.banco = banco or cargar_banco(self.instrumento)
        self.perfil = self.banco.perfil
        self._modelos: dict[str, ModeloKnn] = {}
        self._decodificador: DecodificadorSieteSegmentos | None = None
        self._lock = RLock()

    @property
    def decodificador(self) -> DecodificadorSieteSegmentos | None:
        """Decodificador 7seg construido una sola vez a partir del banco."""
        with self._lock:
            if self._decodificador is None:
                try:
                    self._decodificador = DecodificadorSieteSegmentos.desde_banco(self.banco)
                except ValueError:
                    self._decodificador = None
            return self._decodificador

    def _modelo(self, grupo: str) -> ModeloKnn:
        """Entrene (o reutilice) el KNN del grupo solicitado."""
        with self._lock:
            if grupo not in self._modelos:
                muestras: list[tuple[str, np.ndarray]] = []
                for plantilla in self.banco.por_grupo(grupo):
                    variantes = plantilla.aumentadas or (plantilla.imagen,)
                    muestras.extend((plantilla.etiqueta, variante) for variante in variantes)
                if not muestras:
                    raise ValueError(f"No hay muestras del grupo {grupo!r}.")
                self._modelos[grupo] = ModeloKnn(muestras, config.K_VECINOS)
            return self._modelos[grupo]

    def _puntajes_plantilla(self, normalizada: np.ndarray, grupo: str) -> dict[str, float]:
        """Nivel 1: similitud combinada CCOEFF + SQDIFF por etiqueta."""
        margen = config.DESPLAZAMIENTO_PLANTILLA
        ampliada = cv2.copyMakeBorder(
            normalizada, margen, margen, margen, margen,
            cv2.BORDER_CONSTANT, value=0,
        )
        puntajes: dict[str, float] = {}
        for plantilla in self.banco.por_grupo(grupo):
            if float(np.std(plantilla.imagen)) < 1:
                continue
            ccoeff = float(np.max(cv2.matchTemplate(
                ampliada, plantilla.imagen, cv2.TM_CCOEFF_NORMED)))
            cuadrado = float(np.min(cv2.matchTemplate(
                ampliada, plantilla.imagen, cv2.TM_SQDIFF_NORMED)))
            score = 0.6 * max(0.0, ccoeff) + 0.4 * (1.0 - max(0.0, cuadrado))
            puntajes[plantilla.etiqueta] = max(
                puntajes.get(plantilla.etiqueta, 0.0), float(np.clip(score, 0.0, 1.0))
            )
        return puntajes

    def _puntajes_knn(self, normalizada: np.ndarray, grupo: str) -> dict[str, float]:
        """Nivel 2: KNN (``cv2.ml`` nativo o respaldo numpy)."""
        modelo = self._modelo(grupo)
        return modelo.puntajes(normalizada)

    def _pesos(self, grupo: str, con_topologia: bool) -> dict[str, float]:
        """Pesos de fusión según grupo y perfil del instrumento."""
        if grupo != "digito" or not con_topologia:
            return dict(config.PESOS_FUSION_UNIDAD)
        if self.perfil == "7seg":
            return dict(config.PESOS_FUSION)
        pesos = dict(config.PESOS_FUSION)
        pesos["topologia"] = 0.12
        pesos["template"] = 0.60
        pesos["knn"] = 0.28
        return pesos

    def reconocer(
        self,
        imagen: np.ndarray,
        grupo: str = "auto",
        umbral: float | None = None,
    ) -> ResultadoCaracter:
        """Reconozca un recorte y devuelva la etiqueta con su confianza."""
        if grupo not in {"digito", "puntuacion", "menos", "unidad", "texto", "auto"}:
            raise ValueError(f"Grupo de reconocimiento desconocido: {grupo!r}.")
        limite = float(config.UMBRAL_CONFIANZA if umbral is None else umbral)
        if not np.isfinite(limite) or not 0.0 <= limite <= 1.0:
            raise ValueError("El umbral de confianza debe estar entre 0 y 1.")
        if not isinstance(imagen, np.ndarray) or imagen.size == 0:
            raise ValueError("El recorte para reconocer debe ser una matriz no vacía.")
        try:
            normalizada = normalizar_caracter(imagen, config.TAMANO_PLANTILLA, self.perfil)
        except ValueError as error:
            if "contraste" in str(error) or "primer plano" in str(error):
                return ResultadoCaracter("?", 0.0, "", "sin_tinta", grupo)
            raise
        if cv2.countNonZero(normalizada) < 4:
            return ResultadoCaracter("?", 0.0, "", "sin_tinta", grupo)
        # Restrinja el grupo si el dataset no tiene esa clase (p. ej. Balanza sin textos).
        disponible = self.banco.por_grupo(grupo)
        if not disponible:
            grupo = "auto"
        plantilla = self._puntajes_plantilla(normalizada, grupo)
        knn = self._puntajes_knn(normalizada, grupo)
        decodificador = self.decodificador
        topologia: dict[str, float] = {}
        firma = None
        presencia: dict[str, bool] = {}
        if grupo == "digito" and decodificador is not None:
            firma = medir_firma(normalizada)
            topologia = decodificador.puntajes(firma)
            presencia = medir_presencia(normalizada)
        pesos = self._pesos(grupo, bool(topologia))
        niveles = {"template": plantilla, "knn": knn, "topologia": topologia}
        fusionados: dict[str, float] = {}
        for etiqueta in {clave for tabla in niveles.values() for clave in tabla}:
            total = 0.0
            usado = 0.0
            for nivel, tabla in niveles.items():
                if etiqueta in tabla and pesos.get(nivel, 0.0) > 0:
                    total += pesos[nivel] * tabla[etiqueta]
                    usado += pesos[nivel]
            if usado > 0:
                fusionados[etiqueta] = total / usado
        if not fusionados:
            return ResultadoCaracter("?", 0.0, "", "sin_candidatos", grupo)
        ordenados = sorted(fusionados.items(), key=lambda par: par[1], reverse=True)
        detalle: dict = {
            "template": {k: round(v, 4) for k, v in sorted(plantilla.items(), key=lambda p: -p[1])[:4]},
            "knn": {k: round(v, 4) for k, v in sorted(knn.items(), key=lambda p: -p[1])[:4]},
            "knn_cv2_ml": getattr(self._modelo(grupo), "usa_cv2_ml", False),
            "topologia": {k: round(v, 4) for k, v in sorted(topologia.items(), key=lambda p: -p[1])[:4]},
            "presencia": presencia,
        }
        if firma is not None:
            detalle["firma"] = {k: round(v, 3) for k, v in firma.como_dict().items()}
        # Desempate topológico cuando template y KNN quedan casi empatados.
        if len(ordenados) >= 2 and topologia and ordenados[0][1] - ordenados[1][1] < 0.08:
            primero, segundo = ordenados[0][0], ordenados[1][0]
            ganador, margen, segmentos = decodificador.desempatar(firma, primero, segundo)
            detalle["desempate"] = {
                "par": f"{primero}/{segundo}", "ganador": ganador,
                "margen": round(margen, 3), "segmentos": list(segmentos),
            }
            if margen >= 0.35:
                fusionados[ganador] = max(
                    fusionados.get(ganador, 0.0), fusionados[primero] + 0.02 * margen + 0.01
                )
                ordenados = sorted(fusionados.items(), key=lambda par: par[1], reverse=True)
        candidato, confianza = ordenados[0]
        confianza = float(np.clip(confianza, 0.0, 1.0))
        metodo = "+".join(
            nombre for nombre, tabla in niveles.items() if tabla and pesos.get(nombre, 0) > 0
        )
        etiqueta = candidato if confianza >= limite else "?"
        return ResultadoCaracter(
            etiqueta=etiqueta,
            confianza=confianza,
            candidato=candidato,
            metodo=metodo,
            grupo=grupo,
            desglose=tuple((clave, round(valor, 4)) for clave, valor in ordenados[:4]),
            detalle=detalle,
        )

    def reconocer_segmentos(
        self, segmentos, umbral: float | None = None
    ) -> list[ResultadoCaracter]:
        """Reconozca cada segmento usando su clase geométrica como grupo."""
        resultados = []
        for segmento in segmentos:
            clase = getattr(segmento, "clase", "digito")
            grupo = {"puntuacion": "puntuacion", "menos": "menos"}.get(clase, "digito")
            resultados.append(self.reconocer(segmento.imagen, grupo, umbral))
        return resultados


def obtener_motor(instrumento: str) -> MotorReconocimiento:
    """Devuelva (y cachee) el motor del instrumento."""
    canonico = config.instrumento_canonico(instrumento)
    with _LOCK:
        if canonico not in _MOTORES:
            _MOTORES[canonico] = MotorReconocimiento(canonico)
        return _MOTORES[canonico]


def reconocer_caracter(
    imagen: np.ndarray, instrumento: str, grupo: str = "auto", umbral: float | None = None
) -> ResultadoCaracter:
    """Atajo compatible: reconozca un recorte con el motor del instrumento."""
    return obtener_motor(instrumento).reconocer(imagen, grupo, umbral)


def reconocer_segmentos(
    segmentos, instrumento: str, umbral: float | None = None
) -> list[ResultadoCaracter]:
    """Atajo compatible: reconozca una lista de segmentos en orden."""
    if not segmentos:
        raise ValueError("No hay caracteres segmentados para reconocer.")
    return obtener_motor(instrumento).reconocer_segmentos(segmentos, umbral)


def reconocer_roi_completa(
    imagen: np.ndarray, instrumento: str, grupo: str = "unidad", umbral: float | None = None
) -> ResultadoCaracter:
    """Reconozca una unidad o texto comparando la ROI entera."""
    return obtener_motor(instrumento).reconocer(a_grises(imagen), grupo, umbral)


def limpiar_cache_reconocimiento() -> None:
    """Vacíe motores y modelos cacheados (útil tras recargar plantillas)."""
    with _LOCK:
        _MOTORES.clear()
