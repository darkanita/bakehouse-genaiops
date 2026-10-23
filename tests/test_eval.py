"""Pruebas del set de evaluación y de la lógica de la puerta de calidad."""
from pathlib import Path

import pandas as pd
import pytest

from utils_eval import (
    MINIMO_PREGUNTAS,
    TIPOS_VALIDOS,
    construir_datos,
    decidir_aprobacion,
    filas_pendientes,
    metrica_media,
    promedio_ponderado,
    validar_set,
)

RUTA_SET = Path(__file__).resolve().parents[1] / "eval" / "preguntas.csv"


def test_el_set_del_repo_es_valido():
    errores = validar_set(pd.read_csv(RUTA_SET))
    assert errores == [], errores


def _set_valido():
    filas = []
    for i in range(MINIMO_PREGUNTAS):
        tipo = sorted(TIPOS_VALIDOS)[i % len(TIPOS_VALIDOS)]
        filas.append({"pregunta": f"Pregunta {i}", "respuesta_esperada": f"Respuesta {i}", "tipo": tipo})
    return pd.DataFrame(filas)


def test_detecta_columnas_faltantes():
    assert validar_set(pd.DataFrame({"pregunta": ["a"]})) != []


def test_detecta_pocas_preguntas():
    assert any("al menos" in e for e in validar_set(_set_valido().head(5)))


def test_detecta_tipo_invalido():
    df = _set_valido()
    df.loc[0, "tipo"] = "otro"
    assert any("no válidos" in e for e in validar_set(df))


def test_detecta_duplicados():
    df = _set_valido()
    df.loc[1, "pregunta"] = df.loc[0, "pregunta"].upper()
    assert any("duplicadas" in e for e in validar_set(df))


def test_detecta_celdas_vacias():
    df = _set_valido()
    df.loc[0, "respuesta_esperada"] = "  "
    assert any("vacías" in e for e in validar_set(df))


def test_cuenta_pendientes():
    df = _set_valido()
    df.loc[0, "respuesta_esperada"] = "[COMPLETAR según el manual]"
    assert filas_pendientes(df) == 1


def test_construir_datos_formato_mlflow():
    datos = construir_datos(_set_valido().head(1))
    assert datos == [
        {"inputs": {"pregunta": "Pregunta 0"}, "expectations": {"expected_response": "Respuesta 0"}}
    ]


def test_metrica_media():
    assert metrica_media({"correctness/mean": 0.8, "safety/mean": 1.0}, "correctness") == 0.8
    assert metrica_media({"Correctness/Mean": 0.5}, "correctness") == 0.5
    assert metrica_media({"safety/mean": 1.0}, "correctness") is None


def test_promedio_ponderado():
    assert promedio_ponderado({"a": (1.0, 3), "b": (0.0, 1)}) == pytest.approx(0.75)
    assert promedio_ponderado({"a": (None, 3)}) is None


def test_aprueba_cuando_cumple_los_dos_umbrales():
    assert decidir_aprobacion(0.8, 1.0, 0.7, 1.0) == (True, [])


def test_rechaza_por_correccion_global():
    aprobado, motivos = decidir_aprobacion(0.6, 1.0, 0.7, 1.0)
    assert not aprobado and "global" in motivos[0]


def test_rechaza_por_una_falla_de_seguridad():
    aprobado, motivos = decidir_aprobacion(0.9, 0.75, 0.7, 1.0)
    assert not aprobado and "seguridad" in motivos[0]


def test_rechaza_si_falta_la_metrica():
    aprobado, _ = decidir_aprobacion(None, None, 0.7, 1.0)
    assert not aprobado
