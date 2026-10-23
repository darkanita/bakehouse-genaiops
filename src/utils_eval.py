"""Funciones puras de la evaluación del asistente.

No dependen de Databricks ni de MLflow, así que se prueban con pytest en el CI.
"""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

COLUMNAS = ["pregunta", "respuesta_esperada", "tipo"]
TIPOS_VALIDOS = {"dato_puntual", "procedimiento", "seguridad", "fuera_de_alcance"}
MINIMO_PREGUNTAS = 15
MARCA_PENDIENTE = "[COMPLETAR"


def validar_set(df: pd.DataFrame) -> list[str]:
    """Devuelve la lista de problemas del set de evaluación. Vacía = válido."""
    errores: list[str] = []
    faltantes = [c for c in COLUMNAS if c not in df.columns]
    if faltantes:
        return [f"Faltan columnas: {', '.join(faltantes)}"]

    if len(df) < MINIMO_PREGUNTAS:
        errores.append(f"Se necesitan al menos {MINIMO_PREGUNTAS} preguntas; hay {len(df)}")

    vacias = df[COLUMNAS].isna().any(axis=1) | (df[COLUMNAS].astype(str).apply(lambda s: s.str.strip()) == "").any(axis=1)
    if vacias.any():
        errores.append(f"Hay {int(vacias.sum())} filas con celdas vacías")

    invalidos = sorted(set(df["tipo"].dropna()) - TIPOS_VALIDOS)
    if invalidos:
        errores.append(f"Tipos no válidos: {', '.join(invalidos)}")

    ausentes = sorted(TIPOS_VALIDOS - set(df["tipo"].dropna()))
    if ausentes:
        errores.append(f"Faltan preguntas de tipo: {', '.join(ausentes)}")

    duplicadas = df["pregunta"].str.strip().str.lower().duplicated()
    if duplicadas.any():
        errores.append(f"Hay {int(duplicadas.sum())} preguntas duplicadas")

    return errores


def filas_pendientes(df: pd.DataFrame) -> int:
    """Cuántas respuestas esperadas siguen con la marca [COMPLETAR ...]."""
    return int(df["respuesta_esperada"].astype(str).str.contains(MARCA_PENDIENTE, regex=False).sum())


def construir_datos(df: pd.DataFrame) -> list[dict]:
    """Convierte el CSV al formato de mlflow.genai.evaluate."""
    return [
        {
            "inputs": {"pregunta": fila.pregunta},
            "expectations": {"expected_response": fila.respuesta_esperada},
        }
        for fila in df.itertuples(index=False)
    ]


def metrica_media(metricas: Mapping[str, float], nombre: str) -> Optional[float]:
    """Busca el promedio de un scorer en los resultados de MLflow.

    MLflow lo reporta como "<scorer>/mean" (por ejemplo "correctness/mean").
    """
    nombre = nombre.lower()
    candidatas = {k.lower(): v for k, v in metricas.items()}
    if f"{nombre}/mean" in candidatas:
        return float(candidatas[f"{nombre}/mean"])
    for clave, valor in candidatas.items():
        if clave.startswith(nombre) and "mean" in clave:
            return float(valor)
    if nombre in candidatas:
        return float(candidatas[nombre])
    return None


def promedio_ponderado(valores: Mapping[str, tuple[Optional[float], int]]) -> Optional[float]:
    """Promedio de (valor, cantidad) por grupo, ignorando grupos sin valor."""
    pares = [(v, n) for v, n in valores.values() if v is not None and n > 0]
    total = sum(n for _, n in pares)
    if total == 0:
        return None
    return sum(v * n for v, n in pares) / total


def decidir_aprobacion(
    correccion_global: Optional[float],
    correccion_seguridad: Optional[float],
    umbral: float,
    umbral_seguridad: float,
) -> tuple[bool, list[str]]:
    """Puerta de calidad del asistente. Devuelve (aprobado, motivos de rechazo)."""
    motivos: list[str] = []
    if correccion_global is None:
        motivos.append("No se encontró la métrica de corrección")
    elif correccion_global < umbral:
        motivos.append(f"Corrección global {correccion_global:.2f} < umbral {umbral:.2f}")
    if correccion_seguridad is None:
        motivos.append("No hay resultado para las preguntas de seguridad")
    elif correccion_seguridad < umbral_seguridad:
        motivos.append(
            f"Corrección en seguridad {correccion_seguridad:.2f} < umbral {umbral_seguridad:.2f}"
        )
    return (not motivos, motivos)
