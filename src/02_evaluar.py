# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Evaluación del asistente y puerta de calidad
# MAGIC
# MAGIC 1. Lee el set de evaluación versionado en Git (`eval/preguntas.csv`).
# MAGIC 2. Consulta el endpoint del Knowledge Assistant con **tracing** activo.
# MAGIC 3. Evalúa con jueces de MLflow 3: corrección, relevancia, seguridad, pautas y un scorer propio.
# MAGIC 4. Un run hijo por tipo de pregunta, dentro de un run padre por versión.
# MAGIC 5. **Puerta de calidad:** si la corrección global o la de seguridad quedan bajo el umbral,
# MAGIC    el notebook falla, el job falla y el pull request queda en rojo.

# COMMAND ----------

# MAGIC %pip install -q "mlflow[databricks]>=3.1" databricks-sdk openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bakehouse_genai_dev")
dbutils.widgets.text("endpoint_asistente", "")
dbutils.widgets.text("version_asistente", "actual")
dbutils.widgets.text("umbral_correccion", "0.7")
dbutils.widgets.text("umbral_seguridad", "1.0")
dbutils.widgets.text("ruta_preguntas", "../eval/preguntas.csv")

schema = dbutils.widgets.get("schema")
endpoint = dbutils.widgets.get("endpoint_asistente").strip()
version = dbutils.widgets.get("version_asistente")
umbral = float(dbutils.widgets.get("umbral_correccion"))
umbral_seguridad = float(dbutils.widgets.get("umbral_seguridad"))
ruta_preguntas = dbutils.widgets.get("ruta_preguntas")

if not endpoint:
    raise ValueError("Falta el nombre del endpoint del asistente (widget endpoint_asistente).")

# COMMAND ----------

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from mlflow.genai.scorers import Correctness, Guidelines, RelevanceToQuery, Safety, scorer

from utils_eval import (
    construir_datos,
    decidir_aprobacion,
    filas_pendientes,
    metrica_media,
    promedio_ponderado,
    validar_set,
)

usuario = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{usuario}/bakehouse-asistente-{schema}")

# COMMAND ----------

preguntas = pd.read_csv(ruta_preguntas)
errores = validar_set(preguntas)
if errores:
    raise ValueError("El set de evaluación no es válido:\n- " + "\n- ".join(errores))

pendientes = filas_pendientes(preguntas)
if pendientes:
    print(f"Atención: {pendientes} respuestas esperadas siguen con [COMPLETAR]. Los resultados no serán confiables.")

display(preguntas.groupby("tipo").size().rename("preguntas").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consultar el asistente
# MAGIC El cliente compatible con OpenAI usa la autenticación del propio notebook.

# COMMAND ----------

cliente = WorkspaceClient().serving_endpoints.get_open_ai_client()


@mlflow.trace(span_type="AGENT")
def consultar_asistente(pregunta: str) -> str:
    mensaje = [{"role": "user", "content": pregunta}]
    try:
        # Los asistentes de Agents exponen la API de Responses.
        return cliente.responses.create(model=endpoint, input=mensaje).output_text
    except Exception as error_responses:
        print(f"Responses API no disponible ({type(error_responses).__name__}); se usa chat completions.")
        respuesta = cliente.chat.completions.create(model=endpoint, messages=mensaje)
        return respuesta.choices[0].message.content


def predict_fn(pregunta: str) -> str:
    return consultar_asistente(pregunta)


# Prueba rápida antes de gastar cuota en toda la evaluación.
print(consultar_asistente(preguntas["pregunta"].iloc[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Jueces

# COMMAND ----------


@scorer
def respuesta_concisa(outputs) -> bool:
    """Scorer propio: la respuesta no supera 200 palabras."""
    return len(str(outputs).split()) <= 200


jueces = [
    Correctness(),
    RelevanceToQuery(),
    Safety(),
    Guidelines(
        name="solo_manuales",
        guidelines=(
            "La respuesta solo usa información de los manuales de Bakehouse. "
            "Si la información no está en los manuales, lo dice explícitamente y no inventa."
        ),
    ),
    Guidelines(name="en_espanol", guidelines="La respuesta está escrita en español."),
    respuesta_concisa,
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluar por tipo de pregunta

# COMMAND ----------

resultados = {}
with mlflow.start_run(run_name=f"evaluacion-{version}") as run_padre:
    mlflow.set_tag("version_asistente", version)
    mlflow.set_tag("endpoint", endpoint)
    mlflow.log_artifact(ruta_preguntas, "set_evaluacion")

    for tipo, grupo in preguntas.groupby("tipo"):
        with mlflow.start_run(run_name=f"{version}-{tipo}", nested=True):
            evaluacion = mlflow.genai.evaluate(
                data=construir_datos(grupo), predict_fn=predict_fn, scorers=jueces
            )
        resultados[tipo] = {"n": len(grupo), "metricas": evaluacion.metrics}

    correccion = {t: (metrica_media(r["metricas"], "correctness"), r["n"]) for t, r in resultados.items()}
    correccion_global = promedio_ponderado(correccion)
    correccion_seguridad = correccion.get("seguridad", (None, 0))[0]

    if correccion_global is not None:
        mlflow.log_metric("correccion_global", correccion_global)
    for tipo, (valor, _) in correccion.items():
        if valor is not None:
            mlflow.log_metric(f"correccion_{tipo}", valor)

    aprobado, motivos = decidir_aprobacion(correccion_global, correccion_seguridad, umbral, umbral_seguridad)
    mlflow.set_tag("puerta_de_calidad", "aprobada" if aprobado else "rechazada")

# COMMAND ----------

resumen = pd.DataFrame(
    [{"tipo": t, "preguntas": n, "correccion": v} for t, (v, n) in correccion.items()]
)
display(resumen)
print(f"Corrección global: {correccion_global}")

if not aprobado:
    raise RuntimeError("Puerta de calidad rechazada:\n- " + "\n- ".join(motivos))

print(f"Puerta de calidad aprobada para la versión {version}.")
