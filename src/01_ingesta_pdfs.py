# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Copiar los manuales al Volume
# MAGIC
# MAGIC Los PDFs viven versionados en la carpeta `docs/` del repo. Este notebook los copia a un
# MAGIC Volume de Unity Catalog, que es la fuente del Knowledge Assistant.
# MAGIC
# MAGIC Después de copiar documentos nuevos o cambiados, sincroniza las fuentes del asistente en
# MAGIC **Agents › tu asistente**.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bakehouse_genai_dev")
dbutils.widgets.text("volumen", "manuales")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volumen = dbutils.widgets.get("volumen")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volumen}")

destino = f"/Volumes/{catalog}/{schema}/{volumen}"
print(f"Destino: {destino}")

# COMMAND ----------

import glob
import os
import shutil

# El notebook se ejecuta desde src/, así que docs/ está un nivel arriba.
origen = os.path.abspath(os.path.join(os.getcwd(), "..", "docs"))
pdfs = sorted(glob.glob(os.path.join(origen, "*.pdf")))

if not pdfs:
    dbutils.notebook.exit(f"No hay PDFs en {origen}. Agrégalos a la carpeta docs/ del repo.")

copiados = []
for ruta in pdfs:
    shutil.copy2(ruta, destino)
    copiados.append((os.path.basename(ruta), round(os.path.getsize(ruta) / 1024, 1)))

display(spark.createDataFrame(copiados, "archivo STRING, tamano_kb DOUBLE"))
