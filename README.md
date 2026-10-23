# bakehouse-genaiops

Caso resuelto del curso **Experiencias en Analítica**: un Knowledge Assistant que responde preguntas del
personal de Bakehouse usando manuales en PDF, con su calidad medida por MLflow y vigilada con CI/CD desde GitHub.

```
bakehouse-genaiops/
├── databricks.yml            # Bundle: variables, umbrales y ambientes dev / prod
├── resources/jobs.yml        # Jobs: ingesta de PDFs y evaluación (semanal en prod)
├── src/
│   ├── 01_ingesta_pdfs.py    # Copia docs/*.pdf al Volume
│   ├── 02_evaluar.py         # Evaluación con jueces de MLflow 3 y puerta de calidad
│   └── utils_eval.py         # Funciones puras (se prueban en CI)
├── eval/preguntas.csv        # El set de evaluación, versionado
├── docs/                     # Los PDFs del asistente
├── tests/test_eval.py        # Pruebas con pytest
└── .github/workflows/
    ├── ci.yml                # Pull request: pruebas + deploy a dev + evaluación
    └── cd.yml                # Merge a main: deploy a prod + copia de PDFs
```

## Qué automatiza y qué no

El **asistente se crea y configura en la interfaz** (Agents › Create Agent › Knowledge Assistant): fuentes,
descripción e instrucciones. Este repo versiona y automatiza todo lo que rodea al asistente:

- los PDFs (carpeta `docs/`) y su copia al Volume;
- el **set de evaluación** (`eval/preguntas.csv`) y los jueces;
- la **puerta de calidad**: si la corrección baja del umbral o falla una pregunta de seguridad, el CI queda en rojo.

## Puesta en marcha

### 1. Primera ingesta y creación del asistente
1. Agrega los PDFs a `docs/` (ver `docs/README.md`).
2. Despliega y copia los PDFs al Volume de prod:
   ```bash
   databricks bundle deploy -t prod --var="endpoint_asistente=pendiente"
   databricks bundle run ingesta_manuales -t prod --var="endpoint_asistente=pendiente"
   ```
3. En **Agents › Create Agent › Knowledge Assistant**, usa como fuente el Volume
   `workspace.bakehouse_genai_prod.manuales`. Escribe una descripción de cada documento y las instrucciones.
4. Cuando termine de aprovisionarse, copia el **nombre del endpoint** del asistente.

### 2. GitHub
En **Settings › Secrets and variables › Actions**:

| Tipo | Nombre | Valor |
|---|---|---|
| Secret | `DATABRICKS_HOST` | URL del workspace, sin `/` al final |
| Secret | `DATABRICKS_TOKEN` | Token personal (Settings › Developer › Access tokens) |
| Variable | `ENDPOINT_ASISTENTE` | Nombre del endpoint del paso 1.4 |

> El token nunca va en el código. Los pull requests desde forks no reciben secretos: trabajen con ramas del mismo repo.

### 3. Completar el set de evaluación
Reemplaza cada `[COMPLETAR ...]` de `eval/preguntas.csv` con la respuesta que da el manual.
Columnas: `pregunta`, `respuesta_esperada` y `tipo`, que puede ser `dato_puntual`, `procedimiento`,
`seguridad` o `fuera_de_alcance`. Mínimo 15 preguntas y los cuatro tipos.

## Flujo de trabajo: de la v1 a la v2

1. **v1:** corre la evaluación (`databricks bundle run evaluacion_asistente -t dev --var="version_asistente=v1"`).
2. **Diagnosticar:** en el experimento `bakehouse-asistente-…`, filtra las preguntas con peor calificación
   y abre sus trazas. ¿Falló la recuperación (fragmento equivocado) o la generación (inventó o ignoró el contexto)?
3. **Ajustar** las instrucciones o las descripciones de las fuentes en la interfaz del asistente.
4. **v2:** abre un pull request (por ejemplo, con nuevas preguntas o pautas). El CI evalúa la versión actual del asistente.
5. Compara los runs `evaluacion-v1` y `evaluacion-pr-N` en MLflow. Si aprueba, haz merge.

## La puerta de calidad

Se aprueba si se cumplen las dos condiciones (ajustables en `databricks.yml`):

- corrección global ≥ `umbral_correccion` (por defecto 0,7);
- corrección en preguntas de `seguridad` ≥ `umbral_seguridad` (por defecto 1,0, o sea, cero fallas).

## Notas para Free Edition

- Cada evaluación llama al asistente y a los jueces: abre el pull request cuando el cambio esté listo.
- Desde el 1 de noviembre de 2026, el uso de funciones de IA cuenta contra los presupuestos de Unity Gateway.
- Después de cambiar un PDF, sincroniza las fuentes del asistente en la interfaz.
- Si no quieres la evaluación semanal en prod, cambia `estado_programacion` a `PAUSED`.

## Problemas comunes

| Síntoma | Causa probable |
|---|---|
| `no value assigned to required variable endpoint_asistente` | Falta la variable `ENDPOINT_ASISTENTE` en GitHub o `--var` en local |
| La prueba rápida del notebook falla con 404 | El nombre del endpoint no es el correcto o el asistente aún se aprovisiona |
| Corrección muy baja en todo | Siguen las respuestas `[COMPLETAR ...]` en el CSV |
| `RetrievalGroundedness` no aparece | Ese juez necesita el paso de recuperación en la traza; usa la pauta `solo_manuales` |

## Bonus del reto
- RAG en código con Vector Search y comparación contra el Knowledge Assistant.
- Un scorer propio más útil que `respuesta_concisa` (por ejemplo, que verifique que cita el documento).
- Un tablero con la tendencia semanal de `correccion_global`.
