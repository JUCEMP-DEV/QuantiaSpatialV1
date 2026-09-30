# DEPURACIÓN QUANTIA SPATIALV1 — 2026-09-30

## Objetivo
Separar el árbol operativo vigente de históricos, probes, outputs anteriores, checkpoints y archivos generados sin eliminar trazabilidad.

## Resultado
- Archivos movidos a `REFERENCIAS_PROCESO/`: **226** movimientos.
- Código Python operativo comparado contra el ZIP original: **161 archivos sin cambios**.
- Código productivo modificado durante la depuración: **ninguno**.
- Probes Qwen/Hugging Face retirados de la suite activa: 2 módulos; quedan en referencias.
- `experiments/` separado porque no es importado por el orquestador productivo actual.

## Árbol operativo conservado
- F01 / F01.5 / F02.
- `reconstruction_core/`.
- `adaptive_reconstruction/`.
- `post_reconstruction_filters/`.
- `canonical_wallgraph/`.
- `process_engine.py`.
- proveedores y contratos.
- pruebas de regresión y datos requeridos.
- snapshots actuales necesarios para Call 2 y Space Closure.

## Evidencia vigente conservada
- `canonical_integrity_v2`.
- `canonical_visual_review`.
- `quantia_spatial_v1_process`.
- `call2_interface04/gemini`.
- `space_closure_probe_v1`.
- `space_closure_constraint_probe_v2`.

## Separado como referencia
- históricos/versionados anteriores;
- return points y manifests antiguos;
- checkpoints empaquetados;
- outputs Adaptive antiguos;
- Call 2 V2 anterior;
- pruebas/output Qwen-HF;
- Groq/offline de Interface04;
- código `experiments/`;
- caches Python.

## Validación
- `compileall`: PASS antes de retirar los caches generados por la validación.
- La colección de pytest fuera del Backend completo no puede completarse por ausencia de `app.core`; el ZIP original falla por la misma dependencia externa.
- Original aislado: 104 tests recogidos, 18 errores de importación por entorno/dependencias.
- Depurado aislado: 104 tests recogidos, 16 errores; los 2 errores menos corresponden exactamente a los 2 probes Qwen/HF archivados.
- No se ejecutaron proveedores ni llamadas de red.

## Regla para el nuevo repo
`REFERENCIAS_PROCESO/04_OUTPUTS_HISTORICOS/` y `99_CACHE_GENERADO/` están incluidos localmente para conservación, pero `.gitignore` evita recomendar su incorporación al repositorio.
