# Pruebas activas de Quantia Spatial V1

Baseline: Adaptive Reconstruction + PostFilter V4 + Canonical WallGraph Integrity V2.
Los sufijos historicos en pruebas sinteticas no implican obsolescencia: prueban
contratos que siguen activos en el motor actual.

## Entradas principales

- test_quantia_spatial_v1_process_all_3_v1.py: regresion integral offline de seis LevelViews;
  exige fixtures y PDFs, preserva geometria V4 y verifica integridad/evidencia.
- test_quantia_spatial_canonical_wallgraph_contract_v1_synthetic.py: gate de integridad,
  persistencia del bundle y trazabilidad de correcciones.
- test_quantia_spatial_post_filter*_synthetic.py: clasificacion, preservacion y canonicalizacion.
- render_canonical_visual_review.py: visor del resultado canonico validado.
- Pruebas de escala, contexto, solver, F03 y Adaptive: regresion de subsistemas.
- test_quantia_spatial_v1_call2_all_3_v2.py y pruebas de proveedores: opt-in explicito;
  no son necesarias para cerrar la capa determinista.

## Ejecucion desde Backend (PowerShell)

```powershell
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider --collect-only app/quantia_spatialV1/tests -q
.venv/Scripts/python.exe -B -m pytest -p no:cacheprovider app/quantia_spatialV1/tests/test_quantia_spatial_v1_process_all_3_v1.py -q
.venv/Scripts/python.exe -B -m app.quantia_spatialV1.tests.render_canonical_visual_review
```

Los probes obsoletos y documentation/history se trasladaron al repositorio externo
de antecedentes solicitado. El inventario y los pasos de restauracion estan en
[TEST_ARCHIVE_2026-09-28.md](../documentation/TEST_ARCHIVE_2026-09-28.md).
La suite amplia tiene tres fallos de escala preexistentes, documentados en ese informe;
no se eliminaron ni se marcaron skip para ocultarlos.
