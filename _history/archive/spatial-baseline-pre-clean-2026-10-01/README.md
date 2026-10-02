# Spatial baseline pre-clean — 2026-10-01

Esta carpeta conserva físicamente dentro de `main` los artefactos y pruebas del
proceso Spatial anterior que fueron retirados de las rutas operativas.

Origen congelado:

- rama original: `archive/spatial-baseline-pre-clean-2026-10-01`
- commit: `5bd65974340fd69f0371905df8ba734be6ac3e68`
- archivos conservados: 135

## Estructura

Los archivos mantienen su ruta relativa original debajo de esta carpeta:

`archive/spatial-baseline-pre-clean-2026-10-01/tests/data/`
`archive/spatial-baseline-pre-clean-2026-10-01/tests/output/`
`archive/spatial-baseline-pre-clean-2026-10-01/tests/*.py` para probes/standardized tests históricos dependientes
de esos baselines.

## Regla operativa

Nada bajo esta carpeta forma parte del proceso activo de Spatial. Se conserva
exclusivamente para comparación, auditoría y trazabilidad.

Las nuevas pruebas deben usar las rutas activas de `tests/`, generar sus outputs
localmente y no consumir automáticamente estos replays históricos.
