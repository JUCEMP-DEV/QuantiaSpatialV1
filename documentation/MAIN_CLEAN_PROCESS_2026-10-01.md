# MAIN CLEAN PROCESS V1

Fecha: 2026-10-01

## Punto de retorno / baseline histórico

El estado completo anterior a esta limpieza quedó congelado en la rama:

`archive/spatial-baseline-pre-clean-2026-10-01`

Commit:

`5bd65974340fd69f0371905df8ba734be6ac3e68`

Esa rama conserva los resultados generados por pruebas, replays, historias Gemini,
rasters, imágenes, WallGraphs, Call 2 outputs y probes estandarizados del proceso
anterior.

## Regla de main

`main` contiene el motor Spatial y pruebas activas que no dependen de artefactos
históricos versionados. Los directorios `tests/output/` y `tests/data/` quedan
fuera del versionado para que una nueva corrida no reutilice accidentalmente una
salida anterior.

La información histórica no fue descartada: se consulta y compara directamente
contra la rama archive.

## Alcance

Esta operación es estructural. No modifica la lógica de:

- F02
- Adaptive Reconstruction
- PostFilter V4
- Canonical WallGraph
- Call 2

`PROJECT_SITE_CONTEXT_V1` permanece en main como parte del nuevo proceso candidato.
