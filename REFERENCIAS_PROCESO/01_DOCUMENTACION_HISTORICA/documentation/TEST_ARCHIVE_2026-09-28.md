# Depuracion de pruebas - 2026-09-28

Se archivaron 8 modulos de pruebas obsoletas, sus resultados, dos notas de probes,
los caches asociados y el contenido previamente historico de documentation/history.
Total: 462 archivos, 68459029 bytes (65.29 MiB), verificados por SHA256.

Destino: [D:/03 INGENIEIRA SISTEMAS/03 RESIDENCIAS PROFESIONALES/REFERENCIAS Y ANEXOS Quantia General/quantia_spatialV1/archivo_pruebas_20260928_092025_final](<D:/03 INGENIEIRA SISTEMAS/03 RESIDENCIAS PROFESIONALES/REFERENCIAS Y ANEXOS Quantia General/quantia_spatialV1/archivo_pruebas_20260928_092025_final>)
Inventario por archivo: [ARCHIVE_TESTS_2026-09-28.json](ARCHIVE_TESTS_2026-09-28.json).
El archivo externo contiene manifest.json, README.md, estructura relativa original
bajo quantia_spatialV1/ y una copia de referencia del cargador bajo support_snapshot/.
El primer intento se revirtio por MAX_PATH; su registro se conserva dentro de
previous_attempt_rolled_back. El traslado final admite rutas largas de Windows.

## Criterio

Se retiraron probes que implementaban reconstrucciones previas al orquestador
Adaptive + PostFilter V4 + Canonical Integrity V2, asi como el harness Call 2 V1
sustituido por V2. Los imports entre probes se archivaron juntos. No hay imports
activos que dependan de los modulos movidos.

No se clasificaron pruebas como obsoletas solo por su sufijo V1/V2/V3 o por fallar.
Permanecen activos los contratos sinteticos de PostFilter, F03, escala, contexto,
solver y elementos arquitectonicos, las regresiones estandarizadas de subsistemas,
el diagnostico Adaptive y los harnesses vigentes de Call 2 y proveedores.

## Conservado

- Codigo productivo y parametros del motor sin cambios.
- tests/data: fixtures, replays y geometria baseline PostFilter V4.
- tests/output/quantia_spatial_v1_process: evidencia del baseline previo.
- tests/output/canonical_integrity_v2: resultados validados de los seis niveles.
- tests/output/canonical_visual_review: visor index.html y comparaciones.
- documentation/checkpoints y metadatos actuales de version.
- 27 modulos de pruebas activos.

## Verificacion

- pytest --collect-only: 134 casos recogidos, sin errores de importacion.
- Suite sintetica + contrato Scale Resolver V1.7 antes: 105 PASS, 3 FAIL.
- Misma suite despues: 105 PASS, los mismos 3 FAIL; sin fallos nuevos.
- No se ejecutaron proveedores ni llamadas Gemini/Call 2.
- No se repitio la reconstruccion integral: no cambio el motor ni sus entradas.

Los fallos previos permanecen visibles en test_quantia_spatial_scale_foundation_v1_6_synthetic.py:

- test_fragmented_general_dimension_graphics_are_stitched_before_scale_resolution
- test_direct_hv_graphic_consensus_cannot_be_outvoted_by_many_correlated_axis_spans
- test_real_miguel_v_pb_conflict_policy_closes_only_self_inconsistent_orientation

El segundo espera un metodo privado que ya no existe; los otros dos esperan
RESOLVED donde el resolver devuelve UNRESOLVED/CONFLICT. Requieren una revision
separada de compatibilidad/escala, no eliminar cobertura durante una limpieza.
Detalle: [TEST_ARCHIVE_VALIDATION_2026-09-28.xml](TEST_ARCHIVE_VALIDATION_2026-09-28.xml).

## Restauracion

Consultar manifest.json y README.md en el archivo externo. Restaurar cada ruta
relativa bajo Backend/app/quantia_spatialV1 en un checkout compatible y comprobar
colisiones y SHA256 antes de copiar. Los probes mantienen imports y rutas historicas;
el archivo externo no es una suite autonoma ni debe agregarse al PYTHONPATH productivo.
