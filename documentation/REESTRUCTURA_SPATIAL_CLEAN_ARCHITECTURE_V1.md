# Reestructura Spatial Clean Architecture V1

Fecha: 2026-10-02
Estado: PLAN DE MIGRACION / SIN CAMBIO SEMANTICO DEL MOTOR
Base: 93db041a8d8dda0e7a50dfd7685bc34c6b157061
Punto de retorno: checkpoint/pre-clean-architecture-2026-10-02

## Objetivo

Reducir el ruido estructural de QuantiaSpatialV1 sin perder lógica, trazabilidad ni resultados ya validados.

La reestructura no busca convertir el motor en archivos gigantes. El objetivo es reducir carpetas de primer nivel, eliminar duplicidad conceptual y dejar visible el flujo real del motor.

## Problema actual

La estructura actual creció por iteraciones históricas y separa una misma responsabilidad en varias carpetas:

- reconstruction_core/
- adaptive_reconstruction/
- post_reconstruction_filters/
- canonical_wallgraph/

Todas pertenecen al mismo dominio: reconstrucción y consolidación de muros.

También:

- prompts/
- providers/
- transport/

pertenecen a una misma frontera: integración multimodal/IA.

Y:

- models/
- parametric_model/
- contracts/

son tres representaciones del dominio espacial que deben quedar agrupadas bajo un núcleo común.

El problema no es la cantidad total de módulos sino que las fronteras de carpetas no representan con claridad el flujo de ejecución.

## Flujo canónico objetivo

documento
→ levels
→ evidence
→ perimeter
→ metric scale
→ walls
→ Call 2 / gap review
→ spaces
→ architectural elements
→ correlation/readiness
→ QuantiaSpatialContract

## Estructura objetivo

```text
quantia_spatialV1/
├── engine.py
├── core/
│   ├── models/
│   ├── contracts.py
│   └── scale/
├── stages/
│   ├── levels/
│   ├── evidence/
│   ├── perimeter/
│   ├── walls/
│   ├── spaces/
│   ├── elements/
│   ├── correlation.py
│   └── delivery.py
├── ai/
│   ├── vision.py
│   ├── call2.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── history.py
│   └── replay.py
├── tests/
├── documentation/
└── _history/
```

Solo tres carpetas contienen runtime principal: core, stages y ai.

## Mapa de migración

### core/

models/ → core/models/
parametric_model/ → core/models/parametric/
contracts/spatial_contract.py → core/contracts.py

Escala transversal:
- reconstruction_core/scale_evidence_resolver.py
- reconstruction_core/level_scale_normalizer.py
- reconstruction_core/project_scale_reconciler.py
- reconstruction_core/raster_density_policy.py
- reconstruction_core/metric_normalized_context.py

→ core/scale/

### stages/levels/

phase_01_level/ → stages/levels/

Responsabilidad exclusiva:
- localizar páginas/niveles;
- producir LevelView;
- aislar el raster correcto;
- resolver fallback de identificación sin hardcodes por caso.

### stages/evidence/

phase_015_evidence/ → stages/evidence/

Responsabilidad:
- PyMuPDF/OpenCV/OCR/Gemini;
- normalización a RawEvidence;
- persistencia de evidencia;
- firma de replay.

### stages/perimeter/

phase_02_boundaries/ → stages/perimeter/

Responsabilidad:
- perímetro;
- grounding dimensional del perímetro;
- validación;
- reproyección del F02 ya validado.

### stages/walls/

Se consolida aquí:
- reconstruction_core/ excepto escala transversal;
- adaptive_reconstruction/;
- post_reconstruction_filters/;
- canonical_wallgraph/.

Subresponsabilidades internas:
- drawing/candidates;
- context;
- solver;
- adaptive recovery;
- post-filter;
- canonical wall graph;
- Call 2 corrections;
- candidate selection audit.

No se crearán cuatro subárboles equivalentes a los actuales. Los archivos se agruparán por función dentro de stages/walls/.

### stages/spaces/

NUEVO módulo productivo recuperado desde:
- test_quantia_spatial_v1_space_closure_probe_v1.py
- test_quantia_spatial_v1_space_closure_constraint_probe_v2.py

Responsabilidad:
- red topológica derivada;
- proyección canónica de ejes H/V para cierre;
- colapso residual de caras paralelas;
- logical closures sin crear muro físico;
- polygonize de espacios;
- wall ↔ space ownership;
- footprint constraint;
- semantic ↔ geometric matching;
- detección UNDER_SEGMENTED_MULTIPLE_SPACES;
- métricas de cierre.

### stages/elements/

architectural_elements/ → stages/elements/

Se integrará al flujo principal después de cerrar WallGraph y espacios.

Responsabilidad:
- DOOR;
- WINDOW;
- NOT_OPENING;
- host wall;
- posición/span;
- revisión semántica.

Las extensiones GARAGE_DOOR / STAIR se migrarán desde probes históricos únicamente después de validar sus contratos.

### stages/correlation.py

NUEVO.

Correlacionará sin inventar:
- muro ↔ centerline;
- muro ↔ eje;
- muro ↔ cota;
- muro ↔ espacio;
- opening ↔ host wall;
- espacio ↔ semántica;
- footprint ↔ perímetro;
- relaciones entre niveles/escaleras.

Salida:
MATCH / CONFLICT / UNKNOWN + evidencia.

### stages/delivery.py

NUEVO builder interno del QuantiaSpatialContract.

No será router ni endpoint.
No contendrá lógica del frontend.
Solo transforma resultados internos validados al contrato canónico.

### ai/

prompts/ + providers/ + transport/ + histories/replays → ai/

Separación interna:
- vision.py: provider multimodal base;
- call2.py: revisión multimodal del WallGraph;
- prompts.py: prompts;
- schemas.py: schemas provider-facing;
- history.py: persistencia append-only;
- replay.py: firmas y replay fail-closed.

## Código que debe salir del runtime principal

- renderers de diagnóstico;
- runners manuales;
- loaders de casos;
- fixtures;
- rutas locales;
- replays históricos;
- visualizadores HTML;
- experimentos antiguos.

Se conservarán bajo tests/ o _history/, nunca como dependencias del engine.

## Archivos claramente de ruido

- models/reconstruction_state.py: vacío.
- candidate_exclusion_policy.py: legacy V2, no conectado al pipeline vigente.
- README_PROPOSAL_C.txt: documentación histórica, no runtime.
- artifact_renderer.py de Adaptive/PostFilter/Call2: herramientas de diagnóstico, no motor.
- REFERENCIAS_PROCESO/: histórico.
- archive/: histórico.

No se eliminan hasta completar pruebas de equivalencia.

## Entry point final

Solo debe existir un acceso público:

```python
QuantiaSpatialEngine.run(...)
```

El actual QuantiaSpatialV1ProcessEngine deja de ser un segundo motor público y pasa a ser una etapa interna de reconstrucción de walls.

QuantiaV2L deberá consumir exclusivamente QuantiaSpatialEngine y no ensamblar fases internas.

## Recuperaciones pendientes antes del cierre

1. F01 bootstrap/fallback genérico.
2. CandidateSelectionAudit.
3. Call 2 gap_decisions.
4. SpaceClosureEngine.
5. SpaceConstraintValidator.
6. ArchitecturalElements integrado.
7. Grounding final de ejes/cotas.
8. CorrelationEngine.
9. QuantiaSpatialContractBuilder.
10. Readiness/quality real.

## Estrategia de migración

La reestructura será por equivalencia, no por reescritura.

1. mover sin cambiar comportamiento;
2. corregir imports;
3. ejecutar suite histórica + activa;
4. solo después eliminar compatibilidad;
5. luego recuperar lógica faltante;
6. volver a ejecutar casos Casa Viri, Miguel H y Miguel V;
7. solo entonces actualizar QuantiaV2L.

No se elimina código hasta que su reemplazo esté probado.
