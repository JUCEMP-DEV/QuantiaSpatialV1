PROPOSAL C — RECONSTRUCTION CORE / CONTEXT GATE V5

Base de retorno: Proposal C A1.2 Candidate Discovery.
Rama activa de validación: QUANTIA_V2L_SPATIAL_CONTEXT_GATE_V5_2026-09-09.

OBJETIVO
Aislar geometría repetitiva no-muro antes del CandidateGraph sin modificar F01,
F01.5, F02, engine.py ni la generación amplia de candidatos A1.2.

FLUJO V5
DrawingModel
  -> PhysicalStrokeNormalizer
  -> ContextRegionDetector
  -> ContextRegionAssembler
  -> CandidateContextGate
       ACTIVE      -> CandidateGraph
       REVIEW      -> CandidateGraph como hipótesis provisional
       QUARANTINE  -> encapsulado fuera del CandidateGraph
  -> GlobalTopologySolver

REGLAS CENTRALES
1. Las dos caras raster de un mismo trazo se colapsan solo para análisis de
   periodicidad. El DrawingModel y los WallCandidate originales no se alteran.
2. Una observación semántica sin bbox no puede localizar una escalera/región.
3. Micro-regiones compatibles se ensamblan antes de decidir cuarentena.
4. REVIEW no equivale a exclusión mientras no exista verificador visual.
5. No existe MAX_GRAPH_CANDIDATES ni recorte arbitrario antes del solver.
6. QUARANTINE no genera JUNCTION, CONTINUATION ni rooms en el CandidateGraph.

NO FORMA PARTE DEL FLUJO
- CandidateExclusionPolicy V2/V3.
- Recovery A1.3/A1.4/A1.5/A1.6.
- Visual Resolver V1-V6.

ESTADO
Rama experimental de validación. A1.2 permanece como punto de retorno hasta
completar Casa Viri + Miguel H + Miguel V.


SOLVER EVIDENCE V1 — 2026-09-10
- No cambia Candidate Discovery A1.2 ni F02.
- REVIEW conserva candidato pero agrega context_risk al solver.
- vector/raster confirman observación; no son identidad WALL.
- GlobalTopologySolver usa identidad geométrica + confianza observacional + riesgo contextual + topología.
- Política única para Casa Viri, Miguel H, Miguel V y futuros casos.
