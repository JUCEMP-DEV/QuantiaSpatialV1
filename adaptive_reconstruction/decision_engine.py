from __future__ import annotations

from .contracts import AdaptiveRoutePlan, ModuleDecision, ReconstructionDiagnostics


class AdaptiveModuleDecisionEngine:
    """Decide qué submódulos necesita cada LevelView a partir de evidencia.

    No usa nombres de caso ni reglas por proyecto. La ruta se deriva de densidad de
    seed F03, cantidad de candidatos recuperables, ruido contextual y calidad
    topológica observada.
    """

    def plan_pre_topology(
        self,
        *,
        seed_count: int,
        discovered_count: int,
        quarantined_count: int,
    ) -> AdaptiveRoutePlan:
        discovered = max(discovered_count, 1)
        seed_ratio = seed_count / discovered
        quarantine_ratio = quarantined_count / discovered
        recovery_ratio = max(0.0, min(1.0, (discovered_count - seed_count) / discovered))

        if seed_count == 0:
            mode = "DENSE_RECOVERY"
            recovery_strength = 1.0
            reasons = ["F03 no aporta seeds; se requiere recuperación geométrica amplia."]
        elif seed_ratio >= 0.45 and quarantine_ratio < 0.30:
            mode = "SEED_FIRST"
            recovery_strength = 0.45
            reasons = ["La selección F03 cubre una parte importante de la red; se protege como ancla."]
        elif discovered_count >= max(2 * seed_count, seed_count + 20):
            mode = "DENSE_RECOVERY"
            recovery_strength = max(0.70, recovery_ratio)
            reasons = ["Candidate Discovery contiene mucha geometría adicional respecto a F03; se activa densificación."]
        else:
            mode = "BALANCED"
            recovery_strength = 0.60
            reasons = ["F03 y Candidate Discovery aportan evidencia complementaria; se usa ruta balanceada."]

        decisions = [
            ModuleDecision(module="F03_STRUCTURAL_SEED", enabled=seed_count > 0, reason="Protege muros ya seleccionados por F03."),
            ModuleDecision(module="CONTEXT_NEGATIVE_MASK", enabled=quarantined_count > 0, reason="Encapsula ruido contextual sin borrar evidencia."),
            ModuleDecision(module="WALL_MASK_RECOVERY", enabled=True, reason="Recupera bandas y muros omitidos por la selección conservadora."),
            ModuleDecision(module="WALL_CANONICALIZATION", enabled=True, reason="Publica una sola centerline por muro físico."),
            ModuleDecision(module="STRICT_CONNECTIVITY", enabled=True, reason="Cierra únicamente gaps/junctions con evidencia geométrica."),
            ModuleDecision(module="ROOM_TOPOLOGY", enabled=True, reason="Clasifica perímetro/divisor por adyacencia espacial."),
            ModuleDecision(module="MULTIMODAL_CALL2", enabled=False, reason="Se decide después de evaluar la topología reconstruida."),
        ]
        return AdaptiveRoutePlan(
            mode=mode,
            modules=decisions,
            recovery_strength=recovery_strength,
            require_multimodal_review=False,
            reasons=reasons,
        )

    def plan_post_topology(
        self,
        *,
        current: AdaptiveRoutePlan,
        diagnostics: ReconstructionDiagnostics,
    ) -> AdaptiveRoutePlan:
        reasons = list(current.reasons)
        require_call2 = False

        # Sin habitaciones cerradas o con mucha ambigüedad, la geometría sola no
        # puede cerrar el plano con certeza: se escala a revisión multimodal.
        if diagnostics.interior_space_count == 0:
            require_call2 = True
            reasons.append("No se formaron espacios interiores cerrados; Call 2 debe buscar muros faltantes/gaps.")
        if diagnostics.review_ratio >= 0.30:
            require_call2 = True
            reasons.append("Una fracción alta del WallGraph permanece REVIEW.")
        if diagnostics.component_count >= max(4, diagnostics.selected_wall_count // 8):
            require_call2 = True
            reasons.append("La red sigue fragmentada en demasiados componentes.")
        if diagnostics.quarantine_ratio >= 0.20:
            require_call2 = True
            reasons.append("Existe ruido contextual significativo; Call 2 debe contrastar contra el plano original.")

        # Si la red quedó fragmentada pese a seeds, marca explícitamente rescate.
        mode = current.mode
        if diagnostics.f03_seed_count > 0 and diagnostics.interior_space_count == 0:
            mode = "SEED_RESCUE"

        modules = []
        for item in current.modules:
            if item.module == "MULTIMODAL_CALL2":
                modules.append(item.model_copy(update={
                    "enabled": require_call2,
                    "reason": "Contrasta Single-Line WallGraph con el LevelView original y devuelve solo deltas.",
                }))
            else:
                modules.append(item)

        return current.model_copy(update={
            "mode": mode,
            "modules": modules,
            "require_multimodal_review": require_call2,
            "reasons": reasons,
        })
