from __future__ import annotations

from app.quantia_spatialV1.core.models.project_site_context import ProjectSiteContext
from app.quantia_spatialV1.stages.evidence.project_site_prompt_context import (
    build_project_site_context_prompt,
)


def test_project_site_context_accepts_quantia_v2l_frontend_names() -> None:
    context = ProjectSiteContext.coerce(
        {
            "ubicacionProyecto": "Uruapan, Michoacan",
            "anchoTerrenoM": "8.50",
            "largoTerrenoM": "20.80",
            "areaTerrenoM2": "176.80",
            "areaConstruccionM2": "210.00",
            "niveles": "2",
            "alturaNivel1M": "",
            "sistemaEstructural": "Mamposteria",
            "factorAjuste": "1",
            "notas": "",
        }
    )

    assert context is not None
    assert context.source == "USER_DECLARED"
    assert context.site_width_m == 8.5
    assert context.site_length_m == 20.8
    assert context.site_area_m2 == 176.8
    assert context.construction_area_m2 == 210.0
    assert context.level_count == 2
    assert context.level_1_height_m is None
    assert context.has_site_dimensions


def test_site_area_is_not_forced_to_width_times_length() -> None:
    context = ProjectSiteContext(
        site_width_m=8.5,
        site_length_m=20.8,
        site_area_m2=170.0,
    )
    assert context.site_area_m2 == 170.0


def test_prompt_is_unchanged_without_context() -> None:
    base = "PROMPT_BASE"
    assert (
        build_project_site_context_prompt(
            base_prompt=base,
            project_site_context=None,
        )
        == base
    )


def test_prompt_preserves_user_context_as_independent_evidence() -> None:
    context = ProjectSiteContext(
        site_width_m=8.5,
        site_length_m=20.8,
        site_area_m2=176.8,
        level_count=2,
    )
    prompt = build_project_site_context_prompt(
        base_prompt="PROMPT_BASE",
        project_site_context=context,
    )

    assert "USER_DECLARED" in prompt
    assert '"site_width_m": 8.5' in prompt
    assert "NO es verdad geometrica confirmada" in prompt
    assert "area del terreno = area de Planta Baja = area de Planta Alta" in prompt
    assert "voladizo" in prompt
    assert "banqueta" in prompt
