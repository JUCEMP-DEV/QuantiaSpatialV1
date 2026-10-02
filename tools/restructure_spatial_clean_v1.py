from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE_NAME = "quantia_spatialV1"
CHECKPOINT_BRANCH = "checkpoint/pre-clean-architecture-2026-10-02"

DIR_MOVES = (
    ("models", "core/models"),
    ("parametric_model", "core/models/parametric"),
    ("contracts", "core/contracts"),
    ("phase_01_level", "stages/levels"),
    ("phase_015_evidence", "stages/evidence"),
    ("phase_02_boundaries", "stages/perimeter"),
    ("reconstruction_core", "stages/walls/core"),
    ("adaptive_reconstruction", "stages/walls/adaptive"),
    ("post_reconstruction_filters", "stages/walls/postfilter"),
    ("canonical_wallgraph", "stages/walls/canonical"),
    ("architectural_elements", "stages/elements"),
    ("prompts", "ai/prompts"),
    ("providers", "ai/providers"),
    ("transport", "ai/schemas"),
    ("experiments", "_history/experiments"),
    ("archive", "_history/archive"),
    ("REFERENCIAS_PROCESO", "_history/REFERENCIAS_PROCESO"),
)

FILE_MOVES = (("process_engine.py", "stages/walls/process.py"),)

SCALE_FILES = (
    "scale_evidence_resolver.py",
    "level_scale_normalizer.py",
    "project_scale_reconciler.py",
    "raster_density_policy.py",
    "metric_normalized_context.py",
)

IMPORT_REPLACEMENTS = (
    ("app.quantia_spatialV1.reconstruction_core.scale_evidence_resolver", "app.quantia_spatialV1.core.scale.scale_evidence_resolver"),
    ("app.quantia_spatialV1.reconstruction_core.level_scale_normalizer", "app.quantia_spatialV1.core.scale.level_scale_normalizer"),
    ("app.quantia_spatialV1.reconstruction_core.project_scale_reconciler", "app.quantia_spatialV1.core.scale.project_scale_reconciler"),
    ("app.quantia_spatialV1.reconstruction_core.raster_density_policy", "app.quantia_spatialV1.core.scale.raster_density_policy"),
    ("app.quantia_spatialV1.reconstruction_core.metric_normalized_context", "app.quantia_spatialV1.core.scale.metric_normalized_context"),
    ("app.quantia_spatialV1.models", "app.quantia_spatialV1.core.models"),
    ("app.quantia_spatialV1.parametric_model", "app.quantia_spatialV1.core.models.parametric"),
    ("app.quantia_spatialV1.contracts", "app.quantia_spatialV1.core.contracts"),
    ("app.quantia_spatialV1.phase_01_level", "app.quantia_spatialV1.stages.levels"),
    ("app.quantia_spatialV1.phase_015_evidence", "app.quantia_spatialV1.stages.evidence"),
    ("app.quantia_spatialV1.phase_02_boundaries", "app.quantia_spatialV1.stages.perimeter"),
    ("app.quantia_spatialV1.adaptive_reconstruction", "app.quantia_spatialV1.stages.walls.adaptive"),
    ("app.quantia_spatialV1.post_reconstruction_filters", "app.quantia_spatialV1.stages.walls.postfilter"),
    ("app.quantia_spatialV1.canonical_wallgraph", "app.quantia_spatialV1.stages.walls.canonical"),
    ("app.quantia_spatialV1.reconstruction_core", "app.quantia_spatialV1.stages.walls.core"),
    ("app.quantia_spatialV1.architectural_elements", "app.quantia_spatialV1.stages.elements"),
    ("app.quantia_spatialV1.prompts", "app.quantia_spatialV1.ai.prompts"),
    ("app.quantia_spatialV1.providers", "app.quantia_spatialV1.ai.providers"),
    ("app.quantia_spatialV1.transport", "app.quantia_spatialV1.ai.schemas"),
    ("app.quantia_spatialV1.process_engine", "app.quantia_spatialV1.stages.walls.process"),
)

RELATIVE_SCALE_IMPORTS = {
    ".metric_normalized_context": "app.quantia_spatialV1.core.scale.metric_normalized_context",
    ".scale_evidence_resolver": "app.quantia_spatialV1.core.scale.scale_evidence_resolver",
    ".level_scale_normalizer": "app.quantia_spatialV1.core.scale.level_scale_normalizer",
    ".project_scale_reconciler": "app.quantia_spatialV1.core.scale.project_scale_reconciler",
    ".raster_density_policy": "app.quantia_spatialV1.core.scale.raster_density_policy",
}

PACKAGE_INITS = (
    "core/__init__.py",
    "core/scale/__init__.py",
    "stages/__init__.py",
    "stages/walls/__init__.py",
    "ai/__init__.py",
)


def run(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=check)


def repo_root(package_root: Path) -> Path:
    result = run("git", "rev-parse", "--show-toplevel", cwd=package_root)
    return Path(result.stdout.strip()).resolve()


def ensure_clean(repo: Path) -> None:
    status = run("git", "status", "--porcelain", cwd=repo).stdout.strip()
    if status:
        raise RuntimeError("Working tree no está limpio. Commit/stash antes de reestructurar.\n" + status)


def ensure_checkpoint(repo: Path) -> None:
    existing = run("git", "branch", "--list", CHECKPOINT_BRANCH, cwd=repo).stdout.strip()
    if not existing:
        run("git", "branch", CHECKPOINT_BRANCH, cwd=repo)


def git_move(repo: Path, package_root: Path, src: str, dst: str) -> None:
    source = package_root / src
    target = package_root / dst
    if not source.exists():
        return
    if target.exists():
        raise RuntimeError(f"Destino ya existe: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source_rel = source.relative_to(repo)
    target_rel = target.relative_to(repo)
    run("git", "mv", str(source_rel), str(target_rel), cwd=repo)


def move_scale_files(repo: Path, package_root: Path) -> None:
    source_dir = package_root / "stages/walls/core"
    target_dir = package_root / "core/scale"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in SCALE_FILES:
        source = source_dir / name
        if source.exists():
            run(
                "git", "mv",
                str(source.relative_to(repo)),
                str((target_dir / name).relative_to(repo)),
                cwd=repo,
            )


def ensure_init_files(repo: Path, package_root: Path) -> None:
    for rel in PACKAGE_INITS:
        path = package_root / rel
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
            run("git", "add", str(path.relative_to(repo)), cwd=repo)


def rewrite_imports(package_root: Path) -> int:
    changed = 0
    for path in package_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        for old, new in IMPORT_REPLACEMENTS:
            text = text.replace(old, new)
        if text != original:
            path.write_text(text, encoding="utf-8", newline="\n")
            changed += 1
    return changed


def repair_scale_imports(package_root: Path) -> None:
    metric = package_root / "core/scale/metric_normalized_context.py"
    if metric.exists():
        text = metric.read_text(encoding="utf-8")
        text = text.replace(
            "from .drawing_model import DrawingModel",
            "from app.quantia_spatialV1.stages.walls.core.drawing_model import DrawingModel",
        )
        metric.write_text(text, encoding="utf-8", newline="\n")

    wall_core = package_root / "stages/walls/core"
    if wall_core.exists():
        for path in wall_core.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            original = text
            for old, new in RELATIVE_SCALE_IMPORTS.items():
                text = text.replace(f"from {old} import", f"from {new} import")
            if text != original:
                path.write_text(text, encoding="utf-8", newline="\n")


def remove_noise(package_root: Path) -> None:
    for cache in list(package_root.rglob("__pycache__")):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)

    empty_state = package_root / "core/models/reconstruction_state.py"
    if empty_state.exists() and not empty_state.read_text(encoding="utf-8").strip():
        empty_state.unlink()


def validate(repo: Path, package_root: Path) -> None:
    compile_result = run(sys.executable, "-m", "compileall", "-q", str(package_root), cwd=repo, check=False)
    if compile_result.returncode != 0:
        raise RuntimeError("compileall falló:\n" + compile_result.stdout + compile_result.stderr)

    diff_check = run("git", "diff", "--check", cwd=repo, check=False)
    if diff_check.returncode != 0:
        raise RuntimeError("git diff --check falló:\n" + diff_check.stdout + diff_check.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reestructura QuantiaSpatialV1 sin cambiar lógica de negocio.")
    parser.add_argument("package_root", type=Path, help="Ruta .../Backend/app/quantia_spatialV1")
    args = parser.parse_args()

    package_root = args.package_root.resolve()
    if package_root.name != PACKAGE_NAME or not (package_root / "engine.py").is_file():
        raise RuntimeError(f"Ruta inválida: {package_root}")

    repo = repo_root(package_root)
    ensure_clean(repo)
    ensure_checkpoint(repo)

    for src, dst in DIR_MOVES:
        git_move(repo, package_root, src, dst)
    for src, dst in FILE_MOVES:
        git_move(repo, package_root, src, dst)

    move_scale_files(repo, package_root)
    ensure_init_files(repo, package_root)
    rewrite_imports(package_root)
    repair_scale_imports(package_root)
    remove_noise(package_root)

    run("git", "add", "-A", cwd=repo)
    validate(repo, package_root)

    print("OK: reestructura aplicada y validación estática superada.")
    print("Punto de retorno:", CHECKPOINT_BRANCH)
    print("Siguiente paso: ejecutar pytest antes de hacer commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
