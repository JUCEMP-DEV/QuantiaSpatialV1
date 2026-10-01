# Tests activos en main

Desde MAIN_CLEAN_PROCESS_V1, `main` no versiona resultados generados ni replays
históricos.

- `tests/output/`: salida local de nuevas pruebas; ignorada por Git.
- `tests/data/`: datos/replays locales de ejecución; ignorados por Git.
- los probes y standardized tests del proceso anterior permanecen en
  `archive/spatial-baseline-pre-clean-2026-10-01`.
- las pruebas sintéticas y contractuales independientes permanecen activas en main.

Para comparar una ejecución nueva con el proceso anterior, usar la rama archive en
otro worktree o consultar sus artefactos directamente; no copiarlos nuevamente a main.
