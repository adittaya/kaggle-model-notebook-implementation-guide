# Run ledger (latest tail)

Authoritative detail (decisions, DAGs, lessons) lives in [AGENT.md](../../AGENT.md); this ledger is
the quick reference for recent kernel runs and where their raw evidence lives.

| Kaggle ver | Session | Outcome | Evidence / notes |
|---|---|---|---|
| 20 | 2026-10-07 | **ERROR** | Died at the last op of the music lane: `SaveAudioAdvanced.execute() missing 1 required positional argument: 'format'`. numpy heal PROVEN in prod (`TASKS_HEALTHY=True`); all 3 smokes validated; music AR ran (`57% 72/126 [1.17s/it]`, tiled decode, lazy ComfySwitchNode skipped VAEDecodeAudio). Root cause = API prompt carried the *saved-workflow widget dict* `{"format":"flac"}` where `IO.DynamicCombo` needs the **option-key string** `"flac"`; silently dropped at executor → crash. Raw stream: `v20_run_2026-10-07.log.json`. |
| 21 | 2026-10-08 | **ERROR** | Died with a **2-byte log** (`[]`) — zero stdout/stderr flushed, zero output artifacts → platform-level kill, not a notebook-logic failure (the v21 format fix is proven locally 27/27). No evidence worth archiving beyond the pulled empty stream. |
| 22 | 2026-10-08 | **RUNNING** | v21 code + hardening: line-buffered stdout + `/kaggle/working/state.txt` milestone markers at every cell boundary, smoke submissions, wait-heartbeat ticks, `pub:tunnel-ok`. diff vs v21 = hardening only (`dataset_sources` deliberately off; cache attach lands as v22-feature). Verify: `TASKS_HEALTHY=True`, `SMOKE PASSED: ['image','video','music']`, `h3_music_*.flac` in output, `DUAL-GPU CONFIRMED`, tunnel + tasks green, sync uploads → dataset v3. |