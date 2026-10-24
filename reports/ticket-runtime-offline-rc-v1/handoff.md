# Server handoff

This archive is CPU-only and offline-only. It verifies the frozen source bindings,
runs the 11-scenario in-memory integration acceptance, verifies saved output, and
does not load the model or contact a ticket backend.

```bash
cd /konglingjie/sft_lora/lora_sft_toolcall_day1
unzip -n lora_sft_ticket_runtime_offline_rc_v1.zip

export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
PY=/konglingjie/sft_lora/envs/toolcall-sft-py311/bin/python

"$PY" -B scripts/run_ticket_runtime_offline_rc_v1.py --stage verify --tests &&
"$PY" -B scripts/run_ticket_runtime_offline_rc_v1.py --stage verify-output
```

The package already contains the frozen acceptance output. The test suite
replays the 11 scenarios in memory, while `verify-output` checks the saved
evidence. The separate `accept` stage is intentionally overwrite-protected and
is only for building a new release package in a clean source tree.

