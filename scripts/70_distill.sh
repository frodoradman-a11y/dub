#!/usr/bin/env bash
# D4: few-step student. STUDENT_CFG=configs/dubflow_m.yaml (M->M, init from teacher) or configs/dubflow_s.yaml (M->S).
set -euo pipefail; source scripts/env.sh
STUDENT_CFG=${STUDENT_CFG:-$CFG}
INIT=""; [ "$STUDENT_CFG" = "$CFG" ] && INIT="--student-init $RUN/d3/last.pt"
$TORCHRUN -m dubflow.train.distill --teacher $RUN/d3/last.pt --config $STUDENT_CFG $INIT \
  data.stage=dub data.dub_manifest=data/dub/pairs.jsonl train.out_dir=$RUN/d4 "$@"
