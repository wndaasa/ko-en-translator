#!/usr/bin/env bash
# Stage 0 - PyTorch(ROCm) GPU 동작 검증 (WSL + ROCDXG)
set -uo pipefail

PY="$HOME/miniconda3/envs/tf/bin/python"

# ROCm 7.2.x WSL(ROCDXG)에서 GPU 탐지에 필수. ROCk 7.13부터는 불필요.
export HSA_ENABLE_DXG_DETECTION=1

echo "==> librocdxg 존재 확인"
ls -l /opt/rocm/lib/librocdxg.so 2>&1 || echo "  (경고) librocdxg.so 없음 — 01_install_rocdxg.sh 먼저 실행 필요"

echo
echo "==> rocminfo (GPU 에이전트 확인)"
/opt/rocm/bin/rocminfo 2>&1 | grep -iE 'Marketing|^\s*Name:\s*gfx|gfx11' | head

echo
echo "==> [A] torch 검증 (HSA_OVERRIDE 없이)"
"$PY" - <<'PYEOF'
import torch
print("torch:", torch.__version__)
ok = torch.cuda.is_available()
print("cuda.is_available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul OK, mean:", float(y.float().mean()))
PYEOF
rcA=$?
echo "    [A] exit code: $rcA"

if [ $rcA -ne 0 ]; then
  echo
  echo "==> [B] HSA_OVERRIDE_GFX_VERSION=11.0.0 으로 재시도 (gfx1101 커널 우회)"
  HSA_OVERRIDE_GFX_VERSION=11.0.0 "$PY" - <<'PYEOF'
import torch
ok = torch.cuda.is_available()
print("cuda.is_available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("matmul OK, mean:", float(y.float().mean()))
PYEOF
  echo "    [B] exit code: $?"
fi
echo "==> done"
