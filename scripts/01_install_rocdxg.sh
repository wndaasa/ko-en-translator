#!/usr/bin/env bash
# Stage 0 - librocdxg(ROCDXG) 설치: WSL <-> Windows GPU 드라이버 연결 계층.
# amdgpu-install(네이티브 ROCm)만으로는 WSL에서 GPU가 안 잡힘. 이 라이브러리가 있어야
# /dev/dxg 경유로 GPU 컴퓨트가 동작함. sudo 필요 → WSL 터미널에서 직접 실행:
#   bash /mnt/c/Users/dladu/study/Transformer/scripts/01_install_rocdxg.sh
set -euo pipefail

DEB="$HOME/rocm-setup/rocdxg-roct_1.2.0_amd64.deb"
URL="https://github.com/ROCm/librocdxg/releases/download/v1.2.0/rocdxg-roct_1.2.0_amd64.deb"

echo "==> 1/2 rocdxg-roct deb 확인/다운로드"
mkdir -p "$HOME/rocm-setup"
[ -f "$DEB" ] || curl -sSL -o "$DEB" "$URL"
ls -l "$DEB"

echo "==> 2/2 설치"
sudo apt install -y "$DEB"

echo
echo "==> 설치 확인"
ls -l /opt/rocm/lib/librocdxg.so* 2>&1 || echo "  (경고) librocdxg.so 미발견"
echo "==> 끝. ROCm 7.2.x 에서는 'HSA_ENABLE_DXG_DETECTION=1' 환경변수도 필요합니다(검증 스크립트가 자동 설정)."
