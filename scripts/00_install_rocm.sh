#!/usr/bin/env bash
# Stage 0 - WSL2(Ubuntu 24.04)에 ROCm 7.2.1 설치
# sudo 비밀번호 입력이 필요하므로 WSL 터미널에서 직접 실행하세요.
#   bash /mnt/c/Users/dladu/study/Transformer/scripts/00_install_rocm.sh
set -euo pipefail

ROCM_DEB="amdgpu-install_7.2.1.70201-1_all.deb"
ROCM_URL="https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/${ROCM_DEB}"
WORK="$HOME/rocm-setup"

echo "==> 작업 디렉터리: ${WORK}"
mkdir -p "${WORK}"
cd "${WORK}"

echo "==> 1/4 apt 업데이트"
sudo apt update

echo "==> 2/4 amdgpu-install 패키지 다운로드"
[ -f "${ROCM_DEB}" ] || wget -O "${ROCM_DEB}" "${ROCM_URL}"

echo "==> 3/4 amdgpu-install 설치"
sudo apt install -y "./${ROCM_DEB}"

echo "==> 4/4 ROCm 런타임 설치 (WSL용, 커널모듈 제외)"
# ROCm 7.2.1(amdgpu-install 30.30.x)에는 'wsl' usecase가 없음. WSL 지원이 ROCDXG로
# 'rocm' usecase에 통합됨 (구버전 6.x 의 'wsl,rocm' 은 더 이상 유효하지 않음).
sudo amdgpu-install -y --usecase=rocm --no-dkms

echo
echo "==> 설치 완료. 확인:"
ls -d /opt/rocm* 2>/dev/null || echo "  (경고) /opt/rocm 이 보이지 않습니다."
command -v rocminfo >/dev/null && echo "  rocminfo: $(command -v rocminfo)" || echo "  (참고) rocminfo 미발견 — 보통 동작에는 문제 없음"
echo "==> 끝. 이제 Claude에게 알려주세요."
