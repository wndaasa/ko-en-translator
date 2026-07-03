"""로컬 번역 HTTP 서비스 — 표준 라이브러리만 사용(추가 의존성 없음).

ozai_nb2 협업 에디터의 버블 메뉴 '번역'이 서버 프록시(/api/translate)를 통해 호출한다.
모델·토크나이저는 기동 시 1회 로드해 상주한다(요청마다 로드하면 수 초씩 걸림).

    python -m src.serve --artifacts runs/hf-sentence --port 8531

API
    GET  /health     → {"status": "ok", "device": "...", "direction": "..."}
    POST /translate  {"text": "...", "to": "en"|"ko"} → {"translation": "..."}
"""
from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from .minrnn import MinRNNConfig, MinRNNSeq2Seq
from .model import ModelConfig, Seq2SeqTransformer
from .tokenizer import load_tokenizer
from .translate import greedy_translate


def _load_model(ckpt_path: Path, device: str):
    """체크포인트의 arch 태그(train.py가 저장)에 맞는 클래스로 복원한다.
    translate.load_model은 어텐션 트랜스포머 고정이라 minrnn 가중치를 못 읽는다."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if ckpt.get("arch") == "minrnn":
        cfg = MinRNNConfig(**ckpt["config"])
        model = MinRNNSeq2Seq(cfg).to(device)
    else:
        cfg = ModelConfig(**ckpt["config"])
        model = Seq2SeqTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    direction = "bidi" if ckpt.get("bidirectional") else ckpt.get("direction", "ko2en")
    return model, direction

# 문장 단위(sentence-base) 모델은 greedy가 첫 <eos>에서 정지하므로,
# 여러 문장이 오면 나눠 번역해 이어붙인다. 마침표류 + 줄바꿈 기준의 단순 분리.
_SENT_SPLIT = re.compile(r"(?<=[.!?。…])\s+|\n+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]


# OPUS 자막 코퍼스 학습 흔적: 짧은 인사말류가 "- 안녕하세요 - 안녕하세요" 같은
# 자막 대화체로 나온다(도메인 태그로도 교정 안 됨 — casual/formal 모두 동일 확인).
# 원문에 대시가 없는데 출력이 대시로 시작하면 인공물로 보고 정리한다:
# 대시 구분 조각으로 나눠 연속 중복을 제거하고 이어붙인다.
_DASH_SPLIT = re.compile(r"\s*[-–—]\s*")


def _clean_subtitle_artifacts(source: str, output: str) -> str:
    if not re.match(r"\s*[-–—]", output) or re.match(r"\s*[-–—]", source):
        return output
    parts = [p.strip() for p in _DASH_SPLIT.split(output) if p.strip()]
    deduped: list[str] = []
    for part in parts:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    return " ".join(deduped)


_STATE: dict = {}
# greedy 루프 자체는 무상태지만, 동시 요청이 GPU/CPU를 두고 경합하지 않게 직렬화한다.
_LOCK = threading.Lock()
# 대기열 상한(진행 1 + 대기 7). 로컬(127.0.0.1)에선 프록시 레이트리밋이 1차 방어지만,
# 원격 배포 시 포트가 직접 노출되므로 서비스 자체도 무한 대기열을 거부해야 한다.
_SLOTS = threading.Semaphore(8)


class _Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server 규약)
        if self.path == "/health":
            self._json(200, {
                "status": "ok",
                "device": _STATE["device"],
                "direction": _STATE["direction"],
            })
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/translate":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json(400, {"error": "invalid json"})
            return
        text = (body.get("text") or "").strip()
        to = body.get("to")
        if not text or to not in ("en", "ko"):
            self._json(400, {"error": "required: text, to('en'|'ko')"})
            return
        if not _SLOTS.acquire(blocking=False):
            self._json(429, {"error": "번역 요청이 몰려 있습니다. 잠시 후 다시 시도하세요."})
            return
        try:
            with _LOCK:
                outs = [
                    _clean_subtitle_artifacts(
                        sent,
                        greedy_translate(
                            _STATE["model"], _STATE["tokenizer"], sent,
                            _STATE["device"], target_lang=to, max_new=_STATE["max_new"],
                        ),
                    )
                    for sent in split_sentences(text)
                ]
            self._json(200, {"translation": " ".join(o.strip() for o in outs)})
        except Exception as exc:  # 서비스는 죽지 않고 오류만 돌려준다
            self._json(500, {"error": str(exc)})
        finally:
            _SLOTS.release()


def _main() -> None:
    ap = argparse.ArgumentParser(description="번역 HTTP 서비스")
    ap.add_argument("--artifacts", default="runs/hf-sentence",
                    help="tokenizer.json + best.pt 디렉터리")
    ap.add_argument("--port", type=int, default=8531)
    ap.add_argument("--max-new", type=int, default=128,
                    help="문장당 최대 생성 토큰(긴 문장 잘림 방지용으로 CLI 기본 64보다 크게)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = Path(args.artifacts)
    _STATE["tokenizer"] = load_tokenizer(art / "tokenizer.json")
    _STATE["model"], _STATE["direction"] = _load_model(art / "best.pt", device)
    _STATE["device"] = device
    _STATE["max_new"] = args.max_new
    print(f"[serve] device={device} direction={_STATE['direction']} port={args.port}")

    ThreadingHTTPServer(("127.0.0.1", args.port), _Handler).serve_forever()


if __name__ == "__main__":
    _main()
