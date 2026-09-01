"""공용 도구 패키지의 경로 설정."""

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = SRC_DIR.parent
