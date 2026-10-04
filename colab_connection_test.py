"""GitHub から取得して Colab で実行する接続確認用スクリプト。"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


REVISION_MARKER = "2026-10-04-colab-connection-test-v1"
CALCULATION_INPUT = range(1, 101)
CALCULATION_EXPECTED = 5050


def is_colab_runtime() -> bool:
    """Google Colab の実行環境らしさを、値を表示せずに判定する。"""

    return "COLAB_RELEASE_TAG" in os.environ or Path("/content").is_dir()


def verify_fixed_calculation() -> int:
    """固定値の計算結果を検証し、異常時は例外を送出する。"""

    actual = sum(CALCULATION_INPUT)
    if actual != CALCULATION_EXPECTED:
        raise RuntimeError(
            f"固定計算に失敗しました: 期待値={CALCULATION_EXPECTED}, 実測値={actual}"
        )
    return actual


def main() -> int:
    """環境と計算を確認し、成功なら0、異常なら1を返す。"""

    print(f"revision marker: {REVISION_MARKER}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Colab判定: {'はい' if is_colab_runtime() else 'いいえ'}")

    try:
        actual = verify_fixed_calculation()
    except Exception as exc:
        print("結果: 失敗", file=sys.stderr)
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print(f"固定計算: 1から100までの合計={actual}（検証成功）")
    print("結果: 成功")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
