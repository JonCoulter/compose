#!/usr/bin/env python3
"""Sanity check: local-pattern MM generator (no diffusers/torch I² required)."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import types


def main() -> int:
    mpo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, mpo)

    from src.model import MM_MODEL_MAPPING, get_mm_model  # type: ignore

    assert "local-pattern" in MM_MODEL_MAPPING, MM_MODEL_MAPPING.keys()
    log_dir = tempfile.mkdtemp(prefix="mpo_smoke_")
    _lg = logging.getLogger("mpo_smoke")
    logger = types.SimpleNamespace(
        log_dir=log_dir,
        info=_lg.info,
        error=_lg.error,
        warning=_lg.warning,
    )
    gen = get_mm_model("local-pattern")(
        mm_generator_model_name="local-pattern",
        logger=logger,
    )
    path = gen.generate("smoke test prompt")
    if not path or not os.path.isfile(path):
        print("FAIL: expected a JPEG path, got", path)
        return 1
    print("OK:", type(gen).__name__, "wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
