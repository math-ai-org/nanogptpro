from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nanogptpro.utils.interval_task_prepare_lib import main

_SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    main(
        script_dir=_SCRIPT_DIR,
        dataset_label="interval_sum",
        fixed_task_mode="interval_sum",
    )
