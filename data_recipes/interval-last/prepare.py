from __future__ import annotations

from pathlib import Path

from nanogptpro.utils.interval_task_prepare_lib import main

_SCRIPT_DIR = Path(__file__).resolve().parent


if __name__ == "__main__":
    main(
        script_dir=_SCRIPT_DIR,
        dataset_label="interval_last",
        fixed_task_mode="interval_last",
    )
