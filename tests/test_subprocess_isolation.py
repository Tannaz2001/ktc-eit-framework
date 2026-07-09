import shutil

import numpy as np
import scipy.io

from src.ktc_framework.methods.subprocess_wrapper import prepare_isolated_input


def test_prepare_isolated_input_only_contains_selected_sample(tmp_path):
    """The temp input dir must hold only the chosen data file + ref.mat,
    not the other samples that share its level directory."""
    level_dir = tmp_path / "level3"
    level_dir.mkdir()
    for n in (1, 2, 3):
        scipy.io.savemat(str(level_dir / f"data{n}.mat"), {"Injref": np.zeros((32, 32))})
    # _find_ref_file() checks data_file.parent first, so this is discovered
    # without needing to set $KTC_DATASET_ROOT.
    scipy.io.savemat(str(level_dir / "ref.mat"), {"Uref": np.zeros((32, 32))})

    data_file = level_dir / "data2.mat"

    isolated_dir = prepare_isolated_input(data_file, level=3)
    try:
        entries = sorted(p.name for p in isolated_dir.iterdir())
        assert entries == ["data2.mat", "ref.mat"]
    finally:
        shutil.rmtree(isolated_dir, ignore_errors=True)
