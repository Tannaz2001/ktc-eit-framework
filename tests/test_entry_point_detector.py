import pytest

from src.ktc_framework.methods.entry_detector import (
    CONTRACT_CLI_SCRIPT,
    CONTRACT_METHOD_PLUGIN,
    detect_entry_point,
)


def test_detects_cli_script_via_argparse_and_main_guard(tmp_path):
    (tmp_path / "main.py").write_text(
        "import argparse\n"
        "\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('input_dir')\n"
        "    parser.add_argument('output_dir')\n"
        "    parser.add_argument('level')\n"
        "    args = parser.parse_args()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )

    entry_path, contract = detect_entry_point(tmp_path)

    assert entry_path.name == "main.py"
    assert contract == CONTRACT_CLI_SCRIPT


def test_detects_method_plugin_class_with_reconstruct(tmp_path):
    (tmp_path / "solver.py").write_text(
        "from ktc_framework.methods.method_plugin import MethodPlugin\n"
        "\n"
        "class MySolver(MethodPlugin):\n"
        "    def reconstruct(self, batch):\n"
        "        return None\n",
        encoding="utf-8",
    )

    entry_path, contract = detect_entry_point(tmp_path)

    assert entry_path.name == "solver.py"
    assert contract == CONTRACT_METHOD_PLUGIN


def test_raises_when_no_entry_point_contract_matches(tmp_path):
    (tmp_path / "utils.py").write_text(
        "def helper(x):\n"
        "    return x + 1\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="utils.py"):
        detect_entry_point(tmp_path)
