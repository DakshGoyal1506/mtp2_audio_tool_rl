import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "scripts" / "setup" / "check_repo_setup.py"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_repo_setup", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blocked_artifact_extensions_are_detectable():
    checker = load_checker()

    assert Path("model.pt").suffix in checker.BLOCKED_EXTENSIONS
    assert Path("audio.wav").suffix in checker.BLOCKED_EXTENSIONS
    assert Path("dataset.parquet").suffix in checker.BLOCKED_EXTENSIONS
    assert Path("notes.md").suffix not in checker.BLOCKED_EXTENSIONS


def test_allowed_intended_dirs_are_not_flagged(tmp_path):
    checker = load_checker()
    root = tmp_path

    assert checker.is_allowed_dir(root / "scripts" / "data", root)
    assert checker.is_allowed_dir(root / "scripts" / "data" / "cache_tools", root)
    assert checker.is_allowed_dir(root / "src" / "mtp2_audio_tool_rl" / "datasets", root)
    assert checker.is_allowed_dir(root / "configs" / "datasets", root)
    assert not checker.is_allowed_dir(root / "datasets", root)
    assert not checker.is_allowed_dir(root / "cache_tools", root)


def test_load_submodule_paths_from_gitmodules(tmp_path):
    checker = load_checker()
    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text(
        """
[submodule "third_party/Audio-Maestro"]
	path = third_party/Audio-Maestro
	url = https://example.invalid/Audio-Maestro.git
[submodule "third_party/ToolRL"]
	path = third_party/ToolRL
	url = https://example.invalid/ToolRL.git
""".lstrip(),
        encoding="utf-8",
    )

    paths = checker.load_submodule_paths(tmp_path)

    assert paths == [Path("third_party/Audio-Maestro"), Path("third_party/ToolRL")]
    assert checker.is_submodule_path(Path("third_party/Audio-Maestro"), paths)
    assert not checker.is_submodule_path(Path("third_party"), paths)
