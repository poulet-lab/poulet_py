from datetime import datetime, timezone
from os import chdir, getcwd
from pathlib import Path
from tempfile import TemporaryDirectory

from pytest import fixture, mark

from poulet_py import check_or_create, define_folder_name, go_to, sanitize_path


class TestCheckOrCreate:
    def test_existing_dir(self, tmp_path):
        """Should not raise when directory exists"""
        existing_dir = tmp_path / "existing"
        existing_dir.mkdir()
        check_or_create(existing_dir)  # Should not raise
        assert existing_dir.exists()

    def test_new_dir_creation(self, tmp_path):
        """Should create new directory when it doesn't exist"""
        new_dir = tmp_path / "new_directory"
        check_or_create(new_dir)
        assert new_dir.exists()

    def test_nested_dir_creation(self, tmp_path):
        """Should create nested directory structure"""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        check_or_create(nested_dir)
        assert nested_dir.exists()


class TestDefineFolderName:
    def test_sanitize_special_chars(self):
        """Should replace special characters with underscores"""
        result = define_folder_name("test@name#123!", add_date=False)
        assert result == "test_name_123_"

    def test_with_date(self):
        """Should prepend date when add_date=True"""
        result = define_folder_name("project", add_date=True)
        assert result == f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_project"

    def test_without_date(self):
        """Should not prepend date when add_date=False"""
        result = define_folder_name("project", add_date=False)
        assert result == "project"


class TestSanitizePath:
    @mark.parametrize(
        "input_path,expected",
        [
            ("simple/path", "simple/path"),
            ("s$pecial/ch@rs", "s_pecial/ch_rs"),
            ("with space/and.dot", "with_space/and.dot"),
            ("UPPER/CASE", "upper/case"),
            ("../relative/path", "../relative/path"),
            (r"C:\Windows\Style", r"c:/windows/style"),
        ],
    )
    def test_sanitization(self, input_path, expected):
        """Should properly sanitize different path components"""
        result = sanitize_path(input_path)
        assert str(result) == expected

    def test_add_timestamp(self):
        """Should prepend timestamp to last component when requested"""
        result = sanitize_path("path/to/file.txt", add_timestamp=True)
        assert str(result) == f"path/to/{datetime.now(timezone.utc).strftime('%Y%m%d')}_file.txt"

    def test_path_object_input(self):
        """Should work with Path objects as input"""
        path_obj = Path("some/path")
        result = sanitize_path(path_obj)
        assert str(result) == "some/path"

    def test_root_path_preservation(self):
        """Should preserve root path components"""
        result = sanitize_path("/root/path")
        assert str(result) == "/root/path"
        assert str(result).startswith("/")

    def test_windows_path_preservation(self):
        """Should preserve Windows drive letters"""
        result = sanitize_path("C:/Windows/Path")
        assert str(result).lower() == "c:/windows/path"
        assert str(result).startswith("c:")

    def test_multiple_special_chars(self):
        """Should handle multiple consecutive special characters"""
        result = sanitize_path("a!!b@@c##d")
        assert str(result) == "a__b__c__d"


class TestPathUtils:
    """Test class for path utility functions."""

    @fixture(autouse=True)
    def setup_teardown(self):
        """Fixture to save/restore CWD before/after each test."""
        self.original_cwd = getcwd()
        yield
        chdir(self.original_cwd)

    def test_with_explicit_path(self):
        """Test successful directory change when key exists."""
        with TemporaryDirectory() as tmpdir:
            test_path = Path(tmpdir) / "a/b/c/d/e/f.txt"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            result = go_to("c", path=test_path)
            assert result is True
            assert Path(getcwd()).resolve() == Path(tmpdir).resolve() / "a/b/c"

    def test_go_to_not_found(self):
        """Test failure when key is missing."""
        assert go_to("missing_key", path="a/b/c/d.txt") is False

    def test_go_to_default_path(self):
        """Test that __file__ is used when path is not specified."""
        success = go_to("poulet_py")
        assert success is True
        assert (
            Path(getcwd()).resolve() / "tests/tools/test_organizational.py"
            == Path(__file__).resolve()
        )
