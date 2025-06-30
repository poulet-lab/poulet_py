from json import load, loads
from pathlib import Path
from tempfile import TemporaryDirectory

from numpy import array
from pytest import raises

from poulet_py import json_serializer, save_metadata_exp


class TestJsonSerializer:
    """Tests for the json_serializer function."""

    def test_serialize_to_bytes(self):
        """Test serialization to bytes without file output."""
        data = {"key": "value", "num": 42}
        result = json_serializer(data)
        assert isinstance(result, bytes)
        assert loads(result.decode()) == data

    def test_serialize_with_numpy(self):
        """Test serialization with numpy arrays."""
        data = {"array": array([1, 2, 3])}
        result = json_serializer(data)
        assert isinstance(result, bytes)
        assert loads(result.decode()) == {"array": [1, 2, 3]}

    def test_serialize_to_file(self):
        """Test serialization to a file."""
        data = {"key": "value"}
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            json_serializer(data, file_path)

            assert file_path.exists()
            with open(file_path, "rb") as f:
                content = f.read()
                assert loads(content.decode()) == data

    def test_non_json_extension_raises_error(self):
        """Test that non-.json file extension raises ValueError."""
        with raises(ValueError, match="file must end with '.json' extension"):
            json_serializer({}, "test.txt")

    def test_non_serializable_data_raises_error(self):
        """Test that non-serializable data raises TypeError."""

        class NonSerializable:
            pass

        with raises(TypeError, match="Failed to serialize data to JSON"):
            json_serializer({"obj": NonSerializable()})

    def test_parent_directory_creation(self):
        """Test that parent directories are created if they don't exist."""
        data = {"key": "value"}
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "new_dir" / "test.json"
            json_serializer(data, file_path)

            assert file_path.exists()

    def test_string_path_handling(self):
        """Test that string paths are properly handled."""
        data = {"key": "value"}
        with TemporaryDirectory() as temp_dir:
            file_path = str(Path(temp_dir) / "test.json")
            json_serializer(data, file_path)

            assert Path(file_path).exists()

    def test_return_none_when_file_provided(self):
        """Test that function returns None when file path is provided."""
        with TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test.json"
            result = json_serializer({}, file_path)
            assert result is None


class TestSaveMetadataExp:
    """Tests for the deprecated save_metadata_exp function."""

    def test_save_metadata_exp_creates_directory(self):
        """Test that the function creates the directory if it doesn't exist."""
        metadata = {"test": "data"}
        with TemporaryDirectory() as temp_dir:
            new_dir = Path(temp_dir) / "new_dir"
            save_metadata_exp(metadata, new_dir, "test")

            assert (new_dir / "test.json").exists()

    def test_save_metadata_exp_content(self):
        """Test that the function saves correct content."""
        metadata = {"key": "value"}
        with TemporaryDirectory() as temp_dir:
            save_metadata_exp(metadata, temp_dir, "test")

            file_path = Path(temp_dir) / "test.json"
            with open(file_path) as f:
                content = load(f)
                assert content == metadata
