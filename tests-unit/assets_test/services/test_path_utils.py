"""Tests for path_utils – asset category resolution."""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.assets.services.path_utils import (
    get_asset_category_and_relative_path,
    get_name_and_tags_from_asset_path,
    resolve_destination_from_tags,
)


@pytest.fixture
def fake_dirs():
    """Create temporary input, output, and temp directories."""
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        input_dir = root_path / "input"
        output_dir = root_path / "output"
        temp_dir = root_path / "temp"
        models_dir = root_path / "models" / "checkpoints"
        for d in (input_dir, output_dir, temp_dir, models_dir):
            d.mkdir(parents=True)

        with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
            mock_fp.get_input_directory.return_value = str(input_dir)
            mock_fp.get_output_directory.return_value = str(output_dir)
            mock_fp.get_temp_directory.return_value = str(temp_dir)

            with patch(
                "app.assets.services.path_utils.get_comfy_models_folders",
                return_value=[("checkpoints", [str(models_dir)])],
            ):
                yield {
                    "input": input_dir,
                    "output": output_dir,
                    "temp": temp_dir,
                    "models": models_dir,
                }


class TestGetAssetCategoryAndRelativePath:
    def test_input_file(self, fake_dirs):
        f = fake_dirs["input"] / "photo.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "input"
        assert rel == "photo.png"

    def test_output_file(self, fake_dirs):
        f = fake_dirs["output"] / "result.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "output"
        assert rel == "result.png"

    def test_temp_file(self, fake_dirs):
        """Regression: temp files must be categorised, not raise ValueError."""
        f = fake_dirs["temp"] / "GLSLShader_output_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert rel == "GLSLShader_output_00004_.png"

    def test_temp_file_in_subfolder(self, fake_dirs):
        sub = fake_dirs["temp"] / "sub"
        sub.mkdir()
        f = sub / "ComfyUI_temp_tczip_00004_.png"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "temp"
        assert os.path.normpath(rel) == os.path.normpath("sub/ComfyUI_temp_tczip_00004_.png")

    def test_model_file(self, fake_dirs):
        f = fake_dirs["models"] / "model.safetensors"
        f.touch()
        cat, rel = get_asset_category_and_relative_path(str(f))
        assert cat == "models"

    def test_model_path_tags_include_registered_model_type_only(self, fake_dirs):
        f = fake_dirs["models"] / "subdir" / "model.safetensors"
        f.parent.mkdir()
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "checkpoints" not in tags
        assert "subdir" not in tags

    def test_model_type_preserves_registered_folder_case(self, fake_dirs):
        llm_dir = fake_dirs["models"].parent / "LLM"
        llm_dir.mkdir()
        f = llm_dir / "model.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("LLM", [str(llm_dir)])],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:LLM" in tags
        assert "model_type:llm" not in tags

    def test_path_components_do_not_create_model_type_tags(self, fake_dirs):
        f = fake_dirs["models"] / "loras" / "model.safetensors"
        f.parent.mkdir()
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "loras" not in tags
        assert "model_type:loras" not in tags

    def test_shared_root_returns_all_matching_model_type_tags(self, fake_dirs):
        shared_root = fake_dirs["models"].parent / "shared"
        shared_root.mkdir()
        f = shared_root / "foo.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[
                ("checkpoints", [str(shared_root)]),
                ("loras", [str(shared_root)]),
            ],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "model_type:loras" in tags

    def test_output_backed_registered_folder_gets_model_and_output_tags(self, fake_dirs):
        output_checkpoints_dir = fake_dirs["output"] / "checkpoints"
        output_checkpoints_dir.mkdir()
        f = output_checkpoints_dir / "saved.safetensors"
        f.touch()

        with patch(
            "app.assets.services.path_utils.get_comfy_models_folders",
            return_value=[("checkpoints", [str(output_checkpoints_dir)])],
        ):
            _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "models" in tags
        assert "model_type:checkpoints" in tags
        assert "output" in tags

    def test_temp_path_tags_include_temp_not_output_or_preview(self, fake_dirs):
        f = fake_dirs["temp"] / "preview.png"
        f.touch()

        _name, tags = get_name_and_tags_from_asset_path(str(f))

        assert "temp" in tags
        assert "output" not in tags
        assert "preview:true" not in tags

    def test_unknown_path_raises(self, fake_dirs):
        with pytest.raises(ValueError, match="not within"):
            get_asset_category_and_relative_path("/some/random/path.png")


class TestResolveDestinationFromTags:
    def test_model_upload_rejects_non_writable_registered_folders(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            checkpoints_dir = root_path / "models" / "checkpoints"
            configs_dir = root_path / "models" / "configs"
            custom_nodes_dir = root_path / "custom_nodes"
            for path in (checkpoints_dir, configs_dir, custom_nodes_dir):
                path.mkdir(parents=True)

            with patch("app.assets.services.path_utils.folder_paths") as mock_fp:
                mock_fp.folder_names_and_paths = {
                    "checkpoints": ([str(checkpoints_dir)], set()),
                    "configs": ([str(configs_dir)], set()),
                    "custom_nodes": ([str(custom_nodes_dir)], set()),
                }

                base_dir, subdirs = resolve_destination_from_tags(
                    ["models", "model_type:checkpoints"]
                )
                assert base_dir == os.path.abspath(checkpoints_dir)
                assert subdirs == []

                for folder_name in ("configs", "custom_nodes"):
                    with pytest.raises(ValueError, match="unknown model category"):
                        resolve_destination_from_tags(
                            ["models", f"model_type:{folder_name}"]
                        )
