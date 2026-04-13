"""Unit tests for data/dataset.py."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import AIDataset, get_dataloaders, get_transforms, prepare_splits


# ---------------------------------------------------------------------------
# get_transforms
# ---------------------------------------------------------------------------

class TestGetTransforms:
    def test_returns_two_compose_pipelines(self):
        train_tf, eval_tf = get_transforms()
        assert isinstance(train_tf, transforms.Compose)
        assert isinstance(eval_tf, transforms.Compose)

    def test_train_has_random_horizontal_flip(self):
        train_tf, _ = get_transforms()
        types = [type(t) for t in train_tf.transforms]
        assert transforms.RandomHorizontalFlip in types

    def test_eval_has_center_crop(self):
        _, eval_tf = get_transforms()
        types = [type(t) for t in eval_tf.transforms]
        assert transforms.CenterCrop in types

    def test_eval_does_not_have_random_flip(self):
        _, eval_tf = get_transforms()
        types = [type(t) for t in eval_tf.transforms]
        assert transforms.RandomHorizontalFlip not in types

    def test_both_end_with_normalize(self):
        train_tf, eval_tf = get_transforms()
        assert isinstance(train_tf.transforms[-1], transforms.Normalize)
        assert isinstance(eval_tf.transforms[-1], transforms.Normalize)


# ---------------------------------------------------------------------------
# AIDataset
# ---------------------------------------------------------------------------

class TestAIDataset:
    def test_len(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        assert len(ds) == len(df)

    def test_getitem_types(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, torch.Tensor)

    def test_output_shape(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        image, _ = ds[0]
        assert image.shape == (3, 224, 224)

    def test_labels_are_0_or_1(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        for i in range(len(ds)):
            _, label = ds[i]
            assert label.item() in (0, 1)

    def test_empty_dataset(self, tmp_path):
        df = pd.DataFrame({"file_name": [], "label": []})
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        assert len(ds) == 0

    def test_no_transform_returns_pil_image(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        ds = AIDataset(df, tmp_path, transform=None)
        image, _ = ds[0]
        assert isinstance(image, Image.Image)

    def test_corrupted_image_returns_placeholder(self, tmp_path):
        """Since Round 2, a corrupted file returns a zero-filled placeholder
        tensor instead of crashing."""
        bad = tmp_path / "corrupt.jpg"
        bad.write_bytes(b"not a real image")
        df = pd.DataFrame({"file_name": ["corrupt.jpg"], "label": [0]})
        _, eval_tf = get_transforms()
        ds = AIDataset(df, tmp_path, transform=eval_tf)
        tensor, label = ds[0]  # must not raise
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)
        assert ds.corrupt_count == 1


# ---------------------------------------------------------------------------
# prepare_splits
# ---------------------------------------------------------------------------

class TestPrepareSplits:
    def _make_df(self, n: int = 100) -> pd.DataFrame:
        return pd.DataFrame({
            "file_name": [f"img_{i:04d}.jpg" for i in range(n)],
            "label":     [i % 2 for i in range(n)],
        })

    def test_no_train_val_overlap(self):
        df = self._make_df()
        df_train, df_val, _ = prepare_splits(df, None, 0.2, 0.2, seed=42)
        assert set(df_train["file_name"]).isdisjoint(set(df_val["file_name"]))

    def test_no_train_test_overlap(self):
        df = self._make_df()
        df_train, _, df_test = prepare_splits(df, None, 0.2, 0.2, seed=42)
        assert set(df_train["file_name"]).isdisjoint(set(df_test["file_name"]))

    def test_no_val_test_overlap(self):
        df = self._make_df()
        _, df_val, df_test = prepare_splits(df, None, 0.2, 0.2, seed=42)
        assert set(df_val["file_name"]).isdisjoint(set(df_test["file_name"]))

    def test_all_samples_accounted_for(self):
        df = self._make_df()
        df_train, df_val, df_test = prepare_splits(df, None, 0.2, 0.2, seed=42)
        assert len(df_train) + len(df_val) + len(df_test) == len(df)

    def test_approximate_ratios(self):
        df = self._make_df(100)
        df_train, df_val, df_test = prepare_splits(df, None, 0.2, 0.2, seed=42)
        assert 55 <= len(df_train) <= 65
        assert 15 <= len(df_val) <= 25
        assert 15 <= len(df_test) <= 25

    def test_train_size_limits_pool(self):
        df = self._make_df(100)
        df_train, df_val, df_test = prepare_splits(df, train_size=50, val_ratio=0.2, test_ratio=0.2, seed=42)
        assert len(df_train) + len(df_val) + len(df_test) == 50

    def test_same_seed_gives_same_split(self):
        df = self._make_df()
        a_train, _, _ = prepare_splits(df, None, 0.2, 0.2, seed=0)
        b_train, _, _ = prepare_splits(df, None, 0.2, 0.2, seed=0)
        assert list(a_train["file_name"]) == list(b_train["file_name"])

    def test_different_seeds_give_different_splits(self):
        df = self._make_df()
        a_train, _, _ = prepare_splits(df, None, 0.2, 0.2, seed=0)
        b_train, _, _ = prepare_splits(df, None, 0.2, 0.2, seed=99)
        assert list(a_train["file_name"]) != list(b_train["file_name"])

    def test_missing_column_raises(self):
        """DataFrame without required columns must raise ValueError."""
        df = pd.DataFrame({"filename": ["a.jpg"], "label": [0]})  # wrong col name
        with pytest.raises(ValueError, match="missing required columns"):
            prepare_splits(df, None, 0.2, 0.2, seed=0)

    def test_invalid_labels_raise(self):
        """Labels outside {0, 1} must raise ValueError."""
        df = self._make_df()
        df.loc[0, "label"] = 2  # inject bad label
        with pytest.raises(ValueError, match="contain only 0 or 1"):
            prepare_splits(df, None, 0.2, 0.2, seed=0)


# ---------------------------------------------------------------------------
# get_dataloaders
# ---------------------------------------------------------------------------

class TestGetDataloaders:
    def test_returns_three_dataloaders(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        train_loader, val_loader, test_loader = get_dataloaders(
            df.iloc[:12].copy(), df.iloc[12:16].copy(), df.iloc[16:].copy(),
            data_root=tmp_path, batch_size=4, num_workers=0,
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_batch_size_respected(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        train_loader, _, _ = get_dataloaders(
            df.iloc[:12].copy(), df.iloc[12:16].copy(), df.iloc[16:].copy(),
            data_root=tmp_path, batch_size=4, num_workers=0,
        )
        imgs, _ = next(iter(train_loader))
        assert imgs.shape[0] == 4

    def test_image_channels_and_size(self, tmp_image_dir):
        tmp_path, df = tmp_image_dir
        train_loader, _, _ = get_dataloaders(
            df.iloc[:12].copy(), df.iloc[12:16].copy(), df.iloc[16:].copy(),
            data_root=tmp_path, batch_size=4, num_workers=0,
        )
        imgs, _ = next(iter(train_loader))
        assert imgs.shape[1:] == (3, 224, 224)
