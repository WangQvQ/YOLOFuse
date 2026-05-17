"""Tests for YOLOFuse dual-modality fusion models."""

import pytest
import torch
import yaml
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel, BaseModel
from ultralytics.nn.modules.block import ModalConcat, ModalitySelector, IdentityInput
from ultralytics.utils import ROOT

FUSE_DIR = ROOT / "cfg" / "models" / "fuse"
FUSE_YAMLS = sorted(FUSE_DIR.glob("*.yaml"))
FUSE_NAMES = [p.stem for p in FUSE_YAMLS]


# ---------------------------------------------------------------------------
# 1. Smoke test: every fusion YAML builds successfully with scale 'n'
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaml_path", FUSE_YAMLS, ids=FUSE_NAMES)
def test_build_fuse_model(yaml_path):
    """Each fusion YAML should construct a DetectionModel without error."""
    model = DetectionModel(str(yaml_path), ch=6, nc=1)
    assert model is not None
    assert isinstance(model, DetectionModel)


# ---------------------------------------------------------------------------
# 2. Forward pass: dual-modality 6ch input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaml_path", FUSE_YAMLS, ids=FUSE_NAMES)
def test_forward_dual(yaml_path):
    """Forward pass with 6-channel dual-modality input should produce valid output."""
    model = DetectionModel(str(yaml_path), ch=6, nc=1)
    model.eval()
    x = torch.randn(1, 6, 64, 64)
    with torch.no_grad():
        preds = model(x)
    # Detect head returns a tensor or list of tensors
    if isinstance(preds, (list, tuple)):
        out = preds[0] if isinstance(preds[0], torch.Tensor) else preds[-1]
    else:
        out = preds
    assert out.ndim == 3  # (B, features, anchors)
    assert out.shape[0] == 1  # batch size


# ---------------------------------------------------------------------------
# 3. Loss computation: dual-modality training step
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("yaml_path", FUSE_YAMLS, ids=FUSE_NAMES)
def test_loss_dual(yaml_path):
    """Loss should be computable for each fusion model with dummy labels."""
    from types import SimpleNamespace

    model = DetectionModel(str(yaml_path), ch=6, nc=1)
    # Set up args namespace required by v8DetectionLoss
    model.args = SimpleNamespace(
        box=7.5,
        cls=0.5,
        dfl=1.5,
        overlap_mask=False,
        nc=1,
    )
    model.train()
    batch = {
        "img": torch.randn(2, 6, 64, 64),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0]),
        "img_id": torch.tensor([0]),
    }
    loss = model.loss(batch)
    # loss() returns (loss_tensor, loss_items)
    loss_val = loss[0] if isinstance(loss, tuple) else loss
    assert isinstance(loss_val, torch.Tensor)
    assert loss_val.ndim == 0  # scalar
    assert loss_val.item() > 0


# ---------------------------------------------------------------------------
# 4. Single-modality: standard ch=3 still works with base YOLO
# ---------------------------------------------------------------------------
def test_build_single_modal():
    """Standard YOLO with ch=3 should still work (backward compatibility)."""
    model = DetectionModel("yolo11n.yaml", ch=3, nc=80)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        preds = model(x)
    assert preds is not None


# ---------------------------------------------------------------------------
# 5. ModalConcat: list input vs tensor input
# ---------------------------------------------------------------------------
def test_modal_concat_list_input():
    """ModalConcat should concatenate a list of two 3ch tensors into 6ch."""
    m = ModalConcat()
    a = torch.randn(2, 3, 8, 8)
    b = torch.randn(2, 3, 8, 8)
    out = m([a, b])
    assert out.shape == (2, 6, 8, 8)


def test_modal_concat_tensor_input():
    """ModalConcat should split a 6ch tensor and re-concatenate."""
    m = ModalConcat()
    x = torch.randn(2, 6, 8, 8)
    out = m(x)
    assert out.shape == (2, 6, 8, 8)


# ---------------------------------------------------------------------------
# 6. ModalitySelector: select RGB (out=1) or IR (out=2)
# ---------------------------------------------------------------------------
def test_modality_selector_tensor():
    """ModalitySelector should split a 6ch tensor and select one half."""
    m1 = ModalitySelector(out=1)
    m2 = ModalitySelector(out=2)
    x = torch.randn(2, 6, 8, 8)
    out1 = m1(x)
    out2 = m2(x)
    assert out1.shape == (2, 3, 8, 8)
    assert out2.shape == (2, 3, 8, 8)
    # out1 should be channels 0-2, out2 should be channels 3-5
    assert torch.equal(out1, x[:, :3])
    assert torch.equal(out2, x[:, 3:])


def test_modality_selector_list():
    """ModalitySelector should pick from a list of two tensors."""
    m1 = ModalitySelector(out=1)
    m2 = ModalitySelector(out=2)
    a = torch.randn(2, 3, 8, 8)
    b = torch.randn(2, 3, 8, 8)
    assert torch.equal(m1([a, b]), a)
    assert torch.equal(m2([a, b]), b)


# ---------------------------------------------------------------------------
# 7. IdentityInput: pass-through
# ---------------------------------------------------------------------------
def test_identity_input():
    """IdentityInput should return input unchanged."""
    m = IdentityInput()
    x = torch.randn(2, 6, 8, 8)
    assert torch.equal(m(x), x)


# ---------------------------------------------------------------------------
# 8. _split_modalities: 6ch tensor → list of two 3ch tensors
# ---------------------------------------------------------------------------
def test_split_modalities():
    """_split_modalities should split 6ch input into [RGB, IR]."""
    x = torch.randn(2, 6, 8, 8)
    result = BaseModel._split_modalities(x)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 3, 8, 8)
    assert result[1].shape == (2, 3, 8, 8)
    assert torch.equal(result[0], x[:, :3])
    assert torch.equal(result[1], x[:, 3:])


def test_split_modalities_noop():
    """_split_modalities should be a no-op for 3ch input."""
    x = torch.randn(2, 3, 8, 8)
    result = BaseModel._split_modalities(x)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (2, 3, 8, 8)


# ---------------------------------------------------------------------------
# 9. IR path helper
# ---------------------------------------------------------------------------
def test_get_ir_path():
    """_get_ir_path should replace /images/ with /imagesIR/ exactly once."""
    from ultralytics.data.base import BaseDataset

    # Normal case
    assert BaseDataset._get_ir_path(
        "/data/dataset/images/train/001.jpg"
    ) == "/data/dataset/imagesIR/train/001.jpg"

    # Only first occurrence
    assert BaseDataset._get_ir_path(
        "/data/images_dataset/images/train/001.jpg"
    ) == "/data/images_dataset/imagesIR/train/001.jpg"

    # No match
    assert BaseDataset._get_ir_path(
        "/data/dataset/frames/train/001.jpg"
    ) == "/data/dataset/frames/train/001.jpg"


# ---------------------------------------------------------------------------
# 10. Param count sanity: fusion models should have more params than base YOLO
# ---------------------------------------------------------------------------
def test_fuse_model_param_count():
    """Fusion models (dual-backbone) should have more parameters than single-backbone."""
    base = DetectionModel("yolo11n.yaml", ch=3, nc=1)
    base_params = sum(p.numel() for p in base.parameters())

    easy_fuse = FUSE_DIR / "Easy-level-Feature-Fusion.yaml"
    fuse = DetectionModel(str(easy_fuse), ch=6, nc=1)
    fuse_params = sum(p.numel() for p in fuse.parameters())

    assert fuse_params > base_params, (
        f"Fusion model ({fuse_params}) should have more params than base ({base_params})"
    )
