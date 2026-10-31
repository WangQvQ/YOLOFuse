"""Regression coverage for public RGB/RGB+IR data, prediction and export APIs."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ultralytics import YOLO
from ultralytics.data.augment import Format, RandomPerspective
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.loaders import LoadImagesAndVideos
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ROOT
from ultralytics.cfg import get_cfg

FUSE_YAMLS = sorted((ROOT / "cfg/models/fuse").glob("*.yaml"))


@pytest.fixture
def paired_data(tmp_path):
    """Distinct channel values expose modality swaps that shape-only tests miss."""
    for folder, color in (("images", (10, 20, 30)), ("imagesIR", (70, 80, 90))):
        for split in ("train", "val"):
            d = tmp_path / folder / split
            d.mkdir(parents=True)
            # IR deliberately differs in resolution.
            shape = (32, 48, 3) if folder == "imagesIR" else (64, 64, 3)
            for i in range(2):
                cv2.imwrite(str(d / f"{i}.png"), np.full(shape, color, dtype=np.uint8))
    for split in ("train", "val"):
        d = tmp_path / "labels" / split
        d.mkdir(parents=True)
        for i in range(2):
            (d / f"{i}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
    return tmp_path


def dataset(root, ch, cache=False):
    return YOLODataset(
        str(root / "images/train"),
        data={"names": {0: "person"}, "nc": 1},
        imgsz=64,
        augment=False,
        cache=cache,
        hyp=get_cfg(overrides={"ch": ch}),
    )


@pytest.mark.parametrize("ch", [3, 6])
def test_loader_and_format_order(paired_data, ch):
    path = paired_data / "images/train/0.png"
    _, images, _ = next(iter(LoadImagesAndVideos(str(path), ch=ch)))
    raw = images[0]
    expected = [30, 20, 10] + ([90, 80, 70] if ch == 6 else [])
    formatted = Format()._format_img(raw)
    assert formatted[:, 0, 0].tolist() == expected
    ds = dataset(paired_data, ch)
    assert torch.equal(ds[0]["img"], formatted)
    predictor = BasePredictor(overrides={"imgsz": 64})
    predictor.device = torch.device("cpu")
    predictor.model = SimpleNamespace(ch=ch, fp16=False, pt=True, stride=32)
    predictor.imgsz = [64, 64]
    torch.testing.assert_close(predictor.preprocess([raw])[0], formatted.float() / 255)


def test_single_path_without_images_directory(tmp_path):
    p = tmp_path / "sample.png"
    cv2.imwrite(str(p), np.zeros((32, 32, 3), np.uint8))
    assert next(iter(LoadImagesAndVideos(str(p))))[1][0].shape == (32, 32, 3)


def test_missing_ir_falls_back_consistently(paired_data):
    (paired_data / "imagesIR/train/0.png").unlink()
    path = paired_data / "images/train/0.png"
    img = next(iter(LoadImagesAndVideos(str(path), ch=6)))[1][0]
    np.testing.assert_array_equal(img[..., :3], img[..., 3:])
    np.testing.assert_array_equal(dataset(paired_data, 6).load_image(0)[0], img)


@pytest.mark.parametrize("ch", [3, 6])
@pytest.mark.parametrize("bad_cache", ["corrupt", "other_modality"])
def test_cache_recovers(paired_data, ch, bad_cache):
    ds = dataset(paired_data, ch)
    p = ds.npy_files[0]
    if bad_cache == "corrupt":
        p.write_bytes(b"broken npy")
    else:
        np.save(p, np.zeros((64, 64, 9 - ch), np.uint8))
    raw = ds.load_image(0)[0]
    assert raw.shape == (64, 64, ch)
    if ch == 6:
        assert raw[0, 0].tolist() == [10, 20, 30, 70, 80, 90]
    ds.cache_images_to_disk(0)
    np.testing.assert_array_equal(np.load(p), raw)


def test_single_dual_disk_cache_isolation(paired_data):
    rgb = dataset(paired_data, 3, "disk")
    dual = dataset(paired_data, 6, "disk")
    assert rgb.npy_files[0] != dual.npy_files[0]
    assert rgb.load_image(0)[0].shape[-1] == 3
    assert dual.load_image(0)[0].shape[-1] == 6


@pytest.mark.parametrize("perspective", [0.0, 0.001])
def test_geometric_padding_all_six_channels(perspective):
    aug = RandomPerspective(degrees=0, translate=0, scale=0, shear=0, perspective=perspective)
    aug.size = (96, 96)
    image = np.full((32, 32, 6), 50, dtype=np.uint8)
    out, _, _ = aug.affine_transform(image, border=(32, 32))
    assert out.shape == (96, 96, 6)
    assert (out[0, 0] == 114).all()


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("source_type", ["file", "numpy", "tensor"])
def test_public_predict(paired_data, dual, source_type):
    model = YOLO(str(FUSE_YAMLS[0]) if dual else "yolov8n.yaml")
    path = paired_data / "images/train/0.png"
    if source_type == "file":
        source = str(path)
    else:
        raw = next(iter(LoadImagesAndVideos(str(path), ch=6 if dual else 3)))[1][0]
        source = raw if source_type == "numpy" else Format()._format_img(raw).float().unsqueeze(0) / 255
    results = model.predict(
        source,
        imgsz=64,
        device="cpu",
        save=True,
        verbose=False,
        project=str(paired_data / "runs"),
        name=f"{dual}-{source_type}",
    )
    assert len(results) == 1
    assert results[0].orig_img.shape == (64, 64, 3)
    assert len(list(Path(model.predictor.save_dir).glob("*.jpg"))) == (2 if dual else 1)
    if source_type != "tensor":
        np.testing.assert_array_equal(results[0].orig_img[0, 0], [10, 20, 30])
    assert len(model.predictor.ir_results) == (1 if dual else 0)


def test_training_channel_error_is_actionable():
    trainer = object.__new__(DetectionTrainer)
    trainer.args = SimpleNamespace(ch=3)
    trainer.data = {"nc": 1}
    with pytest.raises(ValueError, match="Use ch=6"):
        trainer.get_model(str(FUSE_YAMLS[0]), verbose=False)


@pytest.mark.parametrize("yaml_path", FUSE_YAMLS, ids=lambda x: x.stem)
@pytest.mark.parametrize("fmt", ["torchscript", "onnx"])
def test_export_round_trip(tmp_path, yaml_path, fmt):
    if fmt == "onnx":
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")
    model = YOLO(str(yaml_path))
    model.model.pt_path = str(tmp_path / f"{yaml_path.stem}.pt")
    exported = model.export(format=fmt, imgsz=64, device="cpu", simplify=False, opset=17)
    backend = AutoBackend(exported, device=torch.device("cpu"), verbose=False)
    assert backend.ch == 6
    model.model.eval()
    x = torch.rand(1, 6, 64, 64)
    with torch.no_grad():
        expected = model.model(x)[0]
        actual = backend(x)
    if isinstance(actual, (list, tuple)):
        actual = actual[0]
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert len(YOLO(exported, task="detect").predict(x, imgsz=64, verbose=False)) == 1


@pytest.mark.parametrize("yaml_path", FUSE_YAMLS, ids=lambda x: x.stem)
def test_dynamic_onnx_shapes(tmp_path, yaml_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = YOLO(str(yaml_path))
    model.model.pt_path = str(tmp_path / "dynamic.pt")
    exported = model.export(format="onnx", imgsz=64, dynamic=True, simplify=True, opset=17)
    session = ort.InferenceSession(exported, providers=["CPUExecutionProvider"])
    for shape in ((1, 6, 64, 64), (2, 6, 96, 64)):
        x = torch.rand(shape)
        with torch.no_grad():
            expected = model.model.eval()(x)[0]
        actual = session.run(None, {"images": x.numpy()})[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-3)


def test_legacy_checkpoint_order_adapter():
    """Old weights retain their learned modality assignment with canonical new inputs."""
    model = YOLO(str(FUSE_YAMLS[2])).model.eval()
    canonical = torch.rand(1, 6, 64, 64)
    with torch.no_grad():
        expected = model(torch.cat((canonical[:, 3:], canonical[:, :3]), dim=1))[0]
        del model.yaml["input_order"]  # simulate a pre-fix checkpoint
        actual = model(canonical)[0]
    torch.testing.assert_close(actual, expected)


def test_relative_ir_path():
    assert BaseDataset._get_ir_path("images/train/a.png") == "imagesIR/train/a.png"


@pytest.mark.parametrize("ch", [3, 6])
@pytest.mark.parametrize("cache", [False, "disk"])
def test_one_epoch_train_and_standalone_val(paired_data, ch, cache, monkeypatch):
    # Version checks are unrelated to the training regression and require network access.
    monkeypatch.setattr("ultralytics.utils.checks.check_pip_update_available", lambda: None)
    config = paired_data / "data.yaml"
    config.write_text(f"path: {paired_data}\ntrain: images/train\nval: images/val\nnames: [person]\n")
    model = YOLO(str(FUSE_YAMLS[2]) if ch == 6 else "yolov8n.yaml")
    model.train(
        data=str(config),
        ch=ch,
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        amp=False,
        cache=cache,
        plots=False,
        project=str(paired_data / "runs"),
        name="train",
    )
    # A fresh validator must infer channels from the checkpoint without ch=6.
    checkpoint = model.trainer.best
    loaded = YOLO(checkpoint)
    loaded.val(data=str(config), imgsz=64, batch=2, device="cpu", plots=False)
    assert loaded.model.yaml["ch"] == ch


@pytest.mark.parametrize("ch", [3, 6])
def test_onnx_nms_and_validation(paired_data, ch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    config = paired_data / "data.yaml"
    config.write_text(f"path: {paired_data}\ntrain: images/train\nval: images/val\nnames: [person]\n")
    model = YOLO(str(FUSE_YAMLS[2]) if ch == 6 else "yolov8n.yaml")
    model.model.pt_path = str(paired_data / "nms.pt")
    exported = model.export(format="onnx", imgsz=64, nms=True, opset=17, simplify=False, max_det=10)
    loaded = YOLO(exported, task="detect")
    loaded.val(data=str(config), imgsz=64, device="cpu", plots=False)
    results = loaded.predict(str(paired_data / "images/val/0.png"), imgsz=64, verbose=False)
    assert len(results) == 1 and results[0].orig_img.shape[-1] == 3


def test_legacy_weight_transfer_preserves_order():
    legacy = YOLO(str(FUSE_YAMLS[2])).model
    del legacy.yaml["input_order"]
    target = YOLO(str(FUSE_YAMLS[2])).model
    target.load(legacy)
    assert target.yaml["input_order"] == "ir_rgb"


def test_training_legacy_yaml_is_not_mutated():
    legacy = YOLO(str(FUSE_YAMLS[2])).model
    del legacy.yaml["input_order"]
    trainer = object.__new__(DetectionTrainer)
    trainer.args = SimpleNamespace(ch=6)
    trainer.data = {"nc": 80}
    target = trainer.get_model(legacy.yaml, weights=legacy, verbose=False)
    assert "input_order" not in legacy.yaml
    assert target.yaml["input_order"] == "ir_rgb"
