# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
from pathlib import Path

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class DetectionPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a detection model.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.detect import DetectionPredictor

        args = dict(model="yolo11n.pt", source=ASSETS)
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Post-processes predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            self.args.classes,
            self.args.agnostic_nms,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            end2end=getattr(self.model, "end2end", False),
            rotated=self.args.task == "obb",
        )

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        return self.construct_results(preds, img, orig_imgs, **kwargs)

    def construct_results(self, preds, img, orig_imgs):
        """Return RGB results and retain optional IR previews on the predictor."""
        results, self.ir_results = [], []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            results.append(self.construct_result(pred, img, orig_img[..., :3], img_path))
            if orig_img.shape[-1] == 6:
                ir_path = str(Path(img_path).with_name(Path(img_path).stem + "_ir" + Path(img_path).suffix))
                self.ir_results.append(self.construct_result(pred, img, orig_img[..., 3:], ir_path))
        return results

    def construct_result(self, pred, img, orig_img, img_path):
        pred = pred.clone()
        pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
