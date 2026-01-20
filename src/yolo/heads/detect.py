"""Detection heads.

Reference: _reference/yolov9/models/yolo.py
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from yolo.blocks.conv import Conv
from yolo.heads.anchor import dist2bbox, make_anchors
from yolo.heads.dfl import DFL


def _make_divisible(x: float, divisor: int) -> int:
    """Return nearest value divisible by divisor."""
    return math.ceil(x / divisor) * divisor


class DetectDFL(nn.Module):
    """YOLO detection head with Distribution Focal Loss.

    Takes multi-scale feature maps and outputs detection predictions.
    During training, returns raw predictions. During inference, decodes
    boxes and applies sigmoid to class scores.

    Reference: _reference/yolov9/models/yolo.py::DDetect
    """

    def __init__(self, num_classes: int, in_channels: tuple[int, ...]):
        """Initialize detection head.

        Args:
            num_classes: Number of object classes.
            in_channels: Tuple of input channel counts for each feature level.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(in_channels)
        self.reg_max = 16
        self.num_outputs = num_classes + self.reg_max * 4

        c2 = _make_divisible(max(in_channels[0] // 4, self.reg_max * 4, 16), 4)
        c3 = max(in_channels[0], min(num_classes * 2, 128))

        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c2, 3),
                Conv(c2, c2, 3, groups=4),
                nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4),
            )
            for ch in in_channels
        )

        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, num_classes, 1),
            )
            for ch in in_channels
        )

        self.dfl = DFL(self.reg_max)

        # Stride is set during model build (after forward pass with dummy input)
        # Not a buffer - computed at runtime, not saved in state_dict
        self.stride = torch.zeros(self.num_levels)

        # Cached anchors for inference
        self._anchors: Tensor | None = None
        self._strides: Tensor | None = None
        self._shape: tuple[int, ...] | None = None

    def forward(self, x: list[Tensor]) -> list[Tensor] | tuple[Tensor, list[Tensor]]:
        """Forward pass.

        Args:
            x: List of feature tensors from backbone/neck.

        Returns:
            During training: List of raw predictions per level.
            During inference: Tuple of (decoded_predictions, raw_predictions).
        """
        for i in range(self.num_levels):
            x[i] = torch.cat((self.box_convs[i](x[i]), self.cls_convs[i](x[i])), 1)

        if self.training:
            return x

        shape = x[0].shape
        if self._shape != shape:
            self._anchors, self._strides = (
                t.transpose(0, 1) for t in make_anchors(x, self.stride, 0.5)
            )
            self._shape = shape

        batch = shape[0]
        flat = torch.cat([xi.view(batch, self.num_outputs, -1) for xi in x], 2)
        box, cls = flat.split((self.reg_max * 4, self.num_classes), 1)

        assert self._anchors is not None and self._strides is not None
        dbox = dist2bbox(self.dfl(box), self._anchors.unsqueeze(0), xywh=True, dim=1)
        dbox = dbox * self._strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, x

    def init_bias(self) -> None:
        """Initialize biases for better convergence."""
        for i, s in enumerate(self.stride):
            # Box branch bias (last layer of sequential)
            box_seq = self.box_convs[i]
            assert isinstance(box_seq, nn.Sequential)
            box_conv = box_seq[-1]
            assert isinstance(box_conv, nn.Conv2d) and box_conv.bias is not None
            box_conv.bias.data[:] = 1.0
            # Class branch bias (assumes ~5 objects per 640x640 image)
            cls_seq = self.cls_convs[i]
            assert isinstance(cls_seq, nn.Sequential)
            cls_conv = cls_seq[-1]
            assert isinstance(cls_conv, nn.Conv2d) and cls_conv.bias is not None
            cls_conv.bias.data[: self.num_classes] = math.log(
                5 / self.num_classes / (640 / s.item()) ** 2
            )


class DualDetectDFL(nn.Module):
    """Dual YOLO detection head with Distribution Focal Loss.

    Used for auxiliary training branch in YOLOv9. Processes two sets
    of feature maps (main and auxiliary) with separate conv branches.

    Reference: _reference/yolov9/models/yolo.py::DualDDetect
    """

    def __init__(self, num_classes: int, in_channels: tuple[int, ...]):
        """Initialize dual detection head.

        Args:
            num_classes: Number of object classes.
            in_channels: Tuple of input channel counts. First half is auxiliary
                branch, second half is main branch.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_levels = len(in_channels) // 2
        self.reg_max = 16
        self.num_outputs = num_classes + self.reg_max * 4

        ch_aux = in_channels[: self.num_levels]
        ch_main = in_channels[self.num_levels :]

        c2 = _make_divisible(max(ch_aux[0] // 4, self.reg_max * 4, 16), 4)
        c3 = max(ch_aux[0], min(num_classes * 2, 128))
        c4 = _make_divisible(max(ch_main[0] // 4, self.reg_max * 4, 16), 4)
        c5 = max(ch_main[0], min(num_classes * 2, 128))

        self.aux_box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c2, 3),
                Conv(c2, c2, 3, groups=4),
                nn.Conv2d(c2, 4 * self.reg_max, 1, groups=4),
            )
            for ch in ch_aux
        )
        self.aux_cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c3, 3),
                Conv(c3, c3, 3),
                nn.Conv2d(c3, num_classes, 1),
            )
            for ch in ch_aux
        )

        self.main_box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c4, 3),
                Conv(c4, c4, 3, groups=4),
                nn.Conv2d(c4, 4 * self.reg_max, 1, groups=4),
            )
            for ch in ch_main
        )
        self.main_cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(ch, c5, 3),
                Conv(c5, c5, 3),
                nn.Conv2d(c5, num_classes, 1),
            )
            for ch in ch_main
        )

        self.dfl = DFL(self.reg_max)
        self.dfl2 = DFL(self.reg_max)

        # Stride is set during model build (after forward pass with dummy input)
        # Not a buffer - computed at runtime, not saved in state_dict
        self.stride = torch.zeros(self.num_levels)

        # Cached anchors for inference
        self._anchors: Tensor | None = None
        self._strides: Tensor | None = None
        self._shape: tuple[int, ...] | None = None

    def forward(
        self, x: list[Tensor]
    ) -> list[list[Tensor]] | tuple[list[Tensor], list[list[Tensor]]]:
        """Forward pass.

        Args:
            x: List of feature tensors. First num_levels are auxiliary,
               remaining num_levels are main branch.

        Returns:
            During training: [aux_predictions, main_predictions]
            During inference: Tuple of ([decoded_aux, decoded_main], [raw_aux, raw_main])
        """
        aux_preds = []
        main_preds = []

        for i in range(self.num_levels):
            aux_preds.append(
                torch.cat((self.aux_box_convs[i](x[i]), self.aux_cls_convs[i](x[i])), 1)
            )
            main_preds.append(
                torch.cat(
                    (
                        self.main_box_convs[i](x[self.num_levels + i]),
                        self.main_cls_convs[i](x[self.num_levels + i]),
                    ),
                    1,
                )
            )

        if self.training:
            return [aux_preds, main_preds]

        shape = aux_preds[0].shape
        if self._shape != shape:
            self._anchors, self._strides = (
                t.transpose(0, 1) for t in make_anchors(aux_preds, self.stride, 0.5)
            )
            self._shape = shape

        batch = shape[0]
        assert self._anchors is not None and self._strides is not None

        flat_aux = torch.cat([p.view(batch, self.num_outputs, -1) for p in aux_preds], 2)
        box_aux, cls_aux = flat_aux.split((self.reg_max * 4, self.num_classes), 1)
        dbox_aux = (
            dist2bbox(self.dfl(box_aux), self._anchors.unsqueeze(0), xywh=True, dim=1)
            * self._strides
        )

        flat_main = torch.cat([p.view(batch, self.num_outputs, -1) for p in main_preds], 2)
        box_main, cls_main = flat_main.split((self.reg_max * 4, self.num_classes), 1)
        dbox_main = (
            dist2bbox(self.dfl2(box_main), self._anchors.unsqueeze(0), xywh=True, dim=1)
            * self._strides
        )

        y = [
            torch.cat((dbox_aux, cls_aux.sigmoid()), 1),
            torch.cat((dbox_main, cls_main.sigmoid()), 1),
        ]
        return y, [aux_preds, main_preds]

    def init_bias(self) -> None:
        """Initialize biases for better convergence."""
        for i, s in enumerate(self.stride):
            bias_val = math.log(5 / self.num_classes / (640 / s.item()) ** 2)

            aux_box_seq = self.aux_box_convs[i]
            aux_cls_seq = self.aux_cls_convs[i]
            assert isinstance(aux_box_seq, nn.Sequential)
            assert isinstance(aux_cls_seq, nn.Sequential)
            box_aux = aux_box_seq[-1]
            cls_aux = aux_cls_seq[-1]
            assert isinstance(box_aux, nn.Conv2d) and box_aux.bias is not None
            assert isinstance(cls_aux, nn.Conv2d) and cls_aux.bias is not None
            box_aux.bias.data[:] = 1.0
            cls_aux.bias.data[: self.num_classes] = bias_val

            main_box_seq = self.main_box_convs[i]
            main_cls_seq = self.main_cls_convs[i]
            assert isinstance(main_box_seq, nn.Sequential)
            assert isinstance(main_cls_seq, nn.Sequential)
            box_main = main_box_seq[-1]
            cls_main = main_cls_seq[-1]
            assert isinstance(box_main, nn.Conv2d) and box_main.bias is not None
            assert isinstance(cls_main, nn.Conv2d) and cls_main.bias is not None
            box_main.bias.data[:] = 1.0
            cls_main.bias.data[: self.num_classes] = bias_val
