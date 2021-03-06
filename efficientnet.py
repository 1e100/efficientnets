""" A clean, single-file EfficientNet implementation as per
https://arxiv.org/pdf/1905.11946.pdf. """

from typing import List
import dataclasses
import math
import collections
import pkg_resources

import torch
from torch import nn

_BN_MOMENTUM = 0.01
_BN_EPSILON = 1e-3

_PYTORCH_VERSION = pkg_resources.parse_version(torch.__version__)
_HAS_HARDSWISH = _PYTORCH_VERSION >= pkg_resources.parse_version("1.6.0")


def sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return x.sigmoid_()
    else:
        return x.sigmoid()


class Sigmoid(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid(x, inplace=self.inplace)


def swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return x.mul_(x.sigmoid()) if inplace else x * x.sigmoid()


class Swish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x, inplace=self.inplace)


def hard_sigmoid(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return nn.functional.relu6(x.add_(3.0), inplace=True).div_(6.0)
    else:
        return nn.functional.relu6(x + 3.0, inplace=False) / 6.0


class HardSigmoid(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_sigmoid(x, inplace=self.inplace)


def hard_swish(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return (x + 3.0).clamp_(0, 6.0).div_(6.0).mul_(x)
    else:
        return nn.functional.relu6(x + 3.0, inplace=False) / 6.0 * x


class HardSwish(nn.Module):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return hard_swish(x, inplace=self.inplace)


def _get_function(name: str, inplace: bool) -> nn.Module:
    """ Use this to instantiate activations and gating functions by name. """
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif name == "swish":
        return Swish(inplace=inplace)
    elif name == "hard_swish":
        if _HAS_HARDSWISH:
            return nn.Hardsiwsh()
        else:
            return HardSwish(inplace=inplace)
    elif name == "sigmoid":
        return Sigmoid(inplace=inplace)
    elif name == "hard_sigmoid":
        if _HAS_HARDSWISH:
            return nn.Hardsigmoid()
        else:
            return HardSigmoid(inplace=inplace)


@dataclasses.dataclass
class NetworkParams:
    width_multiplier: float  # Channel multiplier.
    depth_multiplier: float  # Number of layers multiplier.
    resolution: int
    dropout_fraction: float
    activation_fn: str
    se_gating_fn: str  # Only specified when use_se is true.
    use_se: bool


# "Official" values from here:
# https://github.com/google/automl/blob/master/efficientdet/backbone/efficientnet_builder.py#L40
# For Lite configs:
# https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/lite/efficientnet_lite_builder.py#L35
_NETWORK_PARAMS = {
    "efficientnet-b0": NetworkParams(1.0, 1.0, 224, 0.2, "swish", "sigmoid", True),
    "efficientnet-b1": NetworkParams(1.0, 1.1, 240, 0.2, "swish", "sigmoid", True),
    "efficientnet-b2": NetworkParams(1.1, 1.2, 260, 0.3, "swish", "sigmoid", True),
    "efficientnet-b3": NetworkParams(1.2, 1.4, 300, 0.3, "swish", "sigmoid", True),
    "efficientnet-b4": NetworkParams(1.4, 1.8, 380, 0.4, "swish", "sigmoid", True),
    "efficientnet-b5": NetworkParams(1.6, 2.2, 456, 0.4, "swish", "sigmoid", True),
    "efficientnet-b6": NetworkParams(1.8, 2.6, 528, 0.5, "swish", "sigmoid", True),
    "efficientnet-b7": NetworkParams(2.0, 3.1, 600, 0.5, "swish", "sigmoid", True),
    "efficientnet-b8": NetworkParams(2.2, 3.6, 672, 0.5, "swish", "sigmoid", True),
    "efficientnet-l2": NetworkParams(4.3, 5.3, 800, 0.5, "swish", "sigmoid", True),
    "efficientnet-lite0": NetworkParams(1.0, 1.0, 224, 0.2, "relu6", "", False),
    "efficientnet-lite1": NetworkParams(1.0, 1.1, 240, 0.2, "relu6", "", False),
    "efficientnet-lite2": NetworkParams(1.1, 1.2, 260, 0.3, "relu6", "", False),
    "efficientnet-lite3": NetworkParams(1.2, 1.4, 280, 0.3, "relu6", "", False),
    "efficientnet-lite4": NetworkParams(1.4, 1.8, 300, 0.3, "relu6", "", False),
}


@dataclasses.dataclass
class BlockParams:
    repeats: int
    kernel_size: int
    stride: int
    expand_ratio: int
    in_ch: int
    out_ch: int
    se_ratio: float


# "Official" values from here:
# https://github.com/google/automl/blob/master/efficientdet/backbone/efficientnet_builder.py#L172
_BLOCK_PARAMS = [
    BlockParams(1, 3, 1, 1, 32, 16, 0.25),
    BlockParams(2, 3, 2, 6, 16, 24, 0.25),
    BlockParams(2, 5, 2, 6, 24, 40, 0.25),
    BlockParams(3, 3, 2, 6, 40, 80, 0.25),
    BlockParams(3, 5, 1, 6, 80, 112, 0.25),
    BlockParams(4, 5, 2, 6, 112, 192, 0.25),
    BlockParams(1, 3, 1, 6, 192, 320, 0.25),
]


def _round_to_8(val: int) -> int:
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    round_up_bias = 0.9
    round_up_to = 8
    new_val = max(round_up_to, int(val + round_up_to / 2) // round_up_to * round_up_to)
    return int(new_val if new_val >= round_up_bias * val else new_val + round_up_to)


class _SE(nn.Module):
    def __init__(
        self,
        channels: int,
        reduced_channels: int,
        activation_fn: str = "swish",
        gating_fn: str = "sigmoid",
    ):
        super().__init__()
        self.reduce = nn.Conv2d(channels, reduced_channels, 1, bias=True)
        self.activation = _get_function(activation_fn, inplace=True)
        self.expand = nn.Conv2d(reduced_channels, channels, 1, bias=True)
        self.gate = _get_function(gating_fn, inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = input.mean([2, 3], keepdim=True)
        y = self.activation(self.reduce(y))
        return self.gate(self.expand(y)) * input


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        expand_ratio: float,
        se_ratio: float = None,
        allow_residual: bool = True,
        activation_fn: str = "swish",
        se_gating_fn: str = "sigmoid",
    ):
        super().__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = _round_to_8(in_ch * expand_ratio)
        self.apply_residual = allow_residual and (in_ch == out_ch and stride == 1)

        layers = []
        # If expansion is not needed, skip it.
        if expand_ratio > 1:
            layers += [
                # Pointwise
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch, momentum=_BN_MOMENTUM, eps=_BN_EPSILON),
                _get_function(activation_fn, inplace=True),
            ]
        layers += [
            # Depthwise
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=mid_ch,
                bias=False,
            ),
            nn.BatchNorm2d(mid_ch, momentum=_BN_MOMENTUM, eps=_BN_EPSILON),
            _get_function(activation_fn, inplace=True),
        ]
        if se_ratio is not None:
            # Note that in some models (e.g. in MNV3) channel reduction is done
            # not by the number of "input" channels, but by the number of
            # "middle" channels.
            layers += [
                _SE(
                    mid_ch,
                    int(se_ratio * in_ch),
                    activation_fn=activation_fn,
                    gating_fn=se_gating_fn,
                )
            ]
        # Linear pointwise. Note that there's no activation.
        layers += [
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM, eps=_BN_EPSILON),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


class Classifier(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        num_classes: int,
        activation_fn: str,
        dropout_rate: float,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1, bias=False,),
            nn.BatchNorm2d(mid_ch, momentum=_BN_MOMENTUM, eps=_BN_EPSILON),
            _get_function(activation_fn, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate, inplace=True),
            nn.Linear(mid_ch, num_classes, bias=True),
        )


class EfficientNet(nn.Module):
    def __init__(
        self, in_ch: int = 3, num_classes: int = 1000, variant: str = "efficientnet-b0",
    ) -> None:
        """ Usage:
        >>> model = EfficientNet(1, num_classes=1000, variant="efficientnet-b3")
        >>> res = _NETWORK_PARAMS["efficientnet-b3"].resolution
        >>> x = torch.rand(1, 1, res, res)
        >>> y = model(x)
        >>> list(y.shape)
        [1, 1000]
        """
        super().__init__()
        assert variant in _NETWORK_PARAMS, f"{variant} is not supported"
        assert num_classes > 0, num_classes
        assert in_ch > 0
        network_params = _NETWORK_PARAMS[variant]
        width_multiplier = network_params.width_multiplier
        self.optimal_resolution = network_params.resolution

        # For "lite" variants, stem width is not scaled with the rest of the net.
        stem_width_multiplier = width_multiplier if "lite" not in variant else 1.0
        stem_out_ch = _round_to_8(_BLOCK_PARAMS[0].in_ch * stem_width_multiplier)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_ch, stem_out_ch, kernel_size=3, stride=2, padding=1, bias=False,
            ),
            nn.BatchNorm2d(stem_out_ch, momentum=_BN_MOMENTUM, eps=_BN_EPSILON,),
            _get_function(network_params.activation_fn, inplace=True),
        )

        depth_multiplier = network_params.depth_multiplier
        blocks = collections.OrderedDict()
        for block_idx, block_params in enumerate(_BLOCK_PARAMS):
            repeats = math.ceil(block_params.repeats * depth_multiplier)
            in_ch = (
                stem_out_ch
                if block_idx == 0
                else _round_to_8(block_params.in_ch * width_multiplier)
            )
            out_ch = _round_to_8(block_params.out_ch * width_multiplier)
            block_layers = collections.OrderedDict()
            for layer_idx in range(repeats):
                block_layers[f"ir{layer_idx}"] = _InvertedResidual(
                    in_ch if layer_idx == 0 else out_ch,
                    out_ch,
                    kernel_size=block_params.kernel_size,
                    stride=block_params.stride if layer_idx == 0 else 1,
                    expand_ratio=block_params.expand_ratio,
                    se_ratio=block_params.se_ratio if network_params.use_se else None,
                    activation_fn=network_params.activation_fn,
                    se_gating_fn=network_params.se_gating_fn,
                )
            blocks[f"block{block_idx}"] = nn.Sequential(block_layers)

        self.body = nn.Sequential(blocks)

        self.classifier = Classifier(
            _round_to_8(_BLOCK_PARAMS[-1].out_ch * width_multiplier),
            1280,
            num_classes,
            network_params.activation_fn,
            network_params.dropout_fraction,
        )

        self.init_weights_google()

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # Normally this is a customized kaiming_uniform_().
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Normally this is uniform_ bounded to a computed value.
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_in", nonlinearity="linear"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def init_weights_google(self):
        """ NOTE: ported from, with corrections
        https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    int(
                        module.kernel_size[0]
                        * module.kernel_size[1]
                        * module.out_channels
                    )
                    // module.groups
                )
                nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                fan_out = module.weight.size(0)
                range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(module.weight, -range, range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.classifier(x)
