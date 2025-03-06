# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import os

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython

e = IPython.embed

try:
    from torch_utils import distributed as dist
except:
    from real2sim2real.act_plus_plus.torch_utils import distributed as dist


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
    ):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d,
        )  # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DINOv2BackBone(nn.Module):
    def __init__(self, backone_name="dinov2_vits14") -> None:
        super().__init__()
        try:
            self.body = torch.hub.load("facebookresearch/dinov2", backone_name)
        except:
            import traceback

            traceback.print_exc()
            dist.print0(f"Failed to load dinov2 from torch hub, loading from local")
            weights_path = os.path.expanduser(
                f"~/.cache/torch/hub/checkpoints/{backone_name}_pretrain.pth"
            )

            code_path = os.path.expanduser(
                "~/.cache/torch/hub/facebookresearch_dinov2_main"
            )

            self.body = torch.hub.load(
                code_path, backone_name, source="local", pretrained=False
            )

            state_dict = torch.load(weights_path)
            self.body.load_state_dict(state_dict)
        if backone_name == "dinov2_vits14":
            self.num_channels = 384
        elif backone_name == "dinov2_vitb14":
            self.num_channels = 768
        elif backone_name == "dinov2_vitl14":
            self.num_channels = 1024
        elif backone_name == "dinov2_vitg14":
            self.num_channels = 1408
        else:
            raise NotImplementedError(f"DINOv2 backbone {backone_name} not implemented")

    # @torch.no_grad()
    def forward(self, tensor):
        xs = self.body.forward_features(tensor)["x_norm_patchtokens"]
        od = OrderedDict()
        od["0"] = xs.reshape(xs.shape[0], 22, 16, xs.shape[-1]).permute(0, 3, 2, 1)
        return od


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone == "dino_v2" or args.backbone[:7] == "dinov2_":
        backone_name = "dinov2_vits14" if args.backbone == "dino_v2" else args.backbone
        backbone = DINOv2BackBone(backone_name)
        dist.print0("Using dino v2 backbone ")
        if args.freeze_backbone:
            dist.print0("Freezing dino v2 backbone")
            for p in backbone.parameters():
                p.requires_grad = False
            # backbone._requires_grad = False
        else:
            dist.print0("Finetuning dino v2 backbone")
            for p in backbone.parameters():
                p.requires_grad = True
            # backbone._requires_grad = True

        if hasattr(backbone.body, "mask_token"):
            backbone.body.mask_token.requires_grad = False
    else:
        assert args.backbone in ["resnet18", "resnet34"]
        backbone = Backbone(
            args.backbone, train_backbone, return_interm_layers, args.dilation
        )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
