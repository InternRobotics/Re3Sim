# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import dino_build
from .detr_vae import dino_condition_build
from .detr_vae import build_cnnmlp as build_cnnmlp


def build_ACT_model(args):
    if "is_condition" in args and args.is_condition == True:
        return dino_condition_build(args)
    if args.backbone == "dino_v2" or args.backbone[:7] == "dinov2_":
        return dino_build(args)
    else:
        return build_vae(args)


def build_CNNMLP_model(args):
    return build_cnnmlp(args)
