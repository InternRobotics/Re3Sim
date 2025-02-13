import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
import os

try:
    from torch_utils import distributed as dist
except:
    from .torch_utils import distributed as dist
import numpy as np

try:
    from src.act_plus_plus.detr.main import build_ACT_model_and_optimizer
except:
    from detr.main import build_ACT_model_and_optimizer
import IPython

e = IPython.embed


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTPolicy(nn.Module):
    def __init__(self, args_override, distributed=True):
        super().__init__()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        self.backbone_name = args_override["backbone"]
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"]),
                find_unused_parameters=False,
            )
        else:
            self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.qpos_noise_std = args_override["qpos_noise_std"]
        dist.print0(f"KL Weight {self.kl_weight}")

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        env_state = None
        if self.backbone_name == "dino_v2" or self.backbone_name[:7] == "dinov2_":
            patch_h = 16
            patch_w = 22
            if actions is not None:  # training time
                # transform = v2.Compose([
                #     v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                #     v2.RandomPerspective(distortion_scale=0.5),
                #     v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
                #     v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
                #     v2.Normalize(
                #         mean=[0.485, 0.456, 0.406],
                #         std=[0.229, 0.224, 0.225])
                # ])
                transform = v2.Compose(
                    [
                        v2.ColorJitter(
                            brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                        ),
                        v2.RandomPerspective(distortion_scale=0.5),
                        v2.RandomAffine(
                            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                        ),
                        v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
                        v2.Resize((patch_h * 14, patch_w * 14)),
                        # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                )
                qpos += (self.qpos_noise_std**0.5) * torch.randn_like(qpos)
            else:  # inference time
                transform = v2.Compose(
                    [
                        v2.Resize((patch_h * 14, patch_w * 14)),
                        # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                        v2.Normalize(
                            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                        ),
                    ]
                )
            image = transform(image)
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            image = normalize(image)
        if actions is not None:  # training time
            if self.distributed:
                actions = actions[:, : self.model.module.num_queries]
                is_pad = is_pad[:, : self.model.module.num_queries]
            else:
                actions = actions[:, : self.model.num_queries]
                is_pad = is_pad[:, : self.model.num_queries]

            loss_dict = dict()
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(
                qpos, image, env_state, actions, is_pad, vq_sample
            )

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, _, (_, _), _, _ = self.model(
                qpos, image, env_state, vq_sample=vq_sample
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, : self.model.module.num_queries]
        is_pad = is_pad[:, : self.model.module.num_queries]

        _, _, binaries, _, _ = self.model.module.encode(qpos, actions, is_pad)

        return binaries

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


class ACTPolicyDinov2(nn.Module):
    def __init__(self, args_override, distributed=True):
        super().__init__()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"]),
                find_unused_parameters=False,
            )  # CVAE decoder
        else:
            self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.qpos_noise_std = args_override["qpos_noise_std"]
        dist.print0(f"KL Weight {self.kl_weight}")
        self.distributed = distributed

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        patch_h = 16
        patch_w = 22
        if actions is not None:  # training time
            # transform = v2.Compose([
            #     v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #     v2.RandomPerspective(distortion_scale=0.5),
            #     v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
            #     v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
            #     v2.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ])
            transform = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
                    v2.RandomPerspective(distortion_scale=0.5),
                    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
                    v2.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            qpos += (self.qpos_noise_std**0.5) * torch.randn_like(qpos)
        else:  # inference time
            transform = v2.Compose(
                [
                    v2.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        image = transform(image)
        if actions is not None:  # training time
            if self.distributed:
                actions = actions[:, : self.model.module.num_queries]
                is_pad = is_pad[:, : self.model.module.num_queries]
            else:
                actions = actions[:, : self.model.num_queries]
                is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar), _, _ = self.model(
                qpos, image, env_state, actions, is_pad
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, is_pad_hat, (mu, logvar), _, _ = self.model(
                qpos, image, env_state
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ACTPolicyDinov2Condition(nn.Module):
    def __init__(self, args_override, distributed=True):
        super().__init__()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        args_override["is_condition"] = True
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=int(os.environ["LOCAL_RANK"]),
                find_unused_parameters=False,
            )  # CVAE decoder
        else:
            self.model = model
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.qpos_noise_std = args_override["qpos_noise_std"]
        dist.print0(f"KL Weight {self.kl_weight}")
        self.distributed = distributed

    def __call__(
        self, qpos, image, actions=None, is_pad=None, lang_emb=None, mask=None
    ):
        env_state = None
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        patch_h = 16
        patch_w = 22
        if actions is not None:  # training time
            # transform = v2.Compose([
            #     v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #     v2.RandomPerspective(distortion_scale=0.5),
            #     v2.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),
            #     v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
            #     v2.Normalize(
            #         mean=[0.485, 0.456, 0.406],
            #         std=[0.229, 0.224, 0.225])
            # ])
            transform = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                    ),
                    v2.RandomPerspective(distortion_scale=0.5),
                    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    v2.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 2.0)),
                    v2.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            qpos += (self.qpos_noise_std**0.5) * torch.randn_like(qpos)
        else:  # inference time
            transform = v2.Compose(
                [
                    v2.Resize((patch_h * 14, patch_w * 14)),
                    # v2.CenterCrop((patch_h * 14, patch_w * 14)),
                    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )

        image = transform(image)
        if actions is not None:  # training time
            if self.distributed:
                actions = actions[:, : self.model.module.num_queries]
                is_pad = is_pad[:, : self.model.module.num_queries]
            else:
                actions = actions[:, : self.model.num_queries]
                is_pad = is_pad[:, : self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar), _, _ = self.model(
                qpos, image, env_state, actions, is_pad, None, lang_emb, mask
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict["l1"] = l1
            loss_dict["kl"] = total_kld[0]
            loss_dict["loss"] = loss_dict["l1"] + loss_dict["kl"] * self.kl_weight
            return loss_dict
        else:  # inference time
            a_hat, is_pad_hat, (mu, logvar), _, _ = self.model(
                qpos, image, env_state, lang_emb
            )  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
