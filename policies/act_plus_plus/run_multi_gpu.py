import hydra
from hydra.core.config_store import ConfigStore
from configclass import TrainingConfig
import tempfile
from detr.models.latent_model import Latent_Model_Transformer
from omegaconf import DictConfig, OmegaConf
import os
import subprocess
import torch
import sys
import random

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: TrainingConfig):
    print(OmegaConf.to_yaml(cfg))

    # save config to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as f:
        OmegaConf.save(cfg, f.name)
    print(f"Config saved to {f.name}")

    # 获取当前工作目录
    current_dir = os.getcwd()

    # 获取可用的GPU数量
    num_gpus = torch.cuda.device_count()
    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # 构建torchrun命令
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        f"--master_port={random.randint(10000, 65535)}",
        "imitate_episodes.py",
        f"--config-path={os.path.dirname(f.name)}",
        f"--config-name={os.path.basename(f.name).replace('.yaml', '')}",
        "hydra.job.chdir=True",
        f"hydra.run.dir={hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}",
    ]
    # 设置环境变量
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"

    # 直接使用subprocess.run，不进行输出重定向
    try:
        subprocess.run(
            cmd,
            cwd=current_dir,
            env=env,
            check=True,  # 如果进程返回非零状态码则抛出异常
        )
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败，返回码: {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
