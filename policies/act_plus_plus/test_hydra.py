from omegaconf import DictConfig, OmegaConf
import hydra
from configclass import TrainingConfig
from hydra.core.config_store import ConfigStore
import time

cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: TrainingConfig):
    # time.sleep(60)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
