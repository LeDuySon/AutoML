import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from auto_ml import automl_builder

def run_automl(cfg):
    automl_types = cfg["automl"]["name"]
    automl = automl_builder(automl_types, cfg)
    datasets = cfg["datasets"]
    
    logger.info(f"Start running automl {automl_types}")
    recorder = automl.run(datasets, save_record=cfg["save_record_path"])
    logger.info(f"Finish!!!")

@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    logger.info(f"Start with configs: \n{OmegaConf.to_yaml(cfg)}")

    # start automl test
    run_automl(cfg)
    
if __name__ == "__main__":
    my_app()