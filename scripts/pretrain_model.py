import hydra
import mlflow
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

# 获取当前文件的父目录的父目录 (即/workspace)
base_dir = Path(__file__).resolve().parent.parent

# 添加basics目录到Python路径
sys.path.append(str(base_dir))

from basics.transformer import BasicsTransformerLM
from basics.trainer_model import train, _to_device_and_compile, log_params_from_omegaconf_dict
from basics.tokenizer import get_custom_tokenizer


@hydra.main(config_path="configs/", config_name="pretrain", version_base=None)
def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.exp_name)
    mlflow.start_run()

    model_config, training_config = cfg.model, cfg.training
    model = BasicsTransformerLM(**model_config)

    model, device = _to_device_and_compile(model)

    train(model, device, training_config)

    mlflow.end_run()

if __name__ == '__main__':
    main()