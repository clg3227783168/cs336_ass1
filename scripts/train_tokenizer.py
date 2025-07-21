import os
import pickle
import hydra

from omegaconf import DictConfig
import sys
from pathlib import Path

# 获取当前文件的父目录的父目录 (即/workspace)
base_dir = Path(__file__).resolve().parent.parent

# 添加basics目录到Python路径
sys.path.append(str(base_dir))

# 从tokenizer模块导入类
from basics.trainer_tokenizer import run_train_bpe


@hydra.main(config_path="configs", config_name="tokenizer", version_base=None)
def main(cfg: DictConfig):
    vocab, merges = run_train_bpe(
        input_path=cfg.input_path,
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens,
        num_chunks=cfg.num_chunks,
        num_processes=cfg.num_processes
    )

    os.makedirs(cfg.tokenizer_dir, exist_ok=True)
    with open(cfg.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(cfg.merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 统计最长token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))


if __name__ == "__main__":
    main()