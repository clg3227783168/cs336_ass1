import os
import torch
import hydra
from omegaconf import DictConfig
import sys
from pathlib import Path

# 获取当前文件的父目录的父目录 (即/workspace)
base_dir = Path(__file__).resolve().parent.parent

# 添加basics目录到Python路径
sys.path.append(str(base_dir))
from basics.eva_pretrain import evaluate
from basics.transformer import BasicsTransformerLM
from basics.tokenizer import get_custom_tokenizer
from basics.trainer_model import _to_device_and_compile

@hydra.main(config_path="configs", config_name="evaluate_cs336_lm", version_base=None)
def main(cfg: DictConfig):
    model_config, eval_config, tokenizer_config = cfg.model, cfg.eval, cfg.tokenizer
    tokenizer = get_custom_tokenizer(**tokenizer_config)
    print(tokenizer.vocab_size)
    model = BasicsTransformerLM(**model_config)

    model, device = _to_device_and_compile(model)
    tokenizer = get_custom_tokenizer(**tokenizer_config)

    with open(os.path.join(eval_config.save_path, f"ckpt_iter{eval_config.iteration}.pt"), 'rb') as f:
        checkpoint = torch.load(f, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])


    # 生成与输出
    result_text = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        prompt=eval_config.prompt,
        max_new_tokens=eval_config.max_new_tokens,
        temperature=eval_config.temperature,
        top_k=eval_config.top_k,
        eos_token_id=tokenizer.eos_token_id  # 视你的tokenizer设置而定
    )
    print("输入：", eval_config.prompt)
    print("生成结果：", result_text)

if __name__ == "__main__":
    main()