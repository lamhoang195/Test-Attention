import argparse
import random
import torch
import numpy as np
from utils import open_config, create_model
from detector.attn import AttentionDetector

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(args.seed)

    model_config_path = f"./configs/model_configs/{args.model_name}_config.json"
    model_config = open_config(config_path=model_config_path)
    model = create_model(config=model_config)
    model.print_model_info()

    detector = AttentionDetector(model)

    with open(args.input_file, "r", encoding="utf-8") as f:
        idx = 0
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # Bỏ số thứ tự kiểu "123. " ở đầu dòng nếu có
            import re
            text = re.sub(r'^\s*\d+\.\s*', '', line)

            if not text:
                continue

            # Reset seed for each inference to ensure consistency
            set_seed(args.seed + idx)

            if args.fast:
                result = detector.detect_fast(text)
            else:
                result = detector.detect(text)

            idx += 1
            output_one_line = result[1]['generated_text'].replace("\n", " ").strip()

            print(f"--sample {idx}--")
            print(f"Input: {text}")
            print(f"Output: {output_one_line}")
            print(f"Score: {result[1]['focus_score']}")
            print(f"Detected Injection: {result[0]}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detector on each line of a text file")
    parser.add_argument("--model_name", type=str, default="qwen2-attn")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true", help="Use fast inference mode")
    args = parser.parse_args()
    main(args)