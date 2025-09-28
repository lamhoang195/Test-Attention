import numpy as np
import time
from tqdm import tqdm

from .utils import process_attn, calc_attn_score


class AttentionDetector():
    def __init__(self, model, pos_examples=None, neg_examples=None, use_token="first", instruction="Say xxxxxx", threshold=0.5):
        self.name = "attention"
        self.attn_func = "normalize_sum"
        self.model = model
        self.important_heads = model.important_heads
        self.instruction = instruction
        self.use_token = use_token
        self.threshold = threshold
        if pos_examples and neg_examples:
            pos_scores, neg_scores = [], []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            for prompt in tqdm(neg_examples, desc="neg_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                neg_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2

        if pos_examples and not neg_examples:
            pos_scores = []
            for prompt in tqdm(pos_examples, desc="pos_examples"):
                _, _, attention_maps, _, input_range, generated_probs = self.model.inference(
                    self.instruction, prompt, max_output_tokens=1
                )
                pos_scores.append(self.attn2score(attention_maps, input_range))

            self.threshold = np.mean(pos_scores) - 4 * np.std(pos_scores)

    def attn2score(self, attention_maps, input_range):
        if self.use_token == "first":
            attention_maps = [attention_maps[0]]

        scores = []
        for attention_map in attention_maps:
            heatmap = process_attn(
                attention_map, input_range, self.attn_func)
            score = calc_attn_score(heatmap, self.important_heads)
            scores.append(score)

        return sum(scores) if len(scores) > 0 else 0

    def detect(self, data_prompt):
        # Measure generation time
        start_time = time.time()
        
        generated_text, _, attention_maps, _, input_range, _ = self.model.inference(
            self.instruction, data_prompt)
        
        end_time = time.time()
        generation_time = end_time - start_time

        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {
            "focus_score": focus_score,
            "generated_text": generated_text,
            "generation_time": generation_time
        }
    
    def detect_fast(self, data_prompt):
        """Fast detection using optimized inference"""
        # Measure generation time
        start_time = time.time()
        
        # Sử dụng inference_fast nếu có, nếu không fallback về inference
        if hasattr(self.model, 'inference_fast'):
            generated_text, _, attention_maps, _, input_range, _ = self.model.inference_fast(
                self.instruction, data_prompt)
        else:
            generated_text, _, attention_maps, _, input_range, _ = self.model.inference(
                self.instruction, data_prompt)
        
        end_time = time.time()
        generation_time = end_time - start_time

        focus_score = self.attn2score(attention_maps, input_range)
        return bool(focus_score <= self.threshold), {
            "focus_score": focus_score,
            "generated_text": generated_text,
            "generation_time": generation_time
        }
