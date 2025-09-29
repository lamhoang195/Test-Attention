## Features
- **Identify Important Heads**: Determine which attention heads are critical for detecting prompt injection attacks.
- **Run Experiments**: Execute experiments on datasets to evaluate the model's effectiveness.
- **Test Queries**: Test individual queries against your chosen model.

---

## Getting Started

### Prerequisites
1. Ensure you have Python installed.
2. Install the necessary dependencies listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Find Important Heads
    To identify important attention heads, run:
    ```bash
    python select_head.py --model_name {model} --num_data {num} --dataset llm
    ```
2. To specify important heads, go to `configs/model_configs/{model_name}.json` and change the `["params"]["important_heads"]`
3. To run Experiments on [DeepSet Prompt Injection Dataset](https://huggingface.co/datasets/deepset/prompt-injections?row=19).
    ```bash
    python run_dataset.py --model_name {model}
    ```
4. Test Individual Queries
    ```bash
    python run.py --model_name {model} --test_query "{query you want to test}"
    ```
