import re
import argparse
from typing import List, Dict, Any

try:
    from openpyxl import Workbook
except ImportError as e:
    raise SystemExit("Missing dependency 'openpyxl'. Please install it with: pip install openpyxl")


def parse_run_payload_text(input_path: str) -> List[Dict[str, Any]]:
    """
    Parse the run_payload.txt produced by run_payload.py into structured rows.

    Expected block format per sample:
      --sample X--
      Input: ...
      Output: ...
      Score: 0.123
      Detected Injection: True/False
      (blank line)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")

    rows: List[Dict[str, Any]] = []
    current_sample = None
    current_input = None
    current_output = None
    current_score = None
    current_detect = None

    def maybe_commit():
        nonlocal current_sample, current_input, current_output, current_score, current_detect
        if current_sample is not None and current_score is not None and current_detect is not None:
            rows.append({
                "Sample": current_sample,
                "Input": current_input or "",
                "Output": (current_output or "").replace("\n", " ").strip(),
                "Score": current_score,
                "Detected Injection": bool(current_detect),
            })
        current_sample = None
        current_input = None
        current_output = None
        current_score = None
        current_detect = None

    for raw in lines:
        line = raw.strip()
        if not line:
            # blank line -> boundary; commit if a block is complete
            continue

        if line.startswith("--sample "):
            # Commit previous block before starting a new one
            maybe_commit()
            m = re.match(r"^--sample\s+(\d+)--$", line)
            if m:
                current_sample = int(m.group(1))
            else:
                # malformed header; reset
                current_sample = None
            continue

        if line.startswith("Input:"):
            current_input = line[len("Input:"):].strip()
            continue

        if line.startswith("Output:"):
            current_output = line[len("Output:"):].strip()
            continue

        if line.startswith("Score:"):
            m = re.match(r"^Score:\s*([\d\.eE+-]+)$", line)
            if m:
                try:
                    current_score = float(m.group(1))
                except ValueError:
                    current_score = None
            continue

        if line.startswith("Detected Injection:"):
            m = re.match(r"^Detected Injection:\s*(True|False)$", line, re.IGNORECASE)
            if m:
                current_detect = m.group(1).lower() == "true"
            continue

    # Commit last block if complete
    if current_sample is not None:
        maybe_commit()

    # Sort rows by sample number to ensure order
    rows.sort(key=lambda r: r["Sample"])
    return rows


def write_excel(rows: List[Dict[str, Any]], output_path: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "qwen2_15_payload"

    headers = ["Sample", "Input", "Output", "Score", "Detected Injection"]
    ws.append(headers)

    for row in rows:
        ws.append([
            row.get("Sample", ""),
            row.get("Input", ""),
            row.get("Output", ""),
            row.get("Score", ""),
            row.get("Detected Injection", ""),
        ])

    # Set basic column widths for readability
    ws.column_dimensions['A'].width = 10  # Sample
    ws.column_dimensions['B'].width = 60  # Input
    ws.column_dimensions['C'].width = 60  # Output
    ws.column_dimensions['D'].width = 12  # Score
    ws.column_dimensions['E'].width = 20  # Detected Injection

    wb.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Convert qwen2_15_payload.txt to Excel (qwen2_15_payload.xlsx)")
    parser.add_argument("--input", type=str, default="qwen2_15_payload.txt", help="Path to qwen2_15_payload.txt")
    parser.add_argument("--output", type=str, default="qwen2_15_payload.xlsx", help="Output Excel file path (.xlsx)")
    args = parser.parse_args()

    rows = parse_run_payload_text(args.input)
    if not rows:
        print("No rows parsed. Please ensure the input file follows the expected format.")
        return

    write_excel(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()


