import argparse
import re
from typing import List, Dict, Optional


def parse_samples(lines: List[str]) -> List[Dict[str, Optional[str]]]:
    """Parse groups of lines into structured samples.

    Expected group format (order may be as printed by run_payload.py):
      --sample N--
      Input: ...
      Output: ...
      Score: ...
      Detected Injection: ...
      Check: ...            # optional
    Groups are typically separated by a blank line.
    """
    samples: List[Dict[str, Optional[str]]] = []
    cur: Dict[str, Optional[str]] = {}

    def flush_current():
        nonlocal cur
        if cur:
            # Ensure consistent keys
            for k in ["sample", "input", "output", "score", "detected", "check"]:
                cur.setdefault(k, None)
            samples.append(cur)
            cur = {}

    for raw in lines:
        line = raw.rstrip("\n")
        if not line.strip():
            flush_current()
            continue

        if line.startswith("--sample ") and line.endswith("--"):
            flush_current()
            cur["sample"] = line
            continue

        if line.startswith("Input:"):
            cur["input"] = line[len("Input:"):].strip()
            continue

        if line.startswith("Output:"):
            cur["output"] = line[len("Output:"):].strip()
            continue

        if line.startswith("Score:"):
            cur["score"] = line[len("Score:"):].strip()
            continue

        if line.startswith("Detected Injection:"):
            cur["detected"] = line[len("Detected Injection:"):].strip()
            continue

        if line.startswith("Check:"):
            cur["check"] = line[len("Check:"):].strip()
            continue

        # If none matched, keep as a free-form extra (rare). Append to output.
        # We'll store it concatenated to output to avoid data loss.
        extra = line.strip()
        if extra:
            cur["output"] = (cur.get("output") + " " + extra).strip() if cur.get("output") else extra

    flush_current()
    return [s for s in samples if any(s.values())]


def write_samples(samples: List[Dict[str, Optional[str]]]) -> List[str]:
    out: List[str] = []
    for idx, s in enumerate(samples, start=1):
        out.append(f"--sample {idx}--")
        if s.get("input") is not None:
            out.append(f"Input: {s['input']}")
        if s.get("output") is not None:
            out.append(f"Output: {s['output']}")
        if s.get("score") is not None:
            out.append(f"Score: {s['score']}")
        if s.get("detected") is not None:
            out.append(f"Detected Injection: {s['detected']}")
        if s.get("check") is not None:
            out.append(f"Check: {s['check']}")
        out.append("")
    return out


def unique_by_input(samples: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    """Keep only the first sample for each distinct Input (exact match after strip)."""
    seen = set()
    result: List[Dict[str, Optional[str]]] = []
    for s in samples:
        key = (s.get("input") or "").strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result


def report_input_counts(samples: List[Dict[str, Optional[str]]]) -> List[str]:
    counts: Dict[str, int] = {}
    for s in samples:
        key = (s.get("input") or "").strip()
        counts[key] = counts.get(key, 0) + 1
    # Build human-readable report
    lines: List[str] = []
    uniques = sum(1 for c in counts.values() if c >= 1)
    dups = {k: v for k, v in counts.items() if v > 1}
    lines.append(f"Total samples: {len(samples)}")
    lines.append(f"Unique inputs: {uniques}")
    lines.append(f"Duplicate groups: {len(dups)}")
    if dups:
        lines.append("")
        lines.append("Inputs with count > 1:")
        for k, v in sorted(dups.items(), key=lambda x: -x[1]):
            preview = (k[:120] + "…") if len(k) > 120 else k
            lines.append(f"{v}x | {preview}")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Remove samples that have duplicate Input; keep the first occurrence and renumber. Also supports a report mode.")
    parser.add_argument("--input", dest="input_path", default="mistral_7b_payload.txt", help="Input file path to parse")
    parser.add_argument("--output", dest="output_path", default="mistral_7b_payload_dedup.txt", help="Output file path")
    parser.add_argument("--report", dest="report", action="store_true", help="Only print a report of Input counts and exit")
    parser.add_argument("--counts_out", dest="counts_out", default=None, help="Optional path to write input count report")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    samples = parse_samples(lines)
    if args.report:
        rep_lines = report_input_counts(samples)
        print("\n".join(rep_lines))
        if args.counts_out:
            with open(args.counts_out, "w", encoding="utf-8") as rf:
                rf.write("\n".join(rep_lines).rstrip() + "\n")
        return

    deduped = unique_by_input(samples)
    out_lines = write_samples(deduped)

    with open(args.output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines).rstrip() + "\n")

    print(f"Parsed {len(samples)} samples. Wrote {len(deduped)} unique-by-input samples to {args.output_path}.")


if __name__ == "__main__":
    main()


