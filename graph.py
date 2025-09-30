import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Tuple, Dict, Optional

def extract_final_scores_and_checks(filename: str = 'mistral_7b_deepset.txt') -> Tuple[List[float], List[Optional[float]], List[int], List[str]]:
    """
    Extract scores and optional checks from given file and return:
      - scores: List[float]
      - checks: List[Optional[float]] (parsed from 'Check:' if present; True/False -> 1/0)
      - sample_numbers: List[int]
      - labels: List[str] derived from 'Detected Injection' (Injection/Normal)
    
    Returns:
        Tuple[List[float], List[Optional[float]], List[int], List[str]]
    """
    scores = []
    checks: List[Optional[float]] = []
    sample_numbers = []
    labels = []
    
    try:
        # Thử nhiều encoding khác nhau
        for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
            try:
                with open(filename, 'r', encoding=encoding) as file:
                    content = file.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Không thể đọc file với bất kỳ encoding nào")
        
        print(f"Đã đọc file với encoding: {encoding}")
        print(f"Độ dài nội dung: {len(content)} ký tự")
        
        # Parse định dạng từ run_payload.py/mistral_7b_payload.txt:
        # --sample X--
        # Input: ...
        # Output: ...
        # Score: 0.123
        # Detected Injection: True/False
        # Check: 0.456 (tuỳ chọn) hoặc True/False
        # (dòng trống)
        lines = content.split('\n')

        current_sample = None
        pending_score = None
        pending_detect = None
        pending_check: Optional[float] = None

        for line in lines:
            line = line.strip()

            # Header mẫu: --sample X--
            if line.startswith('--sample '):
                # Nếu đang có mẫu trước đó mà đủ dữ liệu, lưu lại
                if current_sample is not None and pending_score is not None and pending_detect is not None:
                    scores.append(pending_score)
                    checks.append(pending_check)
                    sample_numbers.append(current_sample)
                    labels.append('Injection' if pending_detect else 'Normal')

                # Bắt đầu mẫu mới
                sample_match = re.search(r'^--sample\s+(\d+)--$', line)
                if sample_match:
                    current_sample = int(sample_match.group(1))
                    pending_score = None
                    pending_detect = None
                    pending_check = None
                else:
                    current_sample = None
                    pending_score = None
                    pending_detect = None
                    pending_check = None
                continue

            # Score line
            if line.startswith('Score:'):
                try:
                    score_match = re.search(r'^Score:\s*([\d\.eE+-]+)', line)
                    if score_match:
                        pending_score = float(score_match.group(1))
                except ValueError:
                    pending_score = None
                continue

            # Detected Injection line
            if line.startswith('Detected Injection:'):
                det_match = re.search(r'^Detected Injection:\s*(True|False)', line, re.IGNORECASE)
                if det_match:
                    pending_detect = det_match.group(1).lower() == 'true'
                continue

            # Check line (optional): numeric or boolean)
            if line.startswith('Check:'):
                raw = line[len('Check:'):].strip()
                # Try parse float, else parse True/False to 1/0
                value: Optional[float] = None
                try:
                    value = float(re.search(r'^([\d\.eE+-]+)', raw).group(1)) if re.search(r'^([\d\.eE+-]+)', raw) else None
                except Exception:
                    value = None
                if value is None:
                    if raw.lower() == 'true':
                        value = 1.0
                    elif raw.lower() == 'false':
                        value = 0.0
                pending_check = value
                continue

        # Lưu mẫu cuối cùng nếu đủ dữ liệu
        if current_sample is not None and pending_score is not None and pending_detect is not None:
            scores.append(pending_score)
            checks.append(pending_check)
            sample_numbers.append(current_sample)
            labels.append('Injection' if pending_detect else 'Normal')
        
        print(f"Tìm thấy {len(scores)} mẫu dữ liệu")
        
        if not scores:
            # Debug: in ra vài dòng để kiểm tra format
            print("Một số dòng đầu tiên của file:")
            for i, line in enumerate(content.split('\n')[:20]):
                if line.strip():
                    print(f"{i+1}: {line}")
        
    except FileNotFoundError:
        print(f"Không tìm thấy file: {filename}")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
    
    return scores, checks, sample_numbers, labels

def create_visualizations(scores: List[float], checks: List[Optional[float]], sample_numbers: List[int], labels: List[str]):
    """
    Visualize scores and compare against Check values if available
    """
    
    # Set up the figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Score vs Check Analysis', fontsize=16, fontweight='bold')
    
    # 1. Line plot of scores over samples
    ax1.plot(sample_numbers, scores, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(sample_numbers, scores, c=scores, cmap='viridis', s=20, alpha=0.8)
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Final Score')
    ax1.set_title('Scores Across Samples')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(scores) * 1.1 if scores else 1)
    
    # 2. Histogram of score distribution
    ax2.hist(scores, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Final Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Scores')
    ax2.grid(True, alpha=0.3)
    if scores:
        ax2.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.4f}')
        ax2.axvline(np.median(scores), color='orange', linestyle='--', label=f'Median: {np.median(scores):.4f}')
        ax2.legend()
    
    # 3. Box plot by label
    unique_labels = list(set(labels))
    scores_by_label = {label: [scores[i] for i, l in enumerate(labels) if l == label] 
                      for label in unique_labels}
    
    if unique_labels:
        box_data = [scores_by_label[label] for label in unique_labels]
        box_plot = ax3.boxplot(box_data, tick_labels=unique_labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
        for patch, color in zip(box_plot['boxes'], colors[:len(box_plot['boxes'])]):
            patch.set_facecolor(color)
    
    ax3.set_ylabel('Final Score')
    ax3.set_title('Scores by Label')
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter: distribution of Normal vs Injection using Check as class label
    #   - Check True (1) => Injection, False (0) => Normal, None => Unknown
    has_checks = any(c is not None for c in checks)
    if has_checks:
        mask_inj = [(c is not None and c >= 0.5) for c in checks]
        mask_norm = [(c is not None and c < 0.5) for c in checks]
        mask_unknown = [c is None for c in checks]

        xs_inj = [sample_numbers[i] for i, m in enumerate(mask_inj) if m]
        ys_inj = [scores[i] for i, m in enumerate(mask_inj) if m]
        xs_norm = [sample_numbers[i] for i, m in enumerate(mask_norm) if m]
        ys_norm = [scores[i] for i, m in enumerate(mask_norm) if m]
        xs_unk = [sample_numbers[i] for i, m in enumerate(mask_unknown) if m]
        ys_unk = [scores[i] for i, m in enumerate(mask_unknown) if m]

        if xs_norm:
            ax4.scatter(xs_norm, ys_norm, c='blue', marker='^', label='Normal (Check=False)', alpha=0.7, s=30)
        if xs_inj:
            ax4.scatter(xs_inj, ys_inj, c='red', marker='o', label='Injection (Check=True)', alpha=0.7, s=30)
        if xs_unk:
            ax4.scatter(xs_unk, ys_unk, c='gray', marker='s', label='Unknown (Check missing)', alpha=0.7, s=30)
    else:
        # Fallback to unlabeled scatter if no Check present
        ax4.scatter(sample_numbers, scores, c='blue', marker='o', label='Score', alpha=0.7, s=30)

    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Score')
    ax4.set_title('Score distribution by Check (Normal vs Injection)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('mistral_deepset.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    if scores:
        print(f"\n=== Score Statistics ===")
        print(f"Total samples: {len(scores)}")
        print(f"Mean score: {np.mean(scores):.4f}")
        print(f"Median score: {np.median(scores):.4f}")
        print(f"Min score: {np.min(scores):.4f}")
        print(f"Max score: {np.max(scores):.4f}")
        print(f"Standard deviation: {np.std(scores):.4f}")
        
        print(f"\n=== Score Distribution by Label ===")
        for label in unique_labels:
            label_scores = scores_by_label[label]
            if label_scores:
                print(f"{label}: {len(label_scores)} samples, mean={np.mean(label_scores):.4f}, std={np.std(label_scores):.4f}")

    # If Check values exist, create an overlay plot comparing Score vs Check
    if any(c is not None for c in checks):
        # Align lengths
        xs = sample_numbers
        ys_score = scores
        ys_check = [c if c is not None else np.nan for c in checks]

        plt.figure(figsize=(12, 5))
        plt.plot(xs, ys_score, label='Score', color='blue', alpha=0.7)
        plt.scatter(xs, ys_score, color='blue', s=15, alpha=0.6)
        plt.plot(xs, ys_check, label='Check', color='orange', alpha=0.7)
        plt.scatter(xs, ys_check, color='orange', s=15, alpha=0.6)
        plt.xlabel('Sample Number')
        plt.ylabel('Value')
        plt.title('Score vs Check over Samples')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('mistral_deepset.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Hàm chính để chạy visualization"""
    try:
        # Extract dữ liệu
        print("Extracting scores and checks from mistral_7b_payload.txt...")
        scores, checks, sample_numbers, labels = extract_final_scores_and_checks('mistral_7b_deepset.txt')
        
        if not scores:
            print("Không tìm thấy dữ liệu trong file mistral_7b_payload.txt")
            return
        
        print(f"Found {len(scores)} final scores")
        print("Creating visualizations...")
        
        # Tạo visualization
        create_visualizations(scores, checks, sample_numbers, labels)
        
        print("Analysis complete! Charts saved as 'mistral_deepset.png' and 'mistral_7b_deepset.png'")
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
