import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Tuple, Dict

def extract_final_scores(filename: str = 'run_payload.txt') -> Tuple[List[float], List[int], List[str]]:
    """
    Extract final scores from result.txt and return scores, sample numbers, and labels
    
    Returns:
        Tuple[List[float], List[int], List[str]]: (scores, sample_numbers, labels)
    """
    scores = []
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
        
        # Parse định dạng từ run_payload.py:
        # --sample X--
        # Input: ...
        # Output: ...
        # Score: 0.123
        # Detected Injection: True/False
        # (dòng trống)
        lines = content.split('\n')

        current_sample = None
        pending_score = None
        pending_detect = None

        for line in lines:
            line = line.strip()

            # Header mẫu: --sample X--
            if line.startswith('--sample '):
                # Nếu đang có mẫu trước đó mà đủ dữ liệu, lưu lại
                if current_sample is not None and pending_score is not None and pending_detect is not None:
                    scores.append(pending_score)
                    sample_numbers.append(current_sample)
                    labels.append('Injection' if pending_detect else 'Normal')

                # Bắt đầu mẫu mới
                sample_match = re.search(r'^--sample\s+(\d+)--$', line)
                if sample_match:
                    current_sample = int(sample_match.group(1))
                    pending_score = None
                    pending_detect = None
                else:
                    current_sample = None
                    pending_score = None
                    pending_detect = None
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

        # Lưu mẫu cuối cùng nếu đủ dữ liệu
        if current_sample is not None and pending_score is not None and pending_detect is not None:
            scores.append(pending_score)
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
    
    return scores, sample_numbers, labels

def create_visualizations(scores: List[float], sample_numbers: List[int], labels: List[str]):
    """
    Create multiple visualizations for the final scores (similar to the provided example)
    """
    
    # Set up the figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Final Score Analysis', fontsize=16, fontweight='bold')
    
    # 1. Line plot of scores over samples
    ax1.plot(sample_numbers, scores, 'b-', linewidth=1, alpha=0.7)
    ax1.scatter(sample_numbers, scores, c=scores, cmap='viridis', s=20, alpha=0.8)
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Final Score')
    ax1.set_title('Final Scores Across Samples')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(scores) * 1.1 if scores else 1)
    
    # 2. Histogram of score distribution
    ax2.hist(scores, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Final Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Final Scores')
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
    ax3.set_title('Final Scores by Label')
    ax3.grid(True, alpha=0.3)
    
    # 4. Scatter plot colored by label
    label_colors = {'Injection': 'red', 'Normal': 'blue', 'Unknown': 'gray'}
    label_markers = {'Injection': 'o', 'Normal': '^', 'Unknown': 's'}  # o=circle, ^=triangle, s=square
    
    for label in unique_labels:
        mask = [l == label for l in labels]
        sample_nums = [sample_numbers[i] for i, m in enumerate(mask) if m]
        score_vals = [scores[i] for i, m in enumerate(mask) if m]
        color = label_colors.get(label, 'gray')
        marker = label_markers.get(label, 'o')
        ax4.scatter(sample_nums, score_vals, c=color, marker=marker, label=label, alpha=0.7, s=30)
    
    ax4.set_xlabel('Sample Number')
    ax4.set_ylabel('Final Score')
    ax4.set_title('Final Scores by Sample and Label')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('final_scores_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    if scores:
        print(f"\n=== Final Score Statistics ===")
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

def main():
    """Hàm chính để chạy visualization"""
    try:
        # Extract dữ liệu
        print("Extracting final scores from run_payload.txt...")
        scores, sample_numbers, labels = extract_final_scores('run_payload.txt')
        
        if not scores:
            print("Không tìm thấy dữ liệu trong file run_payload.txt")
            return
        
        print(f"Found {len(scores)} final scores")
        print("Creating visualizations...")
        
        # Tạo visualization
        create_visualizations(scores, sample_numbers, labels)
        
        print("Analysis complete! Chart saved as 'final_scores_analysis.png'")
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()
