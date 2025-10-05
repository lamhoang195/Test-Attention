import matplotlib.pyplot as plt
import re
import numpy as np

def extract_scores_from_file(filename):
    """Trích xuất scores từ file"""
    scores = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            # Tìm tất cả các dòng có "Score:"
            score_lines = re.findall(r'Score: ([\d.]+)', content)
            scores = [float(score) for score in score_lines]
    except FileNotFoundError:
        print(f"File {filename} không tồn tại")
        return []
    return scores

def create_scatter_plot():
    """Tạo biểu đồ scatter so sánh scores của hai file giống như ảnh mẫu"""
    # Đọc scores từ hai file
    scores_deepset = extract_scores_from_file('mistral_7b_deepset.txt')
    scores_tophead = extract_scores_from_file('mistral_7b_deepset_tophead.txt')
    
    if not scores_deepset or not scores_tophead:
        print("Cannot read data from one or both files")
        return
    
    # Tạo figure và axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Tạo scatter plot với style giống ảnh mẫu
    x_indices = range(len(scores_deepset))
    
    # Vẽ scatter plot với màu sắc và style giống ảnh
    scatter1 = ax.scatter(x_indices, scores_deepset, alpha=0.8, s=60, 
                         color='blue', label='All Heads', marker='^', edgecolors='darkblue', linewidth=0.5)
    scatter2 = ax.scatter(x_indices, scores_tophead, alpha=0.8, s=60, 
                         color='red', label='Top Heads', marker='o', edgecolors='darkred', linewidth=0.5)
    
    # Thêm đường y=x để so sánh
    min_val = min(min(scores_deepset), min(scores_tophead))
    max_val = max(max(scores_deepset), max(scores_tophead))
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='y=x')
    
    # Thiết lập labels và title
    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Score', fontsize=12, fontweight='bold')
    ax.set_title('So sánh Scores của 2 cách tìm Important Heads', fontsize=14, fontweight='bold', pad=20)
    
    # Thiết lập range cho axes
    ax.set_xlim(-2, len(scores_deepset) + 2)
    ax.set_ylim(0.05, 0.75)
    
    # Thêm legend ở góc trên phải
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Thêm grid với style giống ảnh
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Thiết lập ticks
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Tính toán và hiển thị thống kê
    mean_deepset = np.mean(scores_deepset)
    mean_tophead = np.mean(scores_tophead)
    std_deepset = np.std(scores_deepset)
    std_tophead = np.std(scores_tophead)
    
    # Thêm text box với thống kê ở góc trên trái
    stats_text = f'All Heads:\nMean: {mean_deepset:.4f}\nStd: {std_deepset:.4f}\n\nTop Heads:\nMean: {mean_tophead:.4f}\nStd: {std_tophead:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Điều chỉnh layout
    plt.tight_layout()
    
    # Lưu biểu đồ
    plt.savefig('score_comparison_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Chart saved as 'score_comparison_scatter.png'")
    
    # Hiển thị biểu đồ
    plt.show()
    
    # In thống kê ra console
    print(f"\n=== STATISTICS COMPARISON ===")
    print(f"All Heads - Number of samples: {len(scores_deepset)}")
    print(f"All Heads - Mean: {mean_deepset:.6f}")
    print(f"All Heads - Std: {std_deepset:.6f}")
    print(f"All Heads - Min: {min(scores_deepset):.6f}")
    print(f"All Heads - Max: {max(scores_deepset):.6f}")
    
    print(f"\nTop Heads - Number of samples: {len(scores_tophead)}")
    print(f"Top Heads - Mean: {mean_tophead:.6f}")
    print(f"Top Heads - Std: {std_tophead:.6f}")
    print(f"Top Heads - Min: {min(scores_tophead):.6f}")
    print(f"Top Heads - Max: {max(scores_tophead):.6f}")
    
    # Tính correlation
    if len(scores_deepset) == len(scores_tophead):
        correlation = np.corrcoef(scores_deepset, scores_tophead)[0, 1]
        print(f"\nCorrelation: {correlation:.6f}")
        
        # Tính số điểm Top Heads cao hơn All Heads
        top_higher = sum(1 for i in range(len(scores_deepset)) if scores_tophead[i] > scores_deepset[i])
        print(f"Samples where Top Heads > All Heads: {top_higher}/{len(scores_deepset)} ({top_higher/len(scores_deepset)*100:.1f}%)")

if __name__ == "__main__":
    create_scatter_plot()
