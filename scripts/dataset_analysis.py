import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_dataset(base_dir):
    """Analyze dataset structure and class distribution"""
    
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    results = {
        'train': {},
        'test': {},
        'total': {}
    }
    
    # Analyze train set
    if os.path.exists(train_dir):
        for category in os.listdir(train_dir):
            category_path = os.path.join(train_dir, category)
            if os.path.isdir(category_path):
                images = [f for f in os.listdir(category_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                results['train'][category] = len(images)
    
    # Analyze test set
    if os.path.exists(test_dir):
        for category in os.listdir(test_dir):
            category_path = os.path.join(test_dir, category)
            if os.path.isdir(category_path):
                images = [f for f in os.listdir(category_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                results['test'][category] = len(images)
    
    # Calculate totals
    all_categories = set(results['train'].keys()) | set(results['test'].keys())
    for category in all_categories:
        results['total'][category] = (
            results['train'].get(category, 0) + 
            results['test'].get(category, 0)
        )
    
    return results

def visualize_distribution(results):
    """Visualize class distribution"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Train distribution
    if results['train']:
        categories = list(results['train'].keys())
        counts = list(results['train'].values())
        axes[0].bar(categories, counts, color='skyblue')
        axes[0].set_title('Train Set Distribution')
        axes[0].set_xlabel('Category')
        axes[0].set_ylabel('Number of Images')
        axes[0].tick_params(axis='x', rotation=45)
        for i, v in enumerate(counts):
            axes[0].text(i, v + 5, str(v), ha='center')
    
    # Test distribution
    if results['test']:
        categories = list(results['test'].keys())
        counts = list(results['test'].values())
        axes[1].bar(categories, counts, color='lightcoral')
        axes[1].set_title('Test Set Distribution')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Number of Images')
        axes[1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(counts):
            axes[1].text(i, v + 5, str(v), ha='center')
    
    # Total distribution
    if results['total']:
        categories = list(results['total'].keys())
        counts = list(results['total'].values())
        axes[2].bar(categories, counts, color='lightgreen')
        axes[2].set_title('Total Dataset Distribution')
        axes[2].set_xlabel('Category')
        axes[2].set_ylabel('Number of Images')
        axes[2].tick_params(axis='x', rotation=45)
        for i, v in enumerate(counts):
            axes[2].text(i, v + 5, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistics(results):
    """Print detailed statistics"""
    
    print("=" * 60)
    print("DATASET ANALYSIS REPORT")
    print("=" * 60)
    
    print("\n CLASS DISTRIBUTION:")
    print("-" * 60)
    print(f"{'Category':<20} {'Train':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    
    for category in sorted(results['total'].keys()):
        train_count = results['train'].get(category, 0)
        test_count = results['test'].get(category, 0)
        total_count = results['total'][category]
        
        print(f"{category:<20} {train_count:<10} {test_count:<10} {total_count:<10}")
        
        total_train += train_count
        total_test += test_count
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_train:<10} {total_test:<10} {total_train + total_test:<10}")
    print("-" * 60)
    
    # Class balance analysis
    print("\n CLASS BALANCE ANALYSIS:")
    print("-" * 60)
    
    if results['total']:
        counts = list(results['total'].values())
        max_count = max(counts)
        min_count = min(counts)
        avg_count = sum(counts) / len(counts)
        
        print(f"Maximum class size: {max_count}")
        print(f"Minimum class size: {min_count}")
        print(f"Average class size: {avg_count:.1f}")
        print(f"Imbalance ratio: {max_count/min_count:.2f}x")
        
        print("\nClass balance (% of largest class):")
        for category in sorted(results['total'].keys()):
            count = results['total'][category]
            percentage = (count / max_count) * 100
            bar = '█' * int(percentage / 5)
            print(f"  {category:<20} {bar:<20} {percentage:.1f}%")
    
    # Recommendations
    print("\n RECOMMENDATIONS:")
    print("-" * 60)
    
    if results['total']:
        counts = list(results['total'].values())
        max_count = max(counts)
        min_count = min(counts)
        
        if max_count / min_count > 1.5:
            print("  Dataset is imbalanced!")
            print("   → Use TA-TiTok + MaskGen to generate more images for minority classes")
            print("   → Apply class weighting during training")
            
            for category, count in results['total'].items():
                if count < avg_count * 0.8:
                    needed = int(max_count - count)
                    print(f"   → Generate ~{needed} images for '{category}'")
        else:
            print(" Dataset is well-balanced")
        
        if total_train < 500:
            print("\n  Small dataset detected!")
            print("   → Use ArtAug for data augmentation")
            print("   → Consider generating synthetic images")
        
        if total_train + total_test < 1000:
            print("\n  Very small dataset!")
            print("   → Strongly recommend using transfer learning")
            print("   → Generate synthetic data with TA-TiTok")

if __name__ == "__main__":
    # Update this path to your dataset location
    dataset_dir = r"C:\Users\22206\Desktop\FoodClassification\dataset"
    
    print("Analyzing dataset...")
    results = analyze_dataset(dataset_dir)
    
    print_statistics(results)
    print("\n Generating visualization...")
    visualize_distribution(results)
    
    print("\n Analysis complete! Check 'dataset_distribution.png'")