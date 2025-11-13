"""
Complete Model Comparison Script
Compares Baseline CNN, Transfer Learning (MobileNetV2), and SOTA (EfficientNetV2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ModelComparator:
    """Compare multiple trained models"""
    
    def __init__(self, test_dir, img_size=128):
        self.test_dir = test_dir
        self.img_size = img_size
        self.models = {}
        self.results = {}
        
    def load_models(self, model_paths):
        """Load trained models"""
        
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                print(f"Loading {name}...")
                self.models[name] = load_model(path)
                print(f"   {name} loaded")
            else:
                print(f"   {name} not found at {path}")
    
    def prepare_test_data(self, img_size=None):
        """Prepare test data generator"""
        
        if img_size is None:
            img_size = self.img_size
            
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_gen = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(img_size, img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        return test_gen
    
    def evaluate_model(self, model_name, model, img_size=None):
        """Evaluate single model"""
        
        print(f"\nEvaluating {model_name}...")
        
        # Prepare data
        test_gen = self.prepare_test_data(img_size)
        
        # Measure inference time
        start_time = time.time()
        predictions = model.predict(test_gen, verbose=0)
        end_time = time.time()
        
        # Calculate metrics
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes
        
        accuracy = (y_pred == y_true).mean()
        
        # Per-class accuracy
        class_accuracies = {}
        for i, class_name in enumerate(test_gen.class_indices.keys()):
            mask = y_true == i
            if mask.sum() > 0:
                class_acc = (y_pred[mask] == y_true[mask]).mean()
                class_accuracies[class_name] = float(class_acc)
        
        # Calculate average inference time
        total_samples = len(y_true)
        avg_inference_time = (end_time - start_time) / total_samples * 1000  # ms
        
        # Store results
        self.results[model_name] = {
            'accuracy': float(accuracy),
            'class_accuracies': class_accuracies,
            'predictions': y_pred,
            'true_labels': y_true,
            'inference_time_ms': float(avg_inference_time),
            'total_params': model.count_params()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Inference time: {avg_inference_time:.2f} ms/image")
        print(f"  Parameters: {model.count_params():,}")
        
        return self.results[model_name]
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        
        print("\n" + "="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            # Use appropriate image size
            if 'SOTA' in name or 'EfficientNet' in name:
                img_size = 160
            else:
                img_size = 128
                
            self.evaluate_model(name, model, img_size)
    
    def create_comparison_table(self):
        """Create comprehensive comparison table"""
        
        data = []
        for name, results in self.results.items():
            data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Accuracy %': f"{results['accuracy']*100:.2f}%",
                'Inference (ms)': f"{results['inference_time_ms']:.2f}",
                'Parameters': f"{results['total_params']:,}"
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON TABLE")
        print("="*60)
        print(df.to_string(index=False))
        
        return df
    
    def visualize_comparison(self, save_path='complete_comparison.png'):
        """Create comprehensive visualization"""
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        model_names = list(self.results.keys())
        
        # 1. Overall Accuracy Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3'][:len(model_names)]
        bars = ax1.bar(model_names, accuracies, color=colors, width=0.6)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1.0])
        ax1.grid(True, axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.4f}\n{acc*100:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Inference Time Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        inference_times = [self.results[m]['inference_time_ms'] for m in model_names]
        bars = ax2.bar(model_names, inference_times, color=colors, width=0.6)
        ax2.set_ylabel('Time (ms)', fontsize=12)
        ax2.set_title('Inference Time per Image', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, time in zip(bars, inference_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}ms',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Per-Class Accuracy Heatmap
        ax3 = fig.add_subplot(gs[1, :])
        
        # Prepare data for heatmap
        class_names = list(self.results[model_names[0]]['class_accuracies'].keys())
        heatmap_data = []
        for model_name in model_names:
            row = [self.results[model_name]['class_accuracies'][c] for c in class_names]
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
                   xticklabels=class_names, yticklabels=model_names,
                   cbar_kws={'label': 'Accuracy'}, ax=ax3)
        ax3.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Food Class', fontsize=12)
        ax3.set_ylabel('Model', fontsize=12)
        
        # 4. Model Parameters Comparison
        ax4 = fig.add_subplot(gs[2, 0])
        params = [self.results[m]['total_params'] / 1e6 for m in model_names]  # in millions
        bars = ax4.bar(model_names, params, color=colors, width=0.6)
        ax4.set_ylabel('Parameters (Millions)', fontsize=12)
        ax4.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
        ax4.grid(True, axis='y', alpha=0.3)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, p in zip(bars, params):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{p:.2f}M',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 5. Summary Statistics
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Find best model
        best_acc_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        fastest_model = min(self.results.items(), key=lambda x: x[1]['inference_time_ms'])
        smallest_model = min(self.results.items(), key=lambda x: x[1]['total_params'])
        
        # Calculate improvements
        baseline_acc = self.results[model_names[0]]['accuracy']
        sota_acc = self.results[model_names[-1]]['accuracy']
        improvement = ((sota_acc - baseline_acc) / baseline_acc) * 100
        
        summary = f"""
        COMPARISON SUMMARY
        
        Best Accuracy:
          {best_acc_model[0]}
          {best_acc_model[1]['accuracy']:.4f} ({best_acc_model[1]['accuracy']*100:.2f}%)
        
        Fastest:
          {fastest_model[0]}
          {fastest_model[1]['inference_time_ms']:.2f} ms/image
        
        Smallest:
          {smallest_model[0]}
          {smallest_model[1]['total_params']/1e6:.2f}M parameters
        
        SOTA Improvement:
          {improvement:+.2f}% over baseline
        
        Recommendation:
          {'Use SOTA model for best accuracy' if sota_acc > baseline_acc else 'Consider trade-offs'}
        """
        
        ax5.text(0.1, 0.5, summary, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                family='monospace')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Comparison visualization saved to {save_path}")
        plt.show()
    
    def save_results(self, output_dir='comparison_results'):
        """Save comparison results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save JSON
        results_to_save = {}
        for name, results in self.results.items():
            results_copy = results.copy()
            # Convert numpy arrays to lists
            results_copy['predictions'] = results_copy['predictions'].tolist()
            results_copy['true_labels'] = results_copy['true_labels'].tolist()
            results_to_save[name] = results_copy
        
        with open(os.path.join(output_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save CSV
        df = self.create_comparison_table()
        df.to_csv(os.path.join(output_dir, 'comparison_table.csv'), index=False)
        
        print(f"\n✅ Results saved to {output_dir}/")


def main():
    """Main comparison pipeline"""
    
    print("="*60)
    print("COMPLETE MODEL COMPARISON")
    print("="*60)
    
    # Configuration
    TEST_DIR = r"C:\Users\22206\Desktop\FoodClassification\dataset\test"
    
    # Model paths (update with your actual paths)
    model_paths = {
        'Baseline CNN': r"C:\Users\22206\Desktop\FoodClassification\models\baseline_cnn.h5",
        'Transfer Learning (MobileNetV2)': r"C:\Users\22206\Desktop\FoodClassification\models\transfer_mobilenetv2.h5",
        'SOTA (MobileNetV2-Optimized)': r"models_sota_fixed\final_sota_model.h5"
    }
    
    # Initialize comparator
    comparator = ModelComparator(TEST_DIR)
    
    # Load models
    comparator.load_models(model_paths)
    
    if not comparator.models:
        print("\n⚠️ No models found! Please train models first.")
        return
    
    # Evaluate all models
    comparator.evaluate_all_models()
    
    # Create comparison table
    comparator.create_comparison_table()
    
    # Visualize results
    comparator.visualize_comparison(save_path='complete_comparison.png')
    
    # Save results
    comparator.save_results(output_dir='comparison_results')
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED!")
    print("="*60)
    print("\n📊 Key Findings:")
    
    # Print key insights
    model_names = list(comparator.results.keys())
    if len(model_names) >= 2:
        baseline_acc = comparator.results[model_names[0]]['accuracy']
        sota_acc = comparator.results[model_names[-1]]['accuracy']
        improvement = ((sota_acc - baseline_acc) / baseline_acc) * 100
        
        print(f"  • SOTA model improves accuracy by {improvement:+.2f}%")
        print(f"  • Baseline: {baseline_acc*100:.2f}%")
        print(f"  • SOTA: {sota_acc*100:.2f}%")


if __name__ == "__main__":
    main()