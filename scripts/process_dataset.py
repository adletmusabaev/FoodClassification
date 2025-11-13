"""
Process Food Classification Dataset with Advanced Augmentation
Run this script to augment your training data
"""

import os
import sys

# Import the augmentor
try:
    from art_augmentation import ArtAugmentor
except ImportError:
    print(" Could not import art_augmentation.py")
    print("Make sure art_augmentation.py is in the same directory")
    sys.exit(1)


def main():
    """Process dataset with augmentation"""
    
    print("="*70)
    print("FOOD CLASSIFICATION DATASET AUGMENTATION")
    print("="*70)
    
    # Configuration
    BASE_DIR = r"C:\Users\22206\Desktop\FoodClassification"
    INPUT_DIR = os.path.join(BASE_DIR, "dataset", "train")
    OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "train_augmented")
    NUM_AUG_PER_IMAGE = 3  # Generate 3 augmented versions per image
    
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"\n Input directory not found: {INPUT_DIR}")
        print("\nPlease make sure your dataset is organized as:")
        print("  FoodClassification/")
        print("    └── dataset/")
        print("        └── train/")
        print("            ├── Bread/")
        print("            ├── Fried food/")
        print("            ├── Seafood/")
        print("            └── Vegetable-Fruit/")
        return
    
    # Display configuration
    print(f"\n Input Directory:  {INPUT_DIR}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print(f" Augmentations per image: {NUM_AUG_PER_IMAGE}")
    
    # Check categories
    categories = [d for d in os.listdir(INPUT_DIR) 
                 if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    if not categories:
        print(f"\n No categories found in {INPUT_DIR}")
        return
    
    print(f"\n📊 Found categories: {categories}")
    
    # Count images per category
    total_images = 0
    for category in categories:
        cat_path = os.path.join(INPUT_DIR, category)
        images = [f for f in os.listdir(cat_path) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        image_count = len(images)
        total_images += image_count
        print(f"   • {category}: {image_count} images")
    
    print(f"\n Total input images: {total_images}")
    print(f" Expected output images: {total_images * (NUM_AUG_PER_IMAGE + 1)}")
    print(f"   (Original + {NUM_AUG_PER_IMAGE} augmented versions per image)")
    
    # Ask for confirmation
    print("\n" + "="*70)
    response = input("🚀 Start augmentation process? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n Augmentation cancelled.")
        return
    
    print("\n" + "="*70)
    print("STARTING AUGMENTATION PROCESS")
    print("="*70)
    
    # Initialize augmentor
    try:
        augmentor = ArtAugmentor(p=0.7)
        print(" Augmentor initialized")
    except Exception as e:
        print(f" Error initializing augmentor: {e}")
        return
    
    # Process dataset
    try:
        augmentor.process_dataset(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            num_aug_per_image=NUM_AUG_PER_IMAGE
        )
        
        print("\n" + "="*70)
        print(" AUGMENTATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        # Count output images
        print("\n Final Statistics:")
        total_output = 0
        for category in categories:
            cat_path = os.path.join(OUTPUT_DIR, category)
            if os.path.exists(cat_path):
                images = [f for f in os.listdir(cat_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                image_count = len(images)
                total_output += image_count
                print(f"   • {category}: {image_count} images")
        
        print(f"\n Total output images: {total_output}")
        print(f" Output saved to: {OUTPUT_DIR}")
        
        print("\n Next steps:")
        print("   1. Review augmented images in the output directory")
        print("   2. Update training script to use augmented dataset:")
        print(f"      train_dir = r'{OUTPUT_DIR}'")
        print("   3. Train your models with the augmented data")
        
    except Exception as e:
        print(f"\n Error during augmentation: {e}")
        import traceback
        traceback.print_exc()


def visualize_samples():
    """Visualize augmentation on sample images"""
    
    print("\n" + "="*70)
    print("VISUALIZE AUGMENTATION SAMPLES")
    print("="*70)
    
    BASE_DIR = r"C:\Users\22206\Desktop\FoodClassification"
    INPUT_DIR = os.path.join(BASE_DIR, "dataset", "train")
    
    if not os.path.exists(INPUT_DIR):
        print(f" Directory not found: {INPUT_DIR}")
        return
    
    # Get one image from each category
    categories = [d for d in os.listdir(INPUT_DIR) 
                 if os.path.isdir(os.path.join(INPUT_DIR, d))]
    
    augmentor = ArtAugmentor(p=0.7)
    
    for category in categories:
        cat_path = os.path.join(INPUT_DIR, category)
        images = [f for f in os.listdir(cat_path) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if images:
            sample_image = os.path.join(cat_path, images[0])
            print(f"\n📸 Creating visualization for: {category}")
            
            try:
                augmentor.visualize_augmentations(sample_image, category=category)
                print(f"    Saved: augmentation_examples_{category}.png")
            except Exception as e:
                print(f"    Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process food dataset with augmentation')
    parser.add_argument('--visualize', action='store_true', 
                       help='Only visualize augmentations, do not process dataset')
    parser.add_argument('--num-aug', type=int, default=3,
                       help='Number of augmentations per image (default: 3)')
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_samples()
    else:
        main()