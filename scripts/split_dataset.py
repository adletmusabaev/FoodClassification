import os
import random
import shutil

dataset_dir = r"C:\Users\22206\Desktop\FoodClassification\dataset"

train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

split_ratio = 0.2  

for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)

    if not os.path.isdir(category_path) or category in ["train", "test"]:
        continue

    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    
    images = [f for f in os.listdir(category_path)
              if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    split_point = int(len(images) * (1 - split_ratio))
    train_images = images[:split_point]
    test_images = images[split_point:]
    
    for img in train_images:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(train_dir, category, img))
    for img in test_images:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(test_dir, category, img))

print("Dataset splited")
