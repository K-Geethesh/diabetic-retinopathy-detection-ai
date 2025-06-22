import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Source directory with all class folders
source_dir = 'dataset'
output_dirs = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]  # 70% train, 20% val, 10% test

# Skip non-class files
class_names = [d for d in os.listdir(source_dir)
               if os.path.isdir(os.path.join(source_dir, d))]

# Create output folders
for split in output_dirs:
    for class_name in class_names:
        os.makedirs(os.path.join(split, class_name), exist_ok=True)

# Split and copy images
for class_name in class_names:
    class_path = os.path.join(source_dir, class_name)
    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(images)
    total = len(images)
    train_end = int(total * split_ratio[0])
    val_end = int(total * (split_ratio[0] + split_ratio[1]))

    for i, img_name in enumerate(images):
        src = os.path.join(class_path, img_name)

        if i < train_end:
            dest = os.path.join('train', class_name, img_name)
        elif i < val_end:
            dest = os.path.join('val', class_name, img_name)
        else:
            dest = os.path.join('test', class_name, img_name)

        shutil.copy2(src, dest)

print("✅ Dataset successfully split into train, val, and test folders!")
