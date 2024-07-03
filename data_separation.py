import os
import random
import shutil

# Seed for reproducibility
random.seed(42)

# Directories
source_dir = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/saved_chopped_train_arrays'
dest_dir = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_sep'
train_dir = os.path.join(dest_dir, 'train')
val_dir = os.path.join(dest_dir, 'val')
test_dir = os.path.join(dest_dir, 'test')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files in the source directory
all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Shuffle the files
random.shuffle(all_files)

# Define the split sizes
total_files = len(all_files)
test_size = int(0.2 * total_files)
val_size = int(0.2 * (total_files - test_size))  # 0.2 of the remaining 80%

# Split the files
test_files = all_files[:test_size]
val_files = all_files[test_size:test_size + val_size]
train_files = all_files[test_size + val_size:]

# Function to move files
def move_files(file_list, target_dir):
    for file_name in file_list:
        src_file = os.path.join(source_dir, file_name)
        dst_file = os.path.join(target_dir, file_name)
        shutil.move(src_file, dst_file)

# Move files to respective directories
move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print(f"Moved {len(train_files)} files to {train_dir}")
print(f"Moved {len(val_files)} files to {val_dir}")
print(f"Moved {len(test_files)} files to {test_dir}")
