""" NOTE: used for separating a dataset into train, validation and test sets, not used for final LibriTTS &
LibriTTS-R dataset, already separated into subsets 
"""
import os
import random
import shutil

random.seed(42)

source_dir = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/libritts_fullspec'
dest_dir = '/work/tc062/tc062/s2501147/autoencoder/libritts_data/train_big_libriTTS'
train_dir = os.path.join(dest_dir, 'train')
val_dir = os.path.join(dest_dir, 'val')
test_dir = os.path.join(dest_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
random.shuffle(all_files)

total_files = len(all_files)
test_size = int(0.2 * total_files)
val_size = int(0.2 * (total_files - test_size))

test_files = all_files[:test_size]
val_files = all_files[test_size:test_size + val_size]
train_files = all_files[test_size + val_size:]

def copy_files(file_list, target_dir):
    for file_name in file_list:
        src_file = os.path.join(source_dir, file_name)
        dst_file = os.path.join(target_dir, file_name)
        shutil.copy(src_file, dst_file)

copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print(f"Copied {len(train_files)} files to {train_dir}")
print(f"Copied {len(val_files)} files to {val_dir}")
print(f"Copied {len(test_files)} files to {test_dir}")
