"""
NOTE: This script was used to remove the file in the LibriTTS dataset not present in the LibriTTS-R
dataset to create a parallel dataset. 
"""
import os

def get_files_in_directory(directory):
    """Get a set of file names in the given directory."""
    return set(os.listdir(directory))

def find_missing_files(dir1, dir2):
    """Find files that are in dir1 but not in dir2."""
    files_in_dir1 = get_files_in_directory(dir1)
    files_in_dir2 = get_files_in_directory(dir2)
    
    missing_files = files_in_dir1 - files_in_dir2
    return missing_files

def delete_specific_files(directory, files_to_delete):
    """Delete specific files in the given directory."""
    for filename in files_to_delete:
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found or is a directory: {file_path}")
        
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

directory_to_trim = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/train"

directory1 = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/test"
directory2 = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/test_enhanced"

missing_files = find_missing_files(directory1, directory2)

if missing_files:
    print("Files in directory1 but not in directory2:")
    for file in missing_files:
        print(file)
    print("All files printed out")
    delete_specific_files(directory_to_trim, missing_files)
    print(f"{len(missing_files)} files were deleted")
else:
    print("No files are missing from directory2.")