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

files_to_delete =["4257_6397_000024_000004.pt",
"4257_6397_000022_000001.pt",
"1638_84447_000056_000000.pt",
"1638_84447_000055_000001.pt",
"1422_146111_000008_000001.pt",
"1382_130492_000055_000000.pt",
"4257_6397_000025_000005.pt",
"335_125945_000018_000001.pt",
"4257_6397_000022_000002.pt",
"4257_6397_000026_000001.pt",
"1382_130492_000057_000000.pt",
"345_1119_000006_000000.pt",
"345_1119_000005_000000.pt",
"1382_130492_000046_000000.pt",
"1382_130492_000092_000000.pt",
"1789_142896_000029_000001.pt",
"3448_5416_000050_000002.pt",
"2494_156017_000007_000000.pt",
"2494_156017_000008_000002.pt",
"3274_163518_000106_000000.pt",
"2494_156017_000009_000001.pt",
"4945_29090_000021_000004.pt",
"1382_130492_000048_000000.pt",
"3448_5416_000057_000000.pt",
"1382_130492_000058_000000.pt",
"1382_130492_000093_000000.pt",
"1422_146111_000009_000005.pt",
"1382_130492_000095_000000.pt",
"3448_5416_000051_000000.pt",
"1422_146111_000010_000001.pt",
"1382_130492_000047_000000.pt",
"1638_84447_000056_000002.pt",
"1382_130492_000056_000000.pt",
"4257_6397_000025_000009.pt",
"1422_146111_000009_000001.pt",
"1382_130492_000094_000000.pt",
"2494_156017_000010_000000.pt",
"1789_142896_000029_000003.pt"]

delete_specific_files(directory_to_trim, files_to_delete)
print(f"{len(files_to_delete)} files were deleted")

# directory1 = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/test"
# directory2 = "/work/tc062/tc062/s2501147/autoencoder/libritts_data/enhancement_dataset/test_enhanced"

# missing_files = find_missing_files(directory1, directory2)

# if missing_files:
#     print("Files in directory1 but not in directory2:")
#     for file in missing_files:
#         print(file)
#     print("All files printed out")
# else:
#     print("No files are missing from directory2.")
