import os

# Define the folder paths
folder1_path = "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/seje_rigtige_iofiles/"
folder2_path = "/Users/hjaltewallin/Code/DP-MESTERNE/Dynammic-Programming/iofiles/"

# Get a list of files in each folder
folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

print("agevec.txt" in folder2_files)

for filename in folder1_files:
    print("")
    if filename in folder2_files:
        
        # Open both files
        file1_path = os.path.join(folder1_path, filename)
        file2_path = os.path.join(folder2_path, filename)
        with open(file1_path, "r") as f1, open(file2_path, "r") as f2:
            # Compare lines
            line1 = f1.readline()
            line2 = f2.readline()
            line_number = 1
            while line1 and line2:
                try:
                    float1 = round(float(line1.strip()), 1)
                    float2 = round(float(line2.strip()), 1)
                except ValueError:
                    print(f"Error parsing float in file {filename} at line {line_number}:")
                    print(f"  Folder 1: {line1.strip()}")
                    print(f"  Folder 2: {line2.strip()}")
                    break

                if float1 != float2:
                    print(f"Difference in file {filename} at line {line_number}:")
                    print(f"  Folder 1: {float1}")
                    print(f"  Folder 2: {float2}")
                    break
                line1 = f1.readline()
                line2 = f2.readline()
                line_number += 1

# Inform about files not present in both folders
missing_files_folder1 = set(folder1_files) - set(folder2_files)
missing_files_folder2 = set(folder2_files) - set(folder1_files)

if missing_files_folder1:
    print(f"Files missing in folder 2: {', '.join(missing_files_folder1)}")
if missing_files_folder2:
    print(f"Files missing in folder 1: {', '.join(missing_files_folder2)}")
