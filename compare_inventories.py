import os

def compare_inventories_to_file(local_path, gdrive_path, output_file):
    # Load local files
    with open(local_path, 'r') as f:
        # Splitting by ';' to ignore size/timestamp metadata
        local_files = set(line.strip().split(';')[0] for line in f if line.strip())
        
    # Load Google Drive files
    with open(gdrive_path, 'r') as f:
        gdrive_files = set(line.strip().split(';')[0] for line in f if line.strip())

    # Find the individual files missing from GDrive
    missing_files = local_files - gdrive_files

    # Extract unique parent directories and count missing items
    missing_folders = {}
    for file_path in missing_files:
        folder = os.path.dirname(file_path)
        if not folder:
            folder = "[Root Directory]"
        missing_folders[folder] = missing_folders.get(folder, 0) + 1

    # Write results to the output file
    with open(output_file, 'w') as out:
        out.write(f"--- Folder-Level Comparison Report ---\n")
        out.write(f"Local Inventory: {local_path}\n")
        out.write(f"GDrive Inventory: {gdrive_path}\n")
        out.write(f"Total individual files missing: {len(missing_files)}\n")
        out.write(f"Unique subfolders containing missing items: {len(missing_folders)}\n")
        out.write(f"{'-' * 40}\n\n")

        if missing_folders:
            out.write("Subfolders that need syncing:\n")
            for folder in sorted(missing_folders.keys()):
                count = missing_folders[folder]
                out.write(f"  - {folder} ({count} items missing)\n")
        else:
            out.write("All local items are present in the GDrive inventory.\n")

    # Final summary to the Terminal
    print(f"Comparison complete.")
    print(f"Total missing files: {len(missing_files)}")
    print(f"Report saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    compare_inventories_to_file('local_inventory.txt', 'gdrive_inventory.txt', 'missing_folders_report.txt')