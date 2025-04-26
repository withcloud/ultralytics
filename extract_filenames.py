import os
import glob

def get_filenames_without_extension(directory):
    """Extract filenames without extensions from the given directory and its subdirectories."""
    filenames = []
    
    # Get all files recursively in directory
    for root, _, _ in os.walk(directory):
        # Get all files in current directory
        files = glob.glob(os.path.join(root, '*.*'))
        
        # Extract filename without extension for each file
        for file_path in files:
            # Skip .DS_Store files
            if os.path.basename(file_path) == '.DS_Store':
                continue
                
            # Get the filename without extension
            filename = os.path.splitext(os.path.basename(file_path))[0]
            filenames.append(filename)
    
    return filenames

def main():
    # Define directories
    preds_dir = 'datasets/yoga82/preds'
    labels_dir = 'datasets/yoga82/labels'
    
    # Get filenames without extensions
    preds_filenames = get_filenames_without_extension(preds_dir)
    labels_filenames = get_filenames_without_extension(labels_dir)
    
    # Save results to files
    with open('preds_filenames.txt', 'w') as f:
        for filename in preds_filenames:
            f.write(f"{filename}\n")
    
    with open('labels_filenames.txt', 'w') as f:
        for filename in labels_filenames:
            f.write(f"{filename}\n")
    
    print(f"Found {len(preds_filenames)} files in preds directory")
    print(f"Found {len(labels_filenames)} files in labels directory")
    print("Filenames saved to preds_filenames.txt and labels_filenames.txt")

if __name__ == "__main__":
    main() 