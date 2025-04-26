import os
import shutil

def get_filenames_without_extension(directory):
    """Get all filenames without extensions from a directory."""
    filenames = set()
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)) and not filename.startswith('.'):
                name_without_ext = os.path.splitext(filename)[0]
                filenames.add(name_without_ext)
    return filenames

def sync_directories(source_dir, target_dir):
    """Keep only files in target_dir that have matching filenames in source_dir."""
    print(f"Syncing {target_dir} based on {source_dir}")
    
    # Get filenames without extensions from both directories
    source_filenames = get_filenames_without_extension(source_dir)
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory {target_dir} does not exist")
        return
    
    # Create a backup directory
    backup_dir = f"{target_dir}_backup"
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    removed_count = 0
    
    for filename in os.listdir(target_dir):
        file_path = os.path.join(target_dir, filename)
        
        # Skip directories and hidden files
        if os.path.isfile(file_path) and not filename.startswith('.'):
            name_without_ext = os.path.splitext(filename)[0]
            
            # If filename not in source, move it to backup
            if name_without_ext not in source_filenames:
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(file_path, backup_path)
                removed_count += 1
    
    # Count remaining files
    remaining_files = len([f for f in os.listdir(target_dir) 
                          if os.path.isfile(os.path.join(target_dir, f)) and not f.startswith('.')])
    
    print(f"Removed {removed_count} files from {target_dir}")
    print(f"Remaining files: {remaining_files}")
    print(f"Source files: {len(source_filenames)}")
    print(f"Removed files backed up to {backup_dir}")

# Define directories
preds_train_dir = "datasets/yoga82/preds/train"
preds_val_dir = "datasets/yoga82/preds/val"
labels_train_dir = "datasets/yoga82/labels/train"
labels_val_dir = "datasets/yoga82/labels/val"
images_train_dir = "datasets/yoga82/images/train"
images_val_dir = "datasets/yoga82/images/val"

# Sync directories
print("\n=== Syncing labels directories ===")
sync_directories(preds_train_dir, labels_train_dir)
sync_directories(preds_val_dir, labels_val_dir)

print("\n=== Syncing images directories ===")
sync_directories(preds_train_dir, images_train_dir)
sync_directories(preds_val_dir, images_val_dir)

print("\nCompleted syncing all directories") 