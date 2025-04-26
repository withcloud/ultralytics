import os

def get_filenames_without_extension(directory):
    """Get all filenames without extensions from a directory."""
    filenames = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            # Skip directories and hidden files
            if os.path.isfile(os.path.join(directory, filename)) and not filename.startswith('.'):
                # Get filename without extension
                name_without_ext = os.path.splitext(filename)[0]
                filenames.append(name_without_ext)
    return filenames

# Define directories to process
train_dir = "datasets/yoga82/preds/train"
val_dir = "datasets/yoga82/preds/val"

# Get filenames without extensions
train_filenames = get_filenames_without_extension(train_dir)
val_filenames = get_filenames_without_extension(val_dir)

# Write results to files
with open("train_filenames.txt", "w") as f:
    for name in train_filenames:
        f.write(f"{name}\n")

with open("val_filenames.txt", "w") as f:
    for name in val_filenames:
        f.write(f"{name}\n")

print(f"Found {len(train_filenames)} files in {train_dir}")
print(f"Found {len(val_filenames)} files in {val_dir}")
print("Filenames saved to train_filenames.txt and val_filenames.txt") 