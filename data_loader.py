import kagglehub
import os
import glob
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
import os
from sklearn.model_selection import train_test_split

path = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
source = path
destination = "./cat_dog_dataset"

if not os.path.exists(destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Successfully copied dataset to {destination}")
else:
    print(f"Dataset folder already exists in {destination}, skipping copy.")

# resizing done by changing this value to 112. Checked if desired result by changing it to 20 pixels and seeing if the bounding boxes are still correct.
INPUT_IMG_SZ = 112
IMG_DIR = "./cat_dog_dataset/images"
ANNOTATION_DIR = "./cat_dog_dataset/annotations"


class CatDogDataset(Dataset):
    def __init__(self, img_files, ann_files, transform=None):
        self.img_files = img_files
        self.ann_files = ann_files
        self.transform = transform
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        # 7x7 grid, 7 channels [conf, x, y, w, h, cat, dog]
        target = torch.zeros((7, 7, 7))
        for obj in objects:
            # Normalize absolute pixels to 0.0 - 1.0 relative to whole image
            xmin, ymin, xmax, ymax = obj["bbox"]
            xn = ((xmin + xmax) / 2) / width
            yn = ((ymin + ymax) / 2) / height
            wn = (xmax - xmin) / width
            hn = (ymax - ymin) / height

            # Determine which grid cell the center falls into
            i, j = int(7 * yn), int(7 * xn)

            # Avoid index out of bounds
            i, j = min(i, 6), min(j, 6)

            # x, y relative to the specific cell (0 to 1)
            x_cell = (7 * xn) - j
            y_cell = (7 * yn) - i

            # If cell is empty, fill it (YOLOv1 handles 1 object per cell)
            if target[i, j, 0] == 0:
                target[i, j, 0] = 1.0  # Confidence
                target[i, j, 1:5] = torch.tensor([x_cell, y_cell, wn, hn])
                target[i, j, 5 + obj["label"]] = 1.0  # One-hot class

        if self.transform:
            image = self.transform(image)

        return image, target


# Define transformations
transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])

# --- 3. Data Splitting Logic (Must come BEFORE dataset initialization) ---
all_img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
all_ann_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))
temp_labels = []

for ann in all_ann_files:
    tree = ET.parse(ann)
    # Get the first object label for stratification
    label_name = tree.getroot().find("object/name").text
    temp_labels.append(label_name)

train_imgs, val_imgs, train_anns, val_anns = train_test_split(
    all_img_files, all_ann_files, test_size=0.20, stratify=temp_labels, random_state=42
)

# --- 4. Initialize Datasets and Loaders ---
transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])

train_ds = CatDogDataset(train_imgs, train_anns, transform=transform)
val_ds = CatDogDataset(val_imgs, val_anns, transform=transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

total = len(train_imgs) + len(val_imgs)
train_pct = (len(train_imgs) / total) * 100
val_pct = (len(val_imgs) / total) * 100

print(f"Split complete:")
print(f"  - Training:   {len(train_imgs)} images ({train_pct:.2f}%)")
print(f"  - Validation: {len(val_imgs)} images ({val_pct:.2f}%)")


# --- 5. Visualization Function ---
def visualize_batch(loader):
    images, targets = next(iter(loader))
    batch_size = len(images)
    # Display up to 4 images
    num_to_show = min(batch_size, 4)
    fig, axes = plt.subplots(1, num_to_show, figsize=(15, 5))

    if num_to_show == 1:
        axes = [axes]

    for b in range(num_to_show):
        img = images[b].permute(1, 2, 0).numpy()
        axes[b].imshow(img)

        # Draw the 7x7 grid (Moved outside the object loop)
        grid_size = INPUT_IMG_SZ / 7
        for step in range(8):  # 0 to 7 to cover all lines
            axes[b].axhline(step * grid_size, color="white", linewidth=0.8, alpha=0.6)
            axes[b].axvline(step * grid_size, color="white", linewidth=0.8, alpha=0.6)

        for i in range(7):
            for j in range(7):
                if targets[b, i, j, 0] > 0.5:
                    x_c, y_c, w, h = targets[b, i, j, 1:5]

                    # Convert cell-relative coordinates to absolute pixel coordinates
                    px = ((j + x_c) / 7) * INPUT_IMG_SZ
                    py = ((i + y_c) / 7) * INPUT_IMG_SZ
                    pw, ph = w * INPUT_IMG_SZ, h * INPUT_IMG_SZ

                    # Create Rectangle
                    rect = patches.Rectangle(
                        (px - pw / 2, py - ph / 2),
                        pw,
                        ph,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    axes[b].add_patch(rect)

                    # Optional: Draw a small dot at the center to verify grid cell ownership
                    axes[b].plot(px, py, "ro", markersize=3)

        axes[b].axis("off")
    plt.tight_layout()
    plt.show()


print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
visualize_batch(train_loader)
