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

# Download latest version
path = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
print("Path to dataset files:", path)
# Copy files to local project folder
source = path
destination = "./cat_dog_dataset"
if not os.path.exists(destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Successfully copied dataset to {destination}")
else:
    print(f"Dataset folder already exists in {destination}, skipping copy.")

# resizing done by changing this value to 112. Checked if desired result by changing it to 10 or 50 and seeing if the bounding boxes are still correct.
INPUT_IMG_SZ = 112
IMG_DIR = "./cat_dog_dataset/images"
ANNOTATION_DIR = "./cat_dog_dataset/annotations"


class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
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


# Function to visualize a batch
def visualize_batch(dataloader):
    # 1. Unpack only TWO items now
    images, targets = next(iter(dataloader))

    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))

    if batch_size == 1:
        axes = [axes]

    for b in range(batch_size):
        # Prepare image for matplotlib
        img = images[b].permute(1, 2, 0).numpy()
        axes[b].imshow(img)

        # 2. Iterate through the 7x7 grid
        for i in range(7):  # Row (y)
            for j in range(7):  # Column (x)
                # Check if an object exists in this cell (Confidence slot 0)
                if targets[b, i, j, 0] > 0.5:

                    # 3. Extract YOLO values [x_cell, y_cell, w_norm, h_norm]
                    x_cell, y_cell, w_norm, h_norm = targets[b, i, j, 1:5]

                    # 4. Convert back to absolute pixels (112 is INPUT_IMG_SZ)
                    # Center of the box in pixels:
                    x_center = ((j + x_cell) / 7) * 112
                    y_center = ((i + y_cell) / 7) * 112

                    # Width and height in pixels:
                    w_pix = w_norm * 112
                    h_pix = h_norm * 112

                    # Calculate xmin, ymin for the Rectangle patch
                    xmin = x_center - (w_pix / 2)
                    ymin = y_center - (h_pix / 2)

                    # 5. Extract Label (Indices 5 and 6 are Cat/Dog)
                    # Get index of the max value in the class slots
                    label = torch.argmax(targets[b, i, j, 5:]).item()
                    label_text = "Cat" if label == 0 else "Dog"

                    # Add the box
                    rect = patches.Rectangle(
                        (xmin, ymin),
                        w_pix,
                        h_pix,
                        linewidth=2,
                        edgecolor="r",
                        facecolor="none",
                    )
                    axes[b].add_patch(rect)
                    axes[b].text(
                        xmin,
                        ymin - 5,
                        f"{label_text}",
                        color="red",
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.5),
                    )

        axes[b].axis("off")

    plt.show()
    return images, targets


if __name__ == "__main__":
    # Define transformations
    transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])

    # Initialize dataset and dataloader
    dataset = CatDogDataset(img_dir=IMG_DIR, ann_dir=ANNOTATION_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Visualize a batch
    visualize_batch(dataloader)
    images, targets = next(iter(dataloader))
    print(f"Single image tensor shape [C, H, W]: {images[0].shape}")

    all_img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    all_ann_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))
    temp_labels = []

    for ann in all_ann_files:
        tree = ET.parse(ann)
        label_name = tree.getroot().find("object/name").text
        temp_labels.append(label_name)

    train_imgs, val_imgs, train_anns, val_anns = train_test_split(
        all_img_files,
        all_ann_files,
        test_size=0.20,
        stratify=temp_labels,
        random_state=42,
    )

    total = len(train_imgs) + len(val_imgs)
    train_pct = (len(train_imgs) / total) * 100
    val_pct = (len(val_imgs) / total) * 100

    print(f"Split complete:")
    print(f"  - Training:   {len(train_imgs)} images ({train_pct:.2f}%)")
    print(f"  - Validation: {len(val_imgs)} images ({val_pct:.2f}%)")
