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
import random
from sklearn.model_selection import train_test_split

SHOW_VISUALIZATION = True  # Set to False to skip visualization
path = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
source = path
destination = "./cat_dog_dataset"

if not os.path.exists(destination):
    shutil.copytree(source, destination, dirs_exist_ok=True)
    print(f"Successfully copied dataset to {destination}")

INPUT_IMG_SZ = 112
IMG_DIR = "./cat_dog_dataset/images"
ANNOTATION_DIR = "./cat_dog_dataset/annotations"


class CatDogDataset(Dataset):
    def __init__(self, img_files, ann_files, transform=None, use_augment=True):
        self.img_files = img_files
        self.ann_files = ann_files
        self.transform = transform
        self.use_augment = use_augment
        self.label_map = {"cat": 0, "dog": 1}

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            label = self.label_map.get(name, -1)
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})
        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]
        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        applied_augs = []
        if self.use_augment:
            # 1. Flip
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                for obj in objects:
                    xmin, ymin, xmax, ymax = obj["bbox"]
                    obj["bbox"][0] = width - xmax
                    obj["bbox"][2] = width - xmin
                applied_augs.append("Flip")

            # 2. Scale
            if random.random() < 0.5:
                scale = 1.0
                while 0.8 < scale < 1.2:  # Only reroll if the change is too small
                    scale = random.uniform(0.5, 1.5)
                new_w, new_h = int(width * scale), int(height * scale)
                image = image.resize((new_w, new_h))
                for obj in objects:
                    obj["bbox"] = [c * scale for c in obj["bbox"]]
                width, height = new_w, new_h
                applied_augs.append(f"Scale({scale:.1f}x)")

            # 3. Shift
            if random.random() < 0.5:
                dx = random.randint(int(-0.15 * width), int(0.15 * width))
                dy = random.randint(int(-0.15 * height), int(0.15 * height))
                new_image = Image.new("RGB", (width, height), (0, 0, 0))
                new_image.paste(image, (dx, dy))
                image = new_image
                new_objects = []
                for obj in objects:
                    xmin, ymin, xmax, ymax = obj["bbox"]
                    xmin, xmax, ymin, ymax = xmin + dx, xmax + dx, ymin + dy, ymax + dy
                    xmin, xmax, ymin, ymax = (
                        max(0, xmin),
                        min(width, xmax),
                        max(0, ymin),
                        min(height, ymax),
                    )
                    if xmax > xmin + 1 and ymax > ymin + 1:
                        obj["bbox"] = [xmin, ymin, xmax, ymax]
                        new_objects.append(obj)
                objects = new_objects
                applied_augs.append("Shift")

            # 4. Color
            if random.random() < 0.5:
                image = T.ColorJitter(brightness=0.4, contrast=0.4)(image)
                applied_augs.append("Color")

        aug_label = ", ".join(applied_augs) if applied_augs else "None"

        # YOLO target building logic
        target = torch.zeros((7, 7, 7))
        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox"]
            xn, yn = ((xmin + xmax) / 2) / width, ((ymin + ymax) / 2) / height
            wn, hn = (xmax - xmin) / width, (ymax - ymin) / height
            i, j = int(7 * yn), int(7 * xn)
            i, j = max(0, min(i, 6)), max(0, min(j, 6))
            if target[i, j, 0] == 0:
                target[i, j, 0] = 1.0
                target[i, j, 1:5] = torch.tensor([(7 * xn) - j, (7 * yn) - i, wn, hn])
                target[i, j, 5 + obj["label"]] = 1.0

        if self.transform:
            image = self.transform(image)

        return image, target, aug_label


all_img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
all_ann_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))

train_imgs, val_imgs, train_anns, val_anns = train_test_split(
    all_img_files, all_ann_files, test_size=0.20, random_state=42
)

transform = T.Compose([T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)), T.ToTensor()])


def draw_boxes(ax, target):
    for i in range(7):
        for j in range(7):
            if target[i, j, 0] > 0.5:
                x_c, y_c, w, h = target[i, j, 1:5]
                px = ((j + x_c) / 7) * INPUT_IMG_SZ
                py = ((i + y_c) / 7) * INPUT_IMG_SZ
                pw, ph = w * INPUT_IMG_SZ, h * INPUT_IMG_SZ
                rect = patches.Rectangle(
                    (px - pw / 2, py - ph / 2),
                    pw,
                    ph,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax.add_patch(rect)


def visualize_comparison(img_files, ann_files, transform, num_samples=5):
    ds_orig = CatDogDataset(
        img_files, ann_files, transform=transform, use_augment=False
    )
    ds_aug = CatDogDataset(img_files, ann_files, transform=transform, use_augment=True)

    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    for i in range(num_samples):
        # Original
        img_o, target_o, _ = ds_orig[i]
        axes[0, i].imshow(img_o.permute(1, 2, 0).numpy())
        draw_boxes(axes[0, i], target_o)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Augmented
        img_a, target_a, label = ds_aug[i]
        axes[1, i].imshow(img_a.permute(1, 2, 0).numpy())
        draw_boxes(axes[1, i], target_a)
        axes[1, i].set_title(f"Aug: {label}", color="red", fontsize=9)
        axes[1, i].axis("off")
    plt.show()


if SHOW_VISUALIZATION:
    visualize_comparison(train_imgs, train_anns, transform=transform)
