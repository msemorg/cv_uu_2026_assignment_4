import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NN_model
from loss import YoloLoss
from data_loader import (
    CatDogDataset,
    train_imgs,
    train_anns,
    val_imgs,
    val_anns,
    transform,
    INPUT_IMG_SZ,
)
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 8
TRAIN_NEW_MODEL = False  # Set to False to load and evaluate


train_ds = CatDogDataset(train_imgs, train_anns, transform=transform, use_augment=True)
val_ds = CatDogDataset(val_imgs, val_anns, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = NN_model().to(device)
criterion = YoloLoss(S=7, C=2).to(device)

from sklearn.metrics import precision_recall_curve
import numpy as np


def find_optimal_threshold(model, loader, device):
    model.eval()
    all_confs = []
    all_gt = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            preds = model(images)

            for b in range(preds.shape[0]):
                for i in range(7):
                    for j in range(7):
                        conf = preds[b, i, j, 0].item()
                        gt_obj = targets[b, i, j, 0].item()

                        all_confs.append(conf)
                        all_gt.append(gt_obj)

    precision, recall, thresholds = precision_recall_curve(all_gt, all_confs)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find the index of the highest F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold, f1_scores[best_idx]


if TRAIN_NEW_MODEL:
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    # Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",  # 'min' because we want to minimize validation loss
        factor=0.5,  # Multiply LR by 0.5 when triggered (2e-4 -> 1e-4)
        patience=5,  # Wait for 5 epochs of no improvement before dropping
    )
    best_val_loss = float("inf")
    history = {
        k: []
        for k in [
            "train_total",
            "val_total",
            "train_coord",
            "val_coord",
            "train_obj",
            "val_obj",
            "train_noobj",
            "val_noobj",
            "train_class",
            "val_class",
        ]
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        tr_metrics = {k: 0 for k in ["total", "coord", "obj", "noobj", "class"]}

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)  # Keep it on GPU
            loss, components = criterion(predictions, targets)
            loss = loss / images.size(0)  # Normalize by batch size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_metrics["total"] += loss.item()
            for k in ["coord", "obj", "noobj", "class"]:
                tr_metrics[k] += components[k]

        # Validation
        model.eval()
        val_metrics = {k: 0 for k in ["total", "coord", "obj", "noobj", "class"]}
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                val_preds = model(images)
                v_loss, v_components = criterion(val_preds, targets)
                val_metrics["total"] += v_loss.item()
                for k in ["coord", "obj", "noobj", "class"]:
                    val_metrics[k] += v_components[k]

        for k in ["total", "coord", "obj", "noobj", "class"]:
            history[f"train_{k}"].append(tr_metrics[k] / len(train_loader))
            history[f"val_{k}"].append(val_metrics[k] / len(val_loader))

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {history['train_total'][-1]:.4f} | Val Loss: {history['val_total'][-1]:.4f}"
        )

        if history["val_total"][-1] < best_val_loss:
            best_val_loss = history["val_total"][-1]
            torch.save(model.state_dict(), "best_yolo_model.pth")
            print("--> Saved Best Model")

    scheduler.step(history["val_total"][-1])

    model.load_state_dict(torch.load("best_yolo_model.pth"))
    best_thresh, best_f1, _ = find_optimal_threshold(model, val_loader, device)
    print(f"Training Complete. Optimal Threshold: {best_thresh:.4f}")

else:
    print("Loading existing model weights...")
    model.load_state_dict(torch.load("best_yolo_model.pth", map_location=device))

    best_thresh, best_f1 = find_optimal_threshold(model, val_loader, device)
    print(f"Using Optimal Threshold: {best_thresh:.4f} (F1: {best_f1:.4f})")


def visualize_inference(loader, model, num_images=4, threshold_value=0.5):
    model.eval()
    images, targets, _ = next(iter(loader))
    images = images.to(device)

    with torch.no_grad():
        predictions = model(images).cpu()

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for b in range(num_images):
        img = images[b].permute(1, 2, 0).cpu().numpy()
        axes[b].imshow(img)

        conf_grid = predictions[b, :, :, 0]  # Shape [7, 7]

        max_idx = torch.argmax(conf_grid)
        i, j = divmod(max_idx.item(), 7)  # Convert flat index back to (row, col)

        conf = predictions[b, i, j, 0].item()
        x_c, y_c, w, h = predictions[b, i, j, 1:5]

        px = ((j + x_c) / 7) * INPUT_IMG_SZ
        py = ((i + y_c) / 7) * INPUT_IMG_SZ
        pw, ph = w * INPUT_IMG_SZ, h * INPUT_IMG_SZ

        label_idx = torch.argmax(predictions[b, i, j, 5:]).item()
        label_text = f"{'Cat' if label_idx==0 else 'Dog'} {conf:.2f}"

        # 3. Plot the single best box
        rect = patches.Rectangle(
            (px - pw / 2, py - ph / 2),
            pw,
            ph,
            linewidth=3,
            edgecolor="lime",
            facecolor="none",
        )
        axes[b].add_patch(rect)
        axes[b].text(
            px - pw / 2,
            py - ph / 2 - 5,
            label_text,
            color="lime",
            weight="bold",
            bbox=dict(facecolor="black", alpha=0.5),
        )

        axes[b].axis("off")
    plt.show()


# Run the visual check
visualize_inference(val_loader, model, threshold_value=best_thresh)
