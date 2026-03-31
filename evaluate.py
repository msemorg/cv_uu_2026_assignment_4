import torch
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from model import NN_model
from data_loader import val_loader, train_loader
import os

results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

import matplotlib.patches as patches


def save_misclassification_images(model, loader, threshold, max_images=10):
    model.eval()
    save_path = "results/misclassifications"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    count = 0
    with torch.no_grad():
        for images, targets in loader:
            images_cuda = images.to(device)
            outputs = model(images_cuda).cpu()

            for b in range(images.shape[0]):
                if count >= max_images:
                    return

                # 1. Extract GT and Predictions with Grid Info
                gt_boxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 0] > 0.5:
                            gt_boxes.append(
                                {
                                    "box": targets[b, i, j, 1:5],
                                    "label": torch.argmax(targets[b, i, j, 5:]).item(),
                                    "grid": (i, j),  # Added this\
                                    "matched": False,
                                }
                            )

                pred_boxes = []
                for i in range(7):
                    for j in range(7):
                        conf = outputs[b, i, j, 0].item()
                        if conf >= threshold:
                            pred_boxes.append(
                                {
                                    "box": outputs[b, i, j, 1:5],
                                    "label": torch.argmax(outputs[b, i, j, 5:]).item(),
                                    "conf": conf,
                                    "grid": (i, j),  # Added this
                                }
                            )
                pred_boxes = apply_nms(pred_boxes)

        # --- Replace Section 2 (Error Logic) with this ---
        error_type = ""
        if len(gt_boxes) > len(pred_boxes):
            error_type = "False Negative (Missed Object)"
            is_error = True
        elif len(pred_boxes) > len(gt_boxes):
            error_type = "False Positive (Ghost Detection)"
            is_error = True
        else:
            # Same number of boxes, check for class or localization errors
            for p in pred_boxes:
                ious = [calculate_iou(p["box"], g["box"]) for g in gt_boxes]
                best_iou = max(ious) if ious else 0
                best_gt_idx = np.argmax(ious) if ious else 0
                
                if best_iou < 0.3:
                    error_type = "Poor Localization"
                    is_error = True
                elif p["label"] != gt_boxes[best_gt_idx]["label"]:
                    error_type = f"Wrong Class (Pred:{p['label']} vs GT:{gt_boxes[best_gt_idx]['label']})"
                    is_error = True

        # --- Replace Section 3 (Plotting Title) ---
        if is_error:
            # ... (your existing image processing) ...
            
            # Add a descriptive title so you know WHY it's in this folder
            plt.title(f"Error {count}: {error_type}\nGreen=GT, Red=Pred", fontsize=10, color='red')
            
            # Optional: Add text to the side or bottom with details
            info_text = f"Pred Conf: {pred_boxes[0]['conf']:.2f}" if pred_boxes else "No Detection"
            plt.figtext(0.5, 0.01, info_text, wrap=True, horizontalalignment='center', fontsize=9)
            
            # Save with a filename that includes the error type for easy sorting
            clean_error_name = error_type.split('(')[0].strip().replace(" ", "_")
            plt.savefig(f"{save_path}/{clean_error_name}_{count}.png")


def calculate_iou(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return inter_area / (union_area + 1e-6)


def apply_nms(bboxes, iou_threshold=0.45):
    """
    bboxes: list of {'box': [x,y,w,h], 'label': l, 'conf': c}
    """
    if not bboxes:
        return []
    # Sort by confidence
    bboxes = sorted(bboxes, key=lambda x: x["conf"], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box["label"] != chosen_box["label"]
            or calculate_iou(chosen_box["box"], box["box"]) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def get_metrics(model, loader, threshold, iou_threshold=0.5, use_nms=True):
    model.eval()
    y_true, y_pred = [], []
    # 2 represents "Background / No Object"

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images).cpu()

            for b in range(outputs.shape[0]):
                gt_boxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 0] > 0.5:
                            gt_boxes.append(
                                {
                                    "box": targets[b, i, j, 1:5],
                                    "label": torch.argmax(targets[b, i, j, 5:]).item(),
                                    "matched": False,
                                }
                            )

                pred_boxes = []
                for i in range(7):
                    for j in range(7):
                        conf = outputs[b, i, j, 0].item()
                        if conf >= threshold:
                            pred_boxes.append(
                                {
                                    "box": outputs[b, i, j, 1:5],
                                    "label": torch.argmax(outputs[b, i, j, 5:]).item(),
                                    "conf": conf,
                                }
                            )

                if use_nms:
                    pred_boxes = apply_nms(pred_boxes)

                # Track which preds found a GT
                for p in pred_boxes:
                    best_iou, best_gt_idx = 0, -1
                    for idx, g in enumerate(gt_boxes):
                        iou = calculate_iou(p["box"], g["box"])
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, idx

                    if best_iou >= iou_threshold and best_gt_idx != -1:
                        if not gt_boxes[best_gt_idx]["matched"]:
                            # True Positive or Misclassification
                            y_true.append(gt_boxes[best_gt_idx]["label"])
                            y_pred.append(p["label"])
                            gt_boxes[best_gt_idx]["matched"] = True
                        else:
                            # Matched an already claimed GT -> False Positive (Duplicate)
                            y_true.append(2)  # Truth is "Background"
                            y_pred.append(p["label"])
                    else:
                        # No GT match -> False Positive (Ghost box)
                        y_true.append(2)  # Truth is "Background"
                        y_pred.append(p["label"])

                # Record False Negatives (Missed GTs)
                for g in gt_boxes:
                    if not g["matched"]:
                        y_true.append(g["label"])
                        y_pred.append(2)  # Prediction is "Background"

    return y_true, y_pred


# --- EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN_model().to(device)
model.load_state_dict(torch.load("best_yolo_model.pth"))

# 1. Denser Sweep for PR Curve & Optimal Threshold
# We use more points (20+) to get a smooth curve and accurate best threshold
thresholds = np.linspace(0.01, 0.99, num=20)  # Adjust range and num as needed
precisions, recalls = [], []

print("Running Threshold Sweep for mAP calculation...")
for t in thresholds:
    # get_metrics now returns (y_true, y_pred)
    y_true_sweep, y_pred_sweep = get_metrics(model, train_loader, t, use_nms=True)
    
    # Calculate Precision and Recall for this specific threshold
    # 0=Cat, 1=Dog, 2=Background
    tp = sum(1 for gt, pd in zip(y_true_sweep, y_pred_sweep) if gt == pd and gt != 2)
    fp = sum(1 for gt, pd in zip(y_true_sweep, y_pred_sweep) if gt == 2 and pd != 2)
    fn = sum(1 for gt, pd in zip(y_true_sweep, y_pred_sweep) if gt != 2 and pd == 2)
    
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    
    precisions.append(p)
    recalls.append(r)
    print(f"Threshold {t:.2f} -> Precision: {p:.4f}, Recall: {r:.4f}")

# Calculate F1-score for each threshold to find the "Optimal" balance
f1_scores = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(precisions, recalls)]
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best Threshold based on F1-Score: {best_threshold:.4f} with F1: {f1_scores[best_idx]:.4f}")
max_f1 = f1_scores[best_idx]

# Calculate mAP using the trapezoidal rule (area under PR curve)
# We sort by recall to ensure the integral is calculated correctly
sorted_indices = np.argsort(recalls)
mAP = np.trapezoid(
    np.array(precisions)[sorted_indices], np.array(recalls)[sorted_indices]
)

print(f" Optimal Threshold: {best_threshold:.4f}")
print(f" Max F1-Score: {max_f1:.4f}")
print(f" Calculated mAP: {mAP:.4f}")

# 2. Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(
    recalls, precisions, color="darkorange", lw=2, label=f"P-R Curve (mAP = {mAP:.2f})"
)
plt.scatter(
    recalls[best_idx],
    precisions[best_idx],
    color="red",
    label=f"Best F1 @ {best_threshold:.2f}",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.savefig("results/Precision_Recall_Curve.png")
plt.close()

# 3. Validation Performance at Optimal Threshold
print(f" Generating Confusion Matrix on Validation Set...")
y_true, y_pred = get_metrics(model, val_loader, best_threshold)

if len(y_true) > 0:
    # labels=[0, 1, 2] corresponds to Cat, Dog, Background
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Create the display with the new background label
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Cat", "Dog", "Background"]
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, ax=ax)

    plt.title(f"Full Confusion Matrix (Threshold {best_threshold:.2f})")
    plt.savefig("results/confusion_matrix_full.png")
    plt.close()

# 4. Save Misclassifications
print(f" Saving misclassification examples...")
save_misclassification_images(model, val_loader, best_threshold, max_images=20)
print("Done! Check the 'results/' directory.")
