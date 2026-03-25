import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import your actual model class and data loader
from model import NN_model
from data_loader import val_loader

def calculate_iou(box1, box2):
    """
    Expects boxes in [x, y, w, h] normalized format (0-1).
    """
    # Convert to x1, y1, x2, y2
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

def get_metrics(model, loader, threshold, iou_threshold=0.5):
    model.eval()
    all_tp = 0
    all_fp = 0
    all_fn = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            # outputs shape: [Batch, 7, 7, 7]
            outputs = model(images).cpu()

            for b in range(outputs.shape[0]):
                # 1. Extract Ground Truths in this image
                gt_boxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 0] > 0.5:
                            gt_label = torch.argmax(targets[b, i, j, 5:]).item()
                            gt_boxes.append({'box': targets[b, i, j, 1:5], 'label': gt_label, 'matched': False})

                # 2. Extract Predictions above threshold
                pred_boxes = []
                for i in range(7):
                    for j in range(7):
                        conf = outputs[b, i, j, 0].item()
                        if conf >= threshold:
                            label = torch.argmax(outputs[b, i, j, 5:]).item()
                            pred_boxes.append({'box': outputs[b, i, j, 1:5], 'label': label, 'conf': conf})

                # 3. Match Preds to GTs
                for p in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    for idx, g in enumerate(gt_boxes):
                        iou = calculate_iou(p['box'], g['box'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx

                    if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_boxes[best_gt_idx]['matched']:
                        # True Positive
                        all_tp += 1
                        gt_boxes[best_gt_idx]['matched'] = True
                        y_true.append(gt_boxes[best_gt_idx]['label'])
                        y_pred.append(p['label'])
                    else:
                        # False Positive (Objectness is high but no match or wrong class)
                        all_fp += 1

                # 4. Count remaining unmatched GTs as False Negatives
                for g in gt_boxes:
                    if not g['matched']:
                        all_fn += 1

    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    return precision, recall, y_true, y_pred

# --- MAIN EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN_model().to(device)
model.load_state_dict(torch.load("best_yolo_model.pth"))

thresholds = np.linspace(0.05, 0.95, 10)
precisions = []
recalls = []

print("Starting Threshold Sweep...")
for t in thresholds:
    p, r, _, _ = get_metrics(model, val_loader, t)
    precisions.append(p)
    recalls.append(r)
    print(f"Threshold: {t:.2f} | Precision: {p:.3f} | Recall: {r:.3f}")

# Calculate mAP (Area under Precision-Recall curve)
# Note: For simple reporting, you can average the precisions or use np.trapezoid
mAP = np.trapezoid(precisions, recalls)
print(f"\nFinal Estimated mAP: {abs(mAP):.4f}")

# --- CONFUSION MATRIX for best threshold (e.g., 0.5) ---
_, _, y_true, y_pred = get_metrics(model, val_loader, 0.5)
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix at Threshold 0.5")
plt.show()