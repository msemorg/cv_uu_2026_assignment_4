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

def save_misclassification_images(model, loader, threshold, max_images=100):
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
                if count >= max_images: return

                # Extract GT and Predictions
                gt_boxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 0] > 0.5:
                            gt_boxes.append({
                                'box': targets[b, i, j, 1:5], 
                                'label': torch.argmax(targets[b, i, j, 5:]).item()
                            })

                pred_boxes = []
                for i in range(7):
                    for j in range(7):
                        conf = outputs[b, i, j, 0].item()
                        if conf >= threshold:
                            pred_boxes.append({
                                'box': outputs[b, i, j, 1:5], 
                                'label': torch.argmax(outputs[b, i, j, 5:]).item(),
                                'conf': conf
                            })
                pred_boxes = apply_nms(pred_boxes)

                # Determine if this is a "Misdetection"
                # (Either count mismatch, or label mismatch with high IoU)
                is_error = len(gt_boxes) != len(pred_boxes)
                if not is_error and len(gt_boxes) > 0:
                    for p in pred_boxes:
                        best_iou = max([calculate_iou(p['box'], g['box']) for g in gt_boxes])
                        if best_iou > 0.5 and p['label'] != gt_boxes[0]['label']:
                            is_error = True
                            break

                if is_error:
                    # Plotting
                    img = images[b].permute(1, 2, 0).numpy()
                    # Basic un-normalization if you used transforms.Normalize
                    img = (img * 0.225) + 0.45 
                    img = np.clip(img, 0, 1)

                    fig, ax = plt.subplots(1)
                    ax.imshow(img)
                    
                    for g in gt_boxes:
                        box = g['box'] * 112
                        rect = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], linewidth=2, edgecolor='g', facecolor='none')
                        ax.add_patch(rect)
                    
                    for p in pred_boxes:
                        box = p['box'] * 112
                        rect = patches.Rectangle((box[0]-box[2]/2, box[1]-box[3]/2), box[2], box[3], linewidth=2, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)
                        ax.text(box[0]-box[2]/2, box[1]-box[3]/2, f"{p['label']} {p['conf']:.2f}", color='white', fontsize=8, backgroundcolor='red')

                    plt.title(f"Misdetection {count}")
                    plt.axis('off')
                    plt.savefig(f"{save_path}/error_{count}.png")
                    plt.close()
                    count += 1

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
    if not bboxes: return []
    # Sort by confidence
    bboxes = sorted(bboxes, key=lambda x: x['conf'], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box['label'] != chosen_box['label'] 
            or calculate_iou(chosen_box['box'], box['box']) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms

def get_metrics(model, loader, threshold, iou_threshold=0.5, use_nms=True):
    model.eval()
    all_tp, all_fp, all_fn = 0, 0, 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images).cpu()

            for b in range(outputs.shape[0]):
                gt_boxes = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 0] > 0.5:
                            gt_label = torch.argmax(targets[b, i, j, 5:]).item()
                            gt_boxes.append({'box': targets[b, i, j, 1:5], 'label': gt_label, 'matched': False})

                pred_boxes = []
                for i in range(7):
                    for j in range(7):
                        conf = outputs[b, i, j, 0].item()
                        #TODO check multiple thresholds
                        if conf >= threshold:
                            label = torch.argmax(outputs[b, i, j, 5:]).item()
                            pred_boxes.append({'box': outputs[b, i, j, 1:5], 'label': label, 'conf': conf})

                # CHOICE 8: Apply NMS to clean up redundant boxes
                if use_nms:
                    pred_boxes = apply_nms(pred_boxes)

                for p in pred_boxes:
                    best_iou, best_gt_idx = 0, -1
                    for idx, g in enumerate(gt_boxes):
                        iou = calculate_iou(p['box'], g['box'])
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, idx

                    if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_boxes[best_gt_idx]['matched']:
                        all_tp += 1
                        gt_boxes[best_gt_idx]['matched'] = True
                        y_true.append(gt_boxes[best_gt_idx]['label'])
                        y_pred.append(p['label'])
                    else:
                        all_fp += 1

                for g in gt_boxes:
                    if not g['matched']: all_fn += 1

    precision = all_tp / (all_tp + all_fp + 1e-6)
    recall = all_tp / (all_tp + all_fn + 1e-6)
    return precision, recall, y_true, y_pred

# --- EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NN_model().to(device)
model.load_state_dict(torch.load("best_yolo_model.pth"))

# 1. Sweep for PR Curve 
thresholds = np.linspace(0.01, 0.99, 3)
precisions, recalls = [], []

print("Running Threshold Sweep with NMS...")
for t in thresholds:
    print("\n testing threshold: ", t, "\n")
    p, r, _, _ = get_metrics(model, train_loader, t, use_nms=True)
    precisions.append(p)
    recalls.append(r)
f1_scores = [
2 * (p*r) / (p+r+1e-6)
for p, r in zip(precisions, recalls)
]

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print("Best threshold:", best_threshold)

# 2. Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recalls, precisions, color='blue', lw=2, label='P-R Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (mAP: {abs(np.trapezoid(precisions, recalls)):.4f})')
plt.legend()
plt.grid(True)
plt.savefig('results/Precision_Recall_Curve.png')
plt.close()

_, _, y_true, y_pred = get_metrics(
    model, val_loader, best_threshold, use_nms=True
)

# 3. Best Threshold Confusion Matrix
if len(y_true) == 0:
    print("⚠️ no detections at this threshold, try lower one")
else:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Threshold {best_threshold:.2f} + NMS)")
    plt.savefig('results/confusion_matrix.png')
    plt.close()

print(f"Saving misclassification examples using threshold {best_threshold:.2f}...")
save_misclassification_images(model, val_loader, best_threshold, max_images=20)
print("Done! Check the results/misclassifications folder.")