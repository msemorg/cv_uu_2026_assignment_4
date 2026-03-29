import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NN_model
from loss import YoloLoss
from data_loader import CatDogDataset, train_imgs, train_anns, val_imgs, val_anns, transform
import matplotlib.pyplot as plt
import numpy as np

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100
BATCH_SIZE = 16
 
# 2. Prepare DataLoaders using the splits from data_loader.py
train_ds = CatDogDataset(train_imgs, train_anns, transform=transform)
val_ds = CatDogDataset(val_imgs, val_anns, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model, Loss, Optimizer
model = NN_model().to(device)
criterion = YoloLoss(S=7, C=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

# 4. Training Loop
best_val_loss = float('inf')

# Initialize history tracking
history = {
    "train_total": [], "val_total": [],
    "train_coord": [], "val_coord": [],
    "train_obj": [],   "val_obj": [],
    "train_noobj": [], "val_noobj": [],
    "train_class": [], "val_class": []
}

print(f"Training started on {device}...")
for epoch in range(EPOCHS):
    model.train()
    # Trackers for this epoch's averages
    tr_metrics = {k: 0 for k in ["total", "coord", "obj", "noobj", "class"]}
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        # 1. Forward Pass
        predictions = model(images)
        loss, components = criterion(predictions, targets)
        
        # 2. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 3. Record Training Metrics
        tr_metrics["total"] += loss.item()
        for k in ["coord", "obj", "noobj", "class"]:
            tr_metrics[k] += components[k]

    # Validation Phase
    model.eval()
    val_metrics = {k: 0 for k in ["total", "coord", "obj", "noobj", "class"]}
    
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            val_preds = model(images)
            v_loss, v_components = criterion(val_preds, targets)
            
            val_metrics["total"] += v_loss.item()
            for k in ["coord", "obj", "noobj", "class"]:
                val_metrics[k] += v_components[k]

    # 4. Append to History for plotting
    for k in ["total", "coord", "obj", "noobj", "class"]:
        history[f"train_{k}"].append(tr_metrics[k] / len(train_loader))
        history[f"val_{k}"].append(val_metrics[k] / len(val_loader))

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {history['train_total'][-1]:.4f} | Val Loss: {history['val_total'][-1]:.4f}")

    # Save best model
    if history['val_total'][-1] < best_val_loss:
        best_val_loss = history['val_total'][-1]
        torch.save(model.state_dict(), "best_yolo_model.pth")
        print("--> Saved Best Model")

def plot_yolo_losses(history):
    epochs = range(1, len(history["train_total"]) + 1)
    loss_names = ["total", "coord", "obj", "noobj", "class"]
    titles = ["Total Loss", "Coordinate Loss (λ_coord)", "Object Confidence", "No-Object Confidence", "Class Probability"]
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, title) in enumerate(zip(loss_names, titles)):
        plt.subplot(2, 3, i + 1)
        plt.plot(epochs, history[f"train_{name}"], 'b', label='Train')
        plt.plot(epochs, history[f"val_{name}"], 'r', label='Val')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig("yolo_loss_breakdown.png")
    plt.show()

# Call after training is complete
plot_yolo_losses(history)