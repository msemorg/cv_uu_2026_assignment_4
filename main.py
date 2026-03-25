import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import NN_model
from loss import YoloLoss
from data_loader import CatDogDataset, train_imgs, train_anns, val_imgs, val_anns, transform

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
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

print(f"Training started on {device}...")
for epoch in range(EPOCHS):
    model.train()
    sum_train_loss = 0
    
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        
        # Forward
        predictions = model(images)
        loss = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        sum_train_loss += loss.item()

    # Validation Phase
    model.eval()
    sum_val_loss = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            val_preds = model(images)
            v_loss = criterion(val_preds, targets)
            sum_val_loss += v_loss.item()

    avg_train = sum_train_loss / len(train_loader)
    avg_val = sum_val_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # Save best model
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), "best_yolo_model.pth")
        print("--> Saved Best Model")

print("Training Complete!")