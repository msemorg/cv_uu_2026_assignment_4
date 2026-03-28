import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self, S=7, C=2):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5.0

    def forward(self, predictions, target):
        # Mask for cells that actually have an object [Batch, 7, 7, 1]
        exists_box = target[..., 0].unsqueeze(-1) 

        # --- 1. COORDINATE LOSS (x, y, w, h) ---
        # Separate x,y from w,h to avoid modifying the original tensor
        pred_conf = torch.sigmoid(predictions[..., 0:1])   # NEW
        pred_xy   = torch.sigmoid(predictions[..., 1:3])   # FIX
        target_xy = target[..., 1:3]
        
        # Calculate sqrt(w,h) into NEW variables (don't use = on slices)
        pred_wh = torch.sqrt(torch.clamp(predictions[..., 3:5], min=0))
        target_wh = torch.sqrt(target[..., 3:5])
        
        # Combine them back into new tensors
        pred_box_final = torch.cat([pred_xy, pred_wh], dim=-1)
        target_box_final = torch.cat([target_xy, target_wh], dim=-1)
        
        coord_loss = self.mse(exists_box * pred_box_final, exists_box * target_box_final)

        # --- 2. OBJECT LOSS (Confidence) ---
        object_loss = self.mse(exists_box * pred_conf, exists_box * target[..., 0:1])

        no_obj_loss = self.mse((1 - exists_box) * pred_conf, (1 - exists_box) * target[..., 0:1])
        # --- 4. CLASS LOSS ---
        class_loss = nn.CrossEntropyLoss(reduction="sum")(
            predictions[..., 5:][exists_box.squeeze(-1) > 0],
            torch.argmax(target[..., 5:][exists_box.squeeze(-1) > 0], dim=-1)
        )
        # Total Loss
        return (self.lambda_coord * coord_loss) + object_loss + (self.lambda_noobj * no_obj_loss) + class_loss


# Initialize YOLO Loss
criterion = YoloLoss(S=7, C=2)

