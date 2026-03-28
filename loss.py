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
        # predictions shape: [Batch, 7, 7, 7] -> (conf, x, y, w, h, c1, c2)
        # target shape:      [Batch, 7, 7, 7]
        
        # Mask for cells that actually have an object
        exists_box = target[..., 0].unsqueeze(-1) 

        # --- 1. COORDINATE LOSS ---
        # predictions are ALREADY sigmoided from the model. 
        # We just extract them.
        pred_xy = predictions[..., 1:3]
        target_xy = target[..., 1:3]
        
        # Original YOLOv1 uses sqrt(w) and sqrt(h)
        # Use epsilon (1e-6) inside sqrt for numerical stability
        pred_wh = torch.sqrt(predictions[..., 3:5] + 1e-6)
        target_wh = torch.sqrt(target[..., 3:5] + 1e-6)
        
        # Only calculate loss where an object exists
        # We multiply by sqrt(lambda_coord) so when squared by MSE, it becomes lambda_coord
        coord_loss = self.mse(
            exists_box * torch.cat([pred_xy, pred_wh], dim=-1),
            exists_box * torch.cat([target_xy, target_wh], dim=-1)
        )

        # --- 2 & 3. CONFIDENCE LOSS (Object & No-Object) ---
        pred_conf = predictions[..., 0:1]
        target_conf = target[..., 0:1]

        # Loss for cells WITH an object
        object_loss = self.mse(exists_box * pred_conf, exists_box * target_conf)

        # Loss for cells WITHOUT an object (weighted by lambda_noobj)
        no_obj_loss = self.mse((1 - exists_box) * pred_conf, (1 - exists_box) * target_conf)

        # --- 4. CLASS LOSS ---
        # Using MSE to stay strictly true to YOLOv1 paper
        pred_class = predictions[..., 5:]
        target_class = target[..., 5:]
        class_loss = self.mse(exists_box * pred_class, exists_box * target_class)

        # Total Loss Calculation
        return (self.lambda_coord * coord_loss) + object_loss + (self.lambda_noobj * no_obj_loss) + class_loss