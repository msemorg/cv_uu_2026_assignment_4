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
        
        # Mask for cells that actually have an object
        exists_box = target[..., 0].unsqueeze(-1) 

        pred_xy = predictions[..., 1:3]
        target_xy = target[..., 1:3]
        
        pred_wh = predictions[..., 3:5]
        target_wh = target[..., 3:5]

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
        weighted_coord = self.lambda_coord * coord_loss
        weighted_noobj = self.lambda_noobj * no_obj_loss
        
        total_loss = weighted_coord + object_loss + weighted_noobj + class_loss
        
        # Return total loss AND a dictionary of components for plotting
        return total_loss, {
            "coord": weighted_coord.item(),
            "obj": object_loss.item(),
            "noobj": weighted_noobj.item(),
            "class": class_loss.item()
        }
    