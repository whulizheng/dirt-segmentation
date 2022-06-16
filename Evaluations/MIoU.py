from turtle import forward
import numpy as np
import torch.nn as nn


import numpy as np
import torch


class MIoU(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        ious = []
        iousSum = 0
        y_pred = torch.from_numpy(y_pred)
        y_pred = y_pred.view(-1)
        y_true = np.array(y_true)
        y_true = torch.from_numpy(y_true)
        y_true = y_true.view(-1)

        # Ignore IoU for background class ("0")
        # This goes from 1:n_classes-1 -> class "0" is ignored
        for cls in range(1, self.num_classes):
            pred_inds = y_pred == cls
            target_inds = y_pred == cls
            intersection = (pred_inds[target_inds]).long().sum(
            ).data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + \
                target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                # If there is no ground truth, do not include in evaluation
                ious.append(float('nan'))
            else:
                ious.append(float(intersection) / float(max(union, 1)))
            iousSum += float(intersection) / float(max(union, 1))
        return iousSum/self.num_classes
