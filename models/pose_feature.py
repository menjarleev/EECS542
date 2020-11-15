import torch
import torch.nn as nn
from torch.nn import LayerNorm

class PoseExtractor(nn.Module):
    def __init__(self, pose_dim=512, use_bbox=False, final_dim=1024, layer_norm_epsilon=1e-12):
        super(PoseExtractor, self).__init__()

        self.ln_f = LayerNorm(final_dim, eps=layer_norm_epsilon)

        self.pose_downsample = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(pose_dim, final_dim),
            nn.ReLU(inplace=True),
        )

        self.use_bbox = use_bbox
        if use_bbox:
            self.bbox_upsample = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(4, final_dim),
                nn.ReLU(inplace=True),
            )

    def forward(self,
                pose_feats: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.Tensor,
                ):
        box_inds = box_mask.nonzero()
        pose_feats_selected = pose_feats[box_inds[:, 0], box_inds[:, 1]]

        pose_aligned_feats = self.ln_f(self.pose_downsample(pose_feats_selected))

        if self.use_bbox:
            bboxes = boxes[box_inds[:, 0], box_inds[:, 1]]
            box_feats = self.ln_f(self.bbox_upsample(bboxes))
            pose_aligned_feats = pose_aligned_feats + box_feats

        return {'pose_reps': pose_aligned_feats}

