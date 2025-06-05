import logging
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing , point_mesh_face_distance , point_mesh_edge_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes , Pointclouds
from numpy.random import default_rng



logger = logging.getLogger(__name__)



class MeshLoss(nn.Module):
    def __init__(
        self,
        chamfer_weight=1.0,
        normal_weight=1.6e-4,
        edge_weight=0.1,
        lap_weight=0.3,
        gt_num_samples=5000,
        pred_num_samples=5000,
    ):

        super(MeshLoss, self).__init__()
        self.chamfer_weight = chamfer_weight
        self.normal_weight = normal_weight
        self.edge_weight = edge_weight
        self.lap_weight = lap_weight
        self.gt_num_samples = gt_num_samples
        self.pred_num_samples = pred_num_samples
        
        self.skip_mesh_loss = False
        if chamfer_weight == 0.0 and normal_weight == 0.0 and edge_weight == 0.0:
            self.skip_mesh_loss = True

    def forward(self, meshes_pred, meshes_gt,obj_total=True,batch_size=1):
        if not obj_total :
            self.pred_num_samples = 22000
            self.gt_num_samples = 22000
            self.normal_weight = 0.003

        # Sample from meshes_gt if we haven't already
        if isinstance(meshes_gt, tuple):
            points_gt, normals_gt = meshes_gt
        else:
            # points_gt = meshes_gt._verts_list[0].view(1,-1,3)
            # normals_gt = meshes_gt.verts_normals_packed().view(1,-1,3)
            points_gt, normals_gt = sample_points_from_meshes(
                meshes_gt, num_samples=self.gt_num_samples, return_normals=True
            )

        total_loss = torch.tensor(0.0).to(points_gt)
        losses = {}

        if isinstance(meshes_pred, Meshes):
            meshes_pred = [meshes_pred]
        elif meshes_pred is None:
            meshes_pred = []

        # Now assume meshes_pred is a list
        if not self.skip_mesh_loss:
            for i, cur_meshes_pred in enumerate(meshes_pred):
                cur_out = self._mesh_loss(cur_meshes_pred,points_gt, normals_gt,batch_size,obj_total)
                cur_loss, cur_losses = cur_out
                if total_loss is None or cur_loss is None:
                    total_loss = None
                else:
                    total_loss = total_loss + cur_loss / len(meshes_pred)
                for k, v in cur_losses.items():
                    losses["%s_%d" % (k, i)] = v

        return total_loss, losses


    def _mesh_loss(self, meshes_pred, points_gt, normals_gt , batch_size=1,obj_total=True):
        zero = torch.tensor(0.0).to(meshes_pred.verts_list()[0])
        losses = {"chamfer": zero, "normal": zero, "edge": zero}

        points_pred, normals_pred = sample_points_from_meshes(
            meshes_pred, num_samples=self.pred_num_samples, return_normals=True
        )

        total_loss = torch.tensor(0.0).to(points_pred)
        if points_pred is None or points_gt is None:
            # Sampling failed, so return None
            total_loss = None
            which = "predictions" if points_pred is None else "GT"
            logger.info("WARNING: Sampling %s failed" % (which))
            return total_loss, losses

        losses = {}
        cham_loss, normal_loss = chamfer_distance(points_pred, points_gt, x_normals=normals_pred, y_normals=normals_gt)

        total_loss = total_loss + self.chamfer_weight * cham_loss
        total_loss = total_loss + self.normal_weight * normal_loss
        losses["normal"] = normal_loss.to('cpu')
        losses["chamfer"] = cham_loss.to('cpu')
        if not obj_total : 
            edge_loss = mesh_edge_loss(meshes_pred)
            total_loss = total_loss + self.edge_weight * edge_loss
            losses["edge"] = edge_loss.to('cpu')

            lap_loss = mesh_laplacian_smoothing(meshes_pred)
            total_loss = total_loss + self.lap_weight * lap_loss
            losses["lap"] = lap_loss.to('cpu')


        return total_loss, losses
