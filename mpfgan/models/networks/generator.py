import torch.nn as nn
import torch
from mpfgan.models.networks.backbone import build_backbone
from mpfgan.models.heads import Reconstruction_Obj
 
from Data_loader import DataLoader
DLoader = DataLoader()

def get_blender_intrinsic_matrix(N=None):
    K = [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
    K = torch.tensor(K)
    if N is not None:
        K = K.view(1, 4, 4).expand(N, 4, 4)
    return K

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()


        self.K = get_blender_intrinsic_matrix()
        self.backbone = build_backbone()
        self.mesh = Reconstruction_Obj()

    def _get_projection_matrix(self, N, device):
        return self.K[None].repeat(N, 1, 1).to(device).detach()

    
    def forward(self, imgs, landmarks=None):  
        N = imgs.shape[0]
        device = imgs.device
        img_feats = self.backbone(imgs)
        # concat_feats = torch.cat((img_feats,z),dim=1)
        P = self._get_projection_matrix(N, device)
#         print(P)
        init_meshes = DLoader.import_mesh('Objs/sphereorrig.obj').extend(N).cuda()
        refined_meshes = self.mesh(img_feats,landmarks, init_meshes, P)
        return refined_meshes
