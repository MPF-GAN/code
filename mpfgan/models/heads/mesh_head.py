
import torch
import torch.nn as nn
from pytorch3d.ops import GraphConv, SubdivideMeshes, vert_align
from torch.nn import functional as F



def project_verts(verts, P, eps=1e-1):
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]
    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)
    verts_proj = verts_cam_hom[:, :, :3] / w
    if singleton:
        return verts_proj[0]
    return verts_proj

class Reconstruction_Obj(nn.Module):
    def __init__(self):
        super(Reconstruction_Obj, self).__init__()

        input_channels  = 4044
        self.num_stages = 4
        hidden_dim      = 128
        stage_depth     = 3
        graph_conv_init = 'normal'

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            landmarks_feat_dim = 0 if i == 0 else hidden_dim
            stage = Reconstruction_Block(
                input_channels, landmarks_feat_dim, hidden_dim, stage_depth, gconv_init=graph_conv_init
            )
            self.stages.append(stage)


    def forward(self, img_feats, landmarks, meshes, P=None):
        vert_feats = None
        for i, stage in enumerate(self.stages):
            meshes, vert_feats = stage(img_feats, landmarks, meshes, vert_feats, P)
            if i < self.num_stages - 1:
                Triangulation_mesh = SubdivideMeshes()
                meshes, vert_feats = Triangulation_mesh(meshes, feats=vert_feats)
            else:
                return meshes


class Reconstruction_Block(nn.Module):
    def __init__(self, img_feat_dim, landmarks_feat_dim, hidden_dim, stage_depth, gconv_init="normal"):
        super(Reconstruction_Block, self).__init__()

        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.vert_offset = nn.Linear(hidden_dim + 3, 3)

        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + landmarks_feat_dim + 3
            else:
                input_dim = hidden_dim + 3
            gconv = GraphConv(input_dim, hidden_dim, init=gconv_init, directed=False)
            self.gconvs.append(gconv)

        # initialization for bottleneck and vert_offset
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        nn.init.zeros_(self.vert_offset.weight)
        nn.init.constant_(self.vert_offset.bias, 0)

        # self.innom = InstanceNorm(1000)


    def forward(self,img_feats, z, meshes, vert_feats=None, P=None):

        # Project verts if we are making predictions in world space
        vertsgather_valid_tokens_idx = meshes.verts_padded_to_packed_idx()

        if P is not None:
            vert_pos_padded = project_verts(meshes.verts_padded(), P)
            vert_pos_packed = gather_valid_tokens(vert_pos_padded, vertsgather_valid_tokens_idx)
        else:
            vert_pos_padded = meshes.verts_padded()
            vert_pos_packed = meshes.verts_packed()

        # flip y coordinate
        device, dtype = vert_pos_padded.device, vert_pos_padded.dtype
        factor = torch.tensor([1, -1, 1], device=device, dtype=dtype).view(1, 1, 3)
        vert_pos_padded = vert_pos_padded * factor
        vert_align_feats = vert_align(img_feats, vert_pos_padded)

        if z is not None:
            V = vert_align_feats.shape[1]
            z = z[:,None,:].repeat(1,V,1)
            # print(z.shape)
            vert_align_feats = torch.cat((vert_align_feats,z),dim=2)


        vert_align_feats = gather_valid_tokens(vert_align_feats, vertsgather_valid_tokens_idx)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        first_layer_feats = [vert_align_feats, vert_pos_packed]   
        if vert_feats is not None:
            first_layer_feats.append(vert_feats)

        vert_feats = torch.cat(first_layer_feats, dim=1)

        # Run graph conv layers
        for gconv in self.gconvs:
            vert_feats_nopos = F.relu(gconv(vert_feats, meshes.edges_packed()))
            vert_feats = torch.cat([vert_feats_nopos, vert_pos_packed], dim=1)


        vert_offsets = self.vert_offset(vert_feats)
        meshes_out = meshes.offset_verts(vert_offsets)
        return meshes_out, vert_feats_nopos


def gather_valid_tokens(x, idx):
    return x.view(-1, x.shape[-1]).gather(0, idx.view(-1, 1).expand(-1, x.shape[-1]))