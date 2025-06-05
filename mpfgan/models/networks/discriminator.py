import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.ops import GraphConv
from pytorch3d.ops import GraphConv


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        input_dim = 3
        hidden_dims = [16, 32, 48, 64, 80, 96]
        classes = 1
        gconv_init = 'normal'
        
        # Graph Convolution Network
        self.gconvs = nn.ModuleList()
        self.drop1 = nn.Dropout(p=0.3)

        dims = hidden_dims
        self.fc1 = nn.Linear(dims[-1], classes)

        self.conv = GraphConv(input_dim, 16, init=gconv_init, directed=False)
        for i in range(len(dims)-1):
            self.gconvs.append(GraphConv(dims[i], dims[0], init=gconv_init, directed=False))

        self.fin = nn.Sequential(self.fc1,self.drop1)
        
        
    def forward(self, mesh):
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        verts = self.conv(verts,edges)

        for gconv in self.gconvs:
            verts_features = gconv(verts, edges)
            verts = torch.cat((verts,verts_features),dim=1)
        
        out = torch.mean(verts,dim=0)
        # out = self.drop1(out)
        out = self.fin(out)
        out = F.sigmoid(out)
        return out






