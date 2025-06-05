import os
from pytorch3d.io import IO
import torch
import glob
from mpfgan.models import Generator
from Data_loader import DataLoader

G = Generator().cuda()

io = IO()

D_loader = DataLoader()

folders = glob.glob("test/*")

PATH = os.path.join('epoch_185.pth')   # last_1001.pth
original = torch.load(PATH)['G_state_dict']
PATH = os.path.join('results','last_1002.pth')
now1 = torch.load(PATH)['G_state_dict']
now = now1
keys_o = original.keys()
keys_now = now.keys()
for (i,key_now) in enumerate(keys_now):
    key_o = list(keys_o)[i]
    now[key_now] = original[key_o]




G.load_state_dict(original['G_state_dict'])
with torch.no_grad():
    meshes_com = []
    for img in folders:
        imgT , imgL = D_loader.importimg(img)
        landm = D_loader.getlandmarks(imgL)
        landm = torch.tensor(landm)
        _ , mesh = G(imgT.cuda(),landm.view(1,-1).cuda())
        meshes_com.append(mesh)