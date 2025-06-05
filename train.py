import logging
import os
from pytorch3d.io import IO
import torch
import numpy as np
from torch import nn
import pytorch3d
import json
from mpfgan.models import Generator, Discriminator, MeshLoss, MouthLoss
from torch.autograd import Variable
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from Data_loader import DataLoader

path = "./datasetFinal/*"
path1 = "./datasetFinalAll/"
path2 = "./mouths/"
path3 = "./dataset-TablesR/"


data_l = DataLoader(path,path1,path2,path3)


import warnings

warnings.filterwarnings("ignore")


logger: logging.Logger = logging.getLogger(__name__)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if __name__ == "__main__":

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")

    fold = 1999
    imgs, eyes_s, meshes, mouth, landmarks, Tables_R = data_l(fold)
    trn_imgs = imgs[:1900]
    trn_meshes = meshes[:1900]
    trn_landmarks = landmarks[:1900]
    trn_eyes_s = eyes_s[:1900]
    trn_mouth_s = mouth[:1900]
    trn_TablesR = Tables_R[:1900]
    val_TablesR = Tables_R[1900:]
    val_imgs = imgs[1900:]
    val_meshes = meshes[1900:]
    val_landmarks = landmarks[1900:]
    val_mouth_s = mouth[:1900]
    Tables_R = []
    imgs = []
    meshes = []
    landmarks = []
    eyes_s = []
    mouth = []
    with open('./JSON/list.json', 'r') as file:
        data = json.load(file)

    listV = data["listV"]
    mouth = data["mouth"]

    _, faces, _ = eyes_Face = load_obj("Objs/Eyes.obj")
    faces_eyes = faces.verts_idx.to(device)

    _, faces, _ = mouth_Face = load_obj("Objs/Mouth.obj")
    faces_mouth = faces.verts_idx.to(device)


    print("Training Samples: " + str(len(trn_imgs)))
    print("Validation Samples: " + str(len(val_imgs)))
    print("Fin.")

    ## Models
    G = Generator().cuda()
    D = Discriminator().cuda()

    # Losses
    loss_fn_kwargs = {
        "chamfer_weight": 1.0,
        "normal_weight": 0.0016,
        "edge_weight": 0.1,
        "lap_weight": 0.00001,
        "gt_num_samples": 13400,
        "pred_num_samples": 13400,
    }

    mesh_loss = MeshLoss(**loss_fn_kwargs).cuda()
    mouth_loss = MouthLoss("./JSON/edges.json").cuda()
    clf_loss = nn.BCELoss().cuda()

    ## Optimizers
    G_optimizer = torch.optim.Adam(G.parameters(), lr=1e-5)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=1e-5)

    epoch = 0
    step = 0
    total_step = len(trn_imgs)
    best_val_loss = 1000
    print("\n ***************** Training *****************")

    while epoch <= 1000:
        step = 0
        # --------------------------------------------------------------------------------------------
        #   TRAINING
        # --------------------------------------------------------------------------------------------
        D_losses = []
        G_losses = []
        trn_losses = []
        trn_losses_mouth = []
        trn_losses_total = []
        val_losses = []
        print("Epoch: " + str(epoch))

        G.train()
        D.train()
        i = 0
        while i < len(trn_meshes):  # trn_meshes
            step += 1
            TablesR_batch = trn_TablesR[i : 1 * step]
            imgs_batch = trn_imgs[i : 1 * step]
            meshes_batch = trn_meshes[i : 1 * step]
            eyes_batch = trn_eyes_s[i : 1 * step]
            mouth_batch = trn_mouth_s[i : 1 * step]

            landmarks_batch = trn_landmarks[i : 1 * step]
            imgs = torch.tensor([])
            landmarks = torch.tensor([])
            Tables_R = torch.tensor([])
            for img in imgs_batch:
                imgs = torch.cat((imgs, img), 0)
            for table in TablesR_batch:
                Tables_R = torch.cat((Tables_R, table), 0)
            for lan in landmarks_batch:
                landmarks = torch.cat((landmarks, torch.tensor(lan)), 0)
            Tables_R = Tables_R.cuda()
            imgs = imgs.cuda()
            landmarks = landmarks.view(1, -1).cuda()
            meshes = pytorch3d.structures.join_meshes_as_batch(meshes_batch).cuda()
            eyes_s = pytorch3d.structures.join_meshes_as_batch(eyes_batch).cuda()
            mouth_s = pytorch3d.structures.join_meshes_as_batch(mouth_batch).cuda()

            meshes_batch = None
            meshes_G = G(imgs, landmarks)
            ## Update D network
            D_optimizer.zero_grad()
            D_neg = D(meshes_G.detach())
            D_pos = D(meshes)

            loss_D = 0.5 * (
                clf_loss(D_neg, Variable(torch.zeros(D_neg.size()).cuda()))
                + clf_loss(D_pos, Variable(torch.ones(D_pos.size()).cuda()))
            )
            loss_D.backward()
            D_optimizer.step()
            D_losses.append(loss_D.item())
            loss_D = 0

            ## Update G network
            G_optimizer.zero_grad()
            meshes_G = meshes_G
            D_neg = D(meshes_G)
            recon_loss_total, _ = mesh_loss(meshes_G, meshes)
            # Selection of the eyes
            verts0 = torch.tensor([]).cuda()
            mesh_g0 = meshes_G._verts_list[0]
            for v in listV:
                verts0 = torch.cat((verts0, mesh_g0[v]))

            eyes_G = Meshes(verts=[verts0.view(-1, 3)], faces=[faces_eyes])

            recon_loss_eyes, _ = mesh_loss(eyes_G, eyes_s, False)

            recon_loss_mouth = mouth_loss(epoch, meshes_G, Tables_R, i)

            recon_loss = recon_loss_eyes + recon_loss_mouth * 1e-4

            loss_G = (
                recon_loss_total
                + recon_loss
                + 5e-7 * clf_loss(D_neg, Variable(torch.ones(D_neg.size()).cuda()))
            )
            trn_losses.append(recon_loss_eyes.item())
            trn_losses_mouth.append(recon_loss_mouth.item())
            trn_losses_total.append(recon_loss_total.item())
            G_losses.append(loss_G.item())

            loss_G.backward()
            G_optimizer.step()

            i = i + 1

        if epoch % 10 == 0:
            print(epoch)
            torch.save(
                {
                    "G_state_dict": G.state_dict(),
                    "D_state_dict": D.state_dict(),
                    "AG_state_dict": G_optimizer.state_dict(),
                    "AD_state_dict": D_optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join("results/", "last_1002.pth"),
            )

        # Print Summary and update tensorboard
        print(
            "===> Epoch[{}]: Loss_D: {:.4f} Loss_G: {:.4f} Loss_Recon_Total: {:.4f} Loss_Recon_eyes: {:.4f} Loss_Recon_mouth: {:.4f}".format(
                epoch,
                np.mean(D_losses),
                np.mean(G_losses),
                np.mean(trn_losses_total),
                np.mean(trn_losses),
                np.mean(trn_losses_mouth),
            )
        )
        print(
            "---------------------------------------------------------------------------------------\n"
        )
        epoch = epoch + 1
    print("Finished Training")
    # tb.close()
