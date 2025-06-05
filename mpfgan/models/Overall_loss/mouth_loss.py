import logging
import torch
import torch.nn as nn
import numpy as np
import json
import os




logger = logging.getLogger(__name__)


class MouthLoss(nn.Module):
    def __init__(self,edges_f):
        super(MouthLoss,self).__init__()
        with open(edges_f, 'r') as file:
            data = json.load(file)

        self.edges = data["edges"]
        self.listAll = np.unique(np.array(self.edges).reshape(-1), axis=0)
        self.Ref = torch.tensor([1,1]).cuda()
        self.listAll = torch.tensor(self.listAll)

    def cretae_obj(self,path,verts):
        if os.path.exists(path):
            os.remove(path)
        f = open(path, "a")
        for ver in verts:
            temp1 = ver[0].detach().to("cpu").numpy().item()
            temp2 = ver[1].detach().to("cpu").numpy().item()
            f.write("v "+str(temp1) + " " + str(temp2) + " 0\n")
        for edge in self.edges:
            f.write("l "+str(edge[0] + 1) + " " + str(edge[1] + 1) + "\n")
        f.close()

    def deleteaxeZ(self,verts):
        return verts[:,:2]
    
    def getexposeY(self,first_ele,second_ele,point):
        a = (first_ele[1] - second_ele[1])/(first_ele[0] - second_ele[0])
        b = first_ele[1] - a * first_ele[0]
        result = a * point + b 

        return result
    
    def min2p(self,x,y,axe):
        if x[axe] > y[axe]:
            return x , y
        else :
            return y , x
    
    def getexpose(self,verts,point):
        for edge in self.edges:
            x = edge[0]
            y = edge[1]
            max , min = self.min2p(verts[x],verts[y],0)
            if ((point[0] > min[0]) & (point[0] < max[0])):
                break
        exposeY = self.getexposeY(min,max,point[0])
        expose = [point[0] , exposeY]
        exist = True
        return expose , exist
    
    def distance(self,X,Y):
        x1 = torch.sub(X[0],Y[0])
        y1 = torch.sub(X[1],Y[1])
        x = torch.pow(x1,2)
        y = torch.pow(y1,2)
        return torch.sqrt(x + y)
    
    def getX(self,first_ele,second_ele,point):
        a = (first_ele[1] - second_ele[1])/(first_ele[0] - second_ele[0])
        b = first_ele[1] - a * first_ele[0]

        result = (point - b)  / a

        return result
    
    def minimum(self,tensor,axe=0):
        index = 0
        for idx , valeur in enumerate(tensor):
            if valeur[axe] < tensor[index][axe]:
                index = idx
        return tensor[index]
    
    def maximum(self,tensor,axe):
        index = 0
        for idx , valeur in enumerate(tensor):
            if valeur[axe] > tensor[index][axe]:
                index = idx
        return tensor[index]
    
    def getpoints(self,verts,point):
        points = []
        for edge in self.edges:
            x = edge[0]
            y = edge[1]
            max , min = self.min2p(verts[x],verts[y],1)
            if ((point[1] > min[1]) & (point[1] < max[1])):
                pointX = self.getX(min,max,point[1])
                points.append([pointX,point[1]])

        spu = []
        inf = []

        if len(points) > 2:
            for pointe in points :
                if pointe[0] < point[0]:
                    inf.append(pointe)
                else:
                    spu.append(pointe) 
            points = []
            if ((len(spu)> 0) & (len(inf)> 0)):
                min = self.minimum(spu,0)
                max = self.maximum(inf,0)    
                points.append(min)
                points.append(max)
        return points
        
    def forward(self,epoch,mesh,TableR,pth):
        error = None
        verts = mesh._verts_list[0]
        if ((epoch % 5 == 0)  & (pth == 674)):
            path = "/media/mehdi/Disque local/github/SICGAN_2/res/Test/AFLW2000/epoch_" + str(epoch) + "_2D.obj"
            self.cretae_obj(path,verts)
        verts = self.deleteaxeZ(verts)
        erreur_total = 0
        expose = []
        for p0 in self.listAll:
            p = verts[p0]
        #  get Centre
            expose , exist = self.getexpose(verts,p)
            if exist:
                L1 = self.distance(p, expose)
                centre1 =[(p[0]+expose[0])/2,(p[1]+expose[1])/2]
                points = []
            # get point left & right
                points = self.getpoints(verts,centre1)
                if len(points) == 2:
                    L2 = self.distance(points[0], points[1])
                    R = L1 / L2
                    pointd1 = [self.Ref[0],points[0][1]]
                    d1 = self.distance(pointd1, self.Ref)
                    pointd2 = [p[0],self.Ref[1]]
                    d2 = self.distance(pointd2, self.Ref)
                    if ((L1 != 0) & (L2 != 0)) :
                        R = L1 / L2
                        R1 = d1 / L1
                        R2 = d2 / L2
                        newele = [R,R1,R2,L1,L2,d1,d2]
                    for ele in TableR[0]:
                        if (ele != None):
                            al = torch.abs(newele[3] - ele[3]) +torch.abs(newele[4] - ele[4]) + torch.abs(newele[5] - ele[5]) + torch.abs(newele[6] - ele[6])
                            if error != None:
                                if al < error:
                                    error = al
                            else:
                                error = al
                    erreur_total = erreur_total + error
        return erreur_total