import os
from pytorch3d.io import IO
import torch
import bpy
import glob
from mpfgan.models import Generator
from Data_loader import DataLoader

G = Generator().cuda()

io = IO()

D_loader = DataLoader()

folders = glob.glob("Test_Data/*")

PATH = os.path.join('results','Generator.pth')   # last_1001.pth
checkpoint = torch.load(PATH)


G.load_state_dict(checkpoint)
with torch.no_grad():
    meshes_com = []
    for img in folders:
        title = img.split("/")[-1].split(".")[0]
        imgT , imgL = D_loader.import_img(img)
        landm = D_loader.get_landmarks(imgL)
        landm = torch.tensor(landm)
        mesh = G(imgT.cuda(),landm.view(1,-1).cuda())
        mesh = mesh.detach()
        path='./results/'+ title +'.obj'
        io.save_mesh(data=mesh, path=path)
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        bpy.ops.wm.obj_import(filepath=path)

        # Find the imported mesh object
        imported_objects = [obj for obj in bpy.context.selected_objects if obj.type == 'MESH']
        if not imported_objects:
            raise RuntimeError("No mesh object found in the imported file.")

        obj = imported_objects[0]
        bpy.context.view_layer.objects.active = obj

        # Add Subdivision Surface modifier
        subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf.levels = 2
        subsurf.render_levels = 5
        subsurf.subdivision_type = 'CATMULL_CLARK'
        subsurf.use_creases = True

        # Apply the modifier
        bpy.ops.object.modifier_apply(modifier=subsurf.name)

        # Export using the new 4.0+ OBJ exporter
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
        
        if os.path.exists(path.replace('.obj','.mtl')):
            os.remove(path.replace('.obj','.mtl'))

        print(f"Saved subdivided object to: {path}")