import glob
import torch
import torchvision.transforms as T
from skimage import io as ioo
from skimage.transform import resize
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
import face_alignment

# Initialize face alignment once (you can move this into the class if needed)
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.THREE_D, device="cuda", flip_input=True
)

device = torch.device("cpu")  # Or "cuda" if using GPU


class DataLoader:
    def __init__(self, path=None, path1=None, path2=None, path3=None):
        self.path = path
        self.path1 = path1
        self.path2 = path2
        self.path3 = path3

    def get_landmarks(self, img):
        return fa.get_landmarks(img)[-1] / 450

    def import_img(self, img_path):
        im = ioo.imread(img_path)
        img = resize(im, (112, 112)).astype("float32")
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(img)
        return img_tensor.reshape(1, 3, 112, 112), im

    def import_mesh(self, obj_path):
        verts, faces, _ = load_obj(obj_path)
        verts = verts.to(device)
        faces_idx = faces.verts_idx.to(device)
        mesh = Meshes(verts=[verts], faces=[faces_idx])
        return mesh

    def import_tables_r(self, npy_path):
        return torch.tensor(torch.load(npy_path)).view(1, -1, 7)

    def __call__(self,frontal=0):
        meshes, imgs, eyes, mouths, tables_r, landmarks = [], [], [], [], [], []
        folders = sorted(glob.glob(self.path))

        while frontal < 2000 and frontal < len(folders):
            if frontal in {1271, 1336, 1994}:
                frontal += 1
                continue

            pathimg = folders[frontal].split("/")[-1][:-4]
            try:
                img_path = f"./AFLW2000/{pathimg}.jpg"
                img_tensor, img = self.import_img(img_path)
                mesh = self.import_mesh(folders[frontal])
                eye_mesh = self.import_mesh(f"{self.path1}{pathimg}.obj")
                mouth_mesh = self.import_mesh(f"{self.path2}{pathimg}.obj")
                table_r = self.import_tables_r(f"{self.path3}{pathimg}.pt")
                landmark = self.get_landmarks(img)

                imgs.append(img_tensor)
                meshes.append(mesh)
                eyes.append(eye_mesh)
                mouths.append(mouth_mesh)
                tables_r.append(table_r)
                landmarks.append(landmark)

            except Exception as e:
                print(f"Error processing {pathimg}: {e}")

            frontal += 1

        return imgs, meshes, eyes, mouths, landmarks, tables_r
