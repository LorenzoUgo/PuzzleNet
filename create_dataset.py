import trimesh as tm
import os
import numpy as np
import open3d as o3d

# data_path = "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/everyday"

# final_data = []

# for obj_type in os.listdir(data_path):
#     for code in os.listdir(os.path.join(data_path, obj_type)):
#         for frac in os.listdir(os.path.join(data_path, obj_type, code)):
#             if len(os.listdir(os.path.join(data_path, obj_type, code, frac))) != 1:
#                 continue
#             print(data_path, obj_type, code, frac)
#             frac = os.path.join(data_path, obj_type, code, frac)
#             for piece in os.listdir(frac):
#                 piece = tm.load_mesh(os.path.join(frac, piece))
#                 pc = tm.sample.sample_surface(piece, 15000)
#                 pc = np.array(pc[0]) / np.linalg.norm(pc[0], axis=1).reshape(15000, 1)
#                 print(pc.shape)
#                 final_data.append(pc)

# final_data = np.array(final_data)

# np.save("everyday.npy", final_data)

data_path = "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/ModelNet40/"

final_data = []

for obj_type in os.listdir(data_path):
    if obj_type == "bed" or obj_type == "airplane":
        for obj in os.listdir(os.path.join(data_path, obj_type, "train")):
            piece = o3d.io.read_point_cloud(os.path.join(data_path, obj_type, obj))
            piece = np.array(piece.points) / np.linalg.norm(
                piece.points, axis=1
            ).reshape(len(piece.points), 1)
            print(piece.shape)
            final_data.append(piece)

final_data = np.array(final_data)
np.save("modelnet40.npy", final_data)
