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

categories = ["bed", "airplane"]  # TODO: automatically get all categories

for dataset in ["train", "test"]:

    final_data = []

    for obj_type in os.listdir(data_path):
        if obj_type in categories:
            print(f"* * * {categories} - {obj_type} * * *")
            for obj in os.listdir(os.path.join(data_path, obj_type, dataset)):
                #   print(obj)
                piece = o3d.io.read_triangle_mesh(
                    os.path.join(data_path, obj_type, dataset, obj)
                )
                pc = piece.sample_points_uniformly(15000)
                pc = np.array(pc.points) / np.linalg.norm(pc.points, axis=1).reshape(
                    len(pc.points), 1
                )
                final_data.append(pc)

    final_data = np.array(final_data)
    np.save(f"data/" + "_".join(categories) + f"_modelnet40_{dataset}.npy", final_data)
