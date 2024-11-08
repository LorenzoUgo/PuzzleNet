import trimesh as tm
import os
import numpy as np
import open3d as o3d

# %% BREAKING BAD DATASET

data_path = "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/"

for dataset in ["train", "test"]:
    text_dir = f"/media/tesistiremoti/Volume/MuseoEgizio/Datasets/data_split/everyday.{dataset if dataset == 'train' else 'val'}.txt"
    final_data = []
    with open(text_dir, "r") as f:
        mesh_list = [line.strip() for line in f.readlines()]

    for mesh in mesh_list:
        print(f"* * * {dataset} - {mesh} * * *")
        mesh_dir = os.path.join(data_path, mesh)
        if not os.path.isdir(mesh_dir):
            print(f"{mesh} does not exist")
            continue
        for frac in os.listdir(mesh_dir):
            if len(os.listdir(os.path.join(mesh_dir, frac))) != 1:
                continue
            for piece in os.listdir(os.path.join(mesh_dir, frac)):
                piece = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, frac, piece))
                pc = piece.sample_points_uniformly(15000)
                pc = np.asarray(pc.points)
                max_distance = np.max(np.linalg.norm(pc, axis=1))
                pc = pc / max_distance
                pc = pc * 0.5
                pc += np.array([0.5, 0.5, 0.5])
                final_data.append(pc)

    final_data = np.array(final_data)

    np.save(f"data/BreakingBad_everyday_1pc_{dataset}.npy", final_data)

# %% MODELNET40 DATASET

data_path = "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/ModelNet40/"
categories = ["bed", "airplane"]  # TODO: automatically get all categories

for dataset in ["train", "test"]:

    final_data = []

    for obj_type in os.listdir(data_path):
        if obj_type in categories:
            print(f"* * * {dataset} - {obj_type} * * *")
            for obj in os.listdir(os.path.join(data_path, obj_type, dataset)):
                piece = o3d.io.read_triangle_mesh(
                    os.path.join(data_path, obj_type, dataset, obj)
                )
                pc = piece.sample_points_uniformly(15000)
                pc = np.asarray(pc.points)
                centroid = np.mean(pc, axis=0)
                pc = pc - centroid[None, :]
                max_distance = np.max(np.linalg.norm(pc, axis=1))
                pc = pc / max_distance
                pc = pc * 0.5
                pc += np.array([0.5, 0.5, 0.5])
                final_data.append(pc)

    final_data = np.array(final_data)
    np.save(f"data/" + "_".join(categories) + f"_modelnet40_{dataset}.npy", final_data)
