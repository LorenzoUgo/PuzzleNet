import trimesh as tm
import os
import numpy as np

data_path = "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/"

train_dir = (
    "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/data_split/everyday.train.txt"
)
with open(train_dir, "r") as f:
    dir_list = [os.path.join(data_path, line.strip()) for line in f.readlines()]

final_data = np.empty((0, 3))

for mesh in dir_list:
    if not os.path.isdir(mesh):
        print(f"{mesh} does not exist")
        continue
    for frac in os.listdir(mesh):
        print(mesh, frac)
        if "fractured" not in frac and "mode" not in frac:
            continue
        if len(os.listdir(os.path.join(data_path, mesh, frac))) == 1:
            continue
        frac = os.path.join(mesh, frac)
        for piece in os.listdir(frac):
            piece = tm.load_mesh(os.path.join(frac, piece))
            pc = tm.sample.sample_surface(piece, 15000)
            pc = np.array(pc[0]) / np.linalg.norm(pc[0], axis=1).reshape(15000, 1)
            final_data = np.concatenate((final_data, pc), axis=0)

np.save("everyday_train.npy", final_data)

test_dir = (
    "/media/tesistiremoti/Volume/MuseoEgizio/Datasets/data_split/everyday.test.txt"
)

with open(train_dir, "r") as f:
    dir_list = [os.path.join(data_path, line.strip()) for line in f.readlines()]

final_data = np.empty((0, 3))

for mesh in dir_list:
    if not os.path.isdir(mesh):
        print(f"{mesh} does not exist")
        continue
    for frac in os.listdir(mesh):
        if "fractured" not in frac and "mode" not in frac:
            continue
        if len(os.listdir(os.path.join(data_path, mesh, frac))) == 1:
            continue
        frac = os.path.join(mesh, frac)
        for piece in os.listdir(frac):
            print(mesh, frac, piece)
            piece = tm.load_mesh(os.path.join(frac, piece))
            pc = tm.sample.sample_surface(piece, 15000)
            pc = np.array(pc[0]) / np.linalg.norm(pc[0], axis=1).reshape(15000, 1)
            final_data = np.concatenate((final_data, pc), axis=0)

np.save("everyday_test.npy", final_data)
