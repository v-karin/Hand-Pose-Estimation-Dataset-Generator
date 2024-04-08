import sys
import os

mano_folder_name = "mano"

mano_root = os.path.abspath(os.path.join(__file__, "..", mano_folder_name))
print(f"MANO Hand Directory: {mano_root}")
sys.path.insert(0, mano_root)
sys.path.insert(0, os.path.join(mano_root, "webuser"))
#print("\n".join(sys.path))

from webuser.smpl_handpca_wrapper_HAND_only import load_model
from webuser.serialization import save_model
import numpy as np
import random as rnd



def save_mano_txt(path: str, model):
    # Convert array to strings
    # pose_position = ", ".join(str(pose[0]) for pose in model.pose[:3])
    # finger_position = ", ".join(str(finger[0]) for finger in model.pose[3:])

    # with open(path, "w") as file:
    #     file.write("\n".join([
    #         f"Pose positions: {pose_position}",
    #         f"Finger positions: {finger_position}"
    #     ]))

    trans = ", ".join(str(tran[0]) for tran in model.trans)
    fullpose = ", ".join(str(param[0]) for param in model.fullpose)
    pose = ", ".join(str(param[0]) for param in model.pose)
    betas = ", ".join(str(beta[0]) for beta in model.betas)

    with open(path, "w") as file:
        file.write("\n".join([
            f"Trans: {trans}",
            f"Full Pose: {fullpose}",
            f"Pose: {pose}",
            f"Betas: {betas}"
        ]))

def save_mano_obj(path: str, model):
    with open(path, "w") as file:
        for v in model.r:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for f in model.f+1: # Faces are 1-based, not 0-based in obj files
            file.write(f"f {f[0]} {f[1]} {f[2]}\n")



# i.e. /handmodel/left/... (pkl, text, obj)
def make_folder_dir(mano_model_base_path: str):
    os.makedirs(os.path.abspath(os.path.join(mano_model_base_path, "pkl")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(mano_model_base_path, "text")), exist_ok=True)
    os.makedirs(os.path.abspath(os.path.join(mano_model_base_path, "obj")), exist_ok=True)



def save_mano_model_pkl_txt_obj(mano_model_base_path: str, name: str, model):
    make_folder_dir(mano_model_base_path)
    save_model(model, os.path.join(mano_model_base_path, "pkl", f"{name}.pkl"))

    # Save MANO parameters as text
    save_mano_txt(os.path.join(mano_model_base_path, "text", f"{name}.txt"), model)

    # Write model mesh to .obj file

    mesh_path = os.path.join(mano_model_base_path, "obj", f"{name}.obj")
    save_mano_obj(mesh_path, model)

    print(f"Output mesh saved to: {mesh_path}")







## Load MANO/SMPL+H model
## Make sure path is correct

def generate_mano_poses(save_folder_root: str, number: int, hand: str):

    m = load_model(os.path.join(mano_root, "models", f"MANO_{hand}.pkl"), ncomps=6, flat_hand_mean=False)

    for i in range(number):
        # Assign random pose and shape parameters
        np.random.seed(0)
        # print("\nBetas Shape:", m.betas.shape)
        # print("Pose Shape:", m.pose.shape)
        # print("Full Pose Shape:", m.fullpose.shape)

        m.betas[:] = np.random.rand(m.betas.size) * .03
        m.pose[:] = np.random.rand(m.pose.size) * 1.0
        # m.pose[:3] = [0., 0., 0.]
        m.pose[:3] = [rnd.uniform(-10, 10), rnd.uniform(-10, 10), rnd.uniform(-10, 10)]
        # m.pose[3:] = [-0.42671473, -0.85829819, -0.50662164, +1.97374622, -0.84298473, -1.29958491]
        m.pose[3:] = [rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(-2, 2)]

        # the first 3 elements correspond to global rotation
        # the next ncomps to the hand pose

        save_folder_path = os.path.join(save_folder_root, f"{hand.lower()}")
        base_name = f"handpose{i}"

        os.makedirs(os.path.abspath(os.path.join(save_folder_path, "debug")), exist_ok=True)
        with open(os.path.join(save_folder_path, "debug", f"{base_name}.txt"), "w") as file:
            file.write("Model:\n\n\n\n\n")
            for k, v in m.__dict__.items():
                file.write(f"Key: {k}\nValue:\n{v}")
                if hasattr(v, "shape"):
                    file.write(f"\nShape: {v.shape}")
                file.write("\n\n\n")

        save_mano_model_pkl_txt_obj(save_folder_path, base_name, m)

def generate_mano_poses_left_right(save_folder_root: str, number: int):
    generate_mano_poses(save_folder_root, number, "LEFT")
    generate_mano_poses(save_folder_root, number, "RIGHT")