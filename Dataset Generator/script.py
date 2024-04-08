import blenderproc as bproc

### CONSTANTS
MAX_LIGHTS = 4
MAX_CAM = 4
MAX_HANDS = 4
###

import numpy as np
import bpy
import mathutils
import os
import sys

# running from outside folder
canvas_path = os.path.join(os.getcwd(), "Dataset Generator")
sys.path.insert(0, canvas_path)

def new_dir_folder(root: str, name: str) -> str:
    new_dir = os.path.join(root, name)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

from mano_loader import generate_mano_poses_left_right

# import debugpy
# debugpy.listen(5678)
# print("Ready To Link")
# debugpy.wait_for_client()



def items_same_len(items: list[list]):
    it = iter(items)
    first_len = len(next(it))
    return all(len(item) == first_len for item in it), first_len



LocRot = tuple[mathutils.Vector, mathutils.Euler]

class HandPositionsIterator:
    def __iter__(self, hand_positions: dict[str, list[LocRot]], hands: dict[str, bproc.types.Entity], start_frame: int=0):
        if hand_positions.keys() != hands.keys():
            raise Exception("Non-matching dictionary entries")

        items = list(hand_positions.values())
        same_lens, first_len = items_same_len(items)
        if first_len < 1 or not same_lens:
            error_msg = "\n".join(["Mismatching list lengths in hand_positions:",
                ", ".join([f"{len(x)}" for x in items])
            ])
            raise Exception(error_msg)

        self.len = first_len
        self.start_frame = start_frame
        self.pos = 0
        self.hand_positions = hand_positions
        self.hands = hands
        return self

    def __next__(self):
        if self.pos == self.len:
            raise StopIteration

        for key in list(self.hands.keys()):
            hand = self.hands[key]
            hand_position = self.hand_positions[key][self.pos]
            hand.set_location(hand_position[0], self.start_frame + self.pos)
            hand.set_rotation_euler(hand_position[1], self.start_frame + self.pos)
        
        self.pos += 1

        return self.hands
    
    def get_pos(self):
        return self.pos





def get_loc_rots(armatures: list[bproc.types.Armature]) -> list[LocRot]:
    loc_rots = [(object.get_location(), object.get_rotation_euler()) for object in armatures]
    return loc_rots

def loc_rots_from_name(entities: list[bproc.types.Entity], regex: str, max: int=100) -> list[LocRot]:
    armatures = bproc.filter.by_attr(entities, "name", regex, regex=True)
    positions = get_loc_rots(armatures)
    return positions[0:max]



def set_camera(camera: bproc.types.Entity):
    pass
    # data: bpy.types.Camera = camera.blender_obj.data

    # bproc.camera.set_intrinsics_from_blender_params(
    #     lens=data.lens,
    #     lens_unit=data.lens_unit,
    #     clip_start=data.clip_start,
    #     clip_end=data.clip_end,
    #     shift_x=data.shift_x,
    #     shift_y=data.shift_y
    # )

    # already using canvas camera it seems

def camera_matrix(position: LocRot) -> np.ndarray:
    return bproc.math.build_transformation_mat(position[0], position[1])







def get_mano_hand_paths(hands_side_root: str, folder: str) -> list[str]:
    hands = []
    for _, _, names in os.walk(os.path.join(hands_side_root, folder)):
        for name in names:
            hands.append(os.path.join(hands_side_root, folder, name))

    return hands

def get_mano_hand_paths_obj(hands_side_root: str) -> list[str]:
    return get_mano_hand_paths(hands_side_root, "obj")

def get_mano_hand_paths_txt(hands_side_root: str) -> list[str]:
    return get_mano_hand_paths(hands_side_root, "text")

def get_mano_hand_paths_pair(input_hands_root: str, side: str):
    hands_side_root = os.path.join(input_hands_root, side)
    side_hand_paths = get_mano_hand_paths_obj(hands_side_root)
    side_hand_paths_txt = get_mano_hand_paths_txt(hands_side_root)

    return side_hand_paths, side_hand_paths_txt



def load_mano_hand(path: str) -> bproc.types.MeshObject:
    return bproc.loader.load_obj(path)[0]

def load_mano_txt(path: str) -> str:
    with open(path) as file:
        return file.read()










def ask_config():
    print("\nConfiguring Renderer")

    resols = [[1920, 1080], [640, 360]]
    noise_thr = [0.1, 0.9]
    max_samp = [4096, 4] # fast - will take ~1m on my laptop

    fast = input("Use fast settings?\nEnter 'f' to apply: ")
    if fast == "f":
        idx = 1
    else:
        idx = 0
    
    print("Selected settings: ", ((idx == 1) and "fast") or "slow")

    config = [resols[idx], noise_thr[idx], max_samp[idx]]
    return config

def configure_renderer(config):
    bproc.camera.set_resolution(config[0][0], config[0][1])
    bproc.renderer.set_noise_threshold(config[1])
    bproc.renderer.set_max_amount_of_samples(config[2])



def main_render(output_root: str, num: int, config: list):
    input_root = os.path.join(canvas_path, "input")
    input_hands_root = os.path.join(input_root, "hands")

    yes = input("Generate MANO obj poses?\nEnter 'y' to apply: ")
    if yes == "y":
        generate_mano_poses_left_right(input_hands_root, num)

    output_rgb_dir = new_dir_folder(output_root, "rgb")
    output_seg_dir = new_dir_folder(output_root, "segmap")
    output_dep_dir = new_dir_folder(output_root, "depth")

    master_list = []
    
    print("\nSetting up...")

    bproc.clean_up(True)

    objs = bproc.loader.load_blend(
        os.path.join(canvas_path, "dataset_canvas.blend"),
        obj_types=["mesh", "armature", "light", "camera"]
    )

    camera = bproc.filter.one_by_attr(objs, "name", "Canvas_Camera")
    light = bproc.filter.one_by_attr(objs, "name", "Canvas_Light")
    camera_positions = loc_rots_from_name(objs, "Camera_Position.*", MAX_CAM)
    light_positions = loc_rots_from_name(objs, "Light_Position.*", MAX_LIGHTS)

    existing_hands = bproc.filter.by_attr(objs, "name", "_handpose.*", regex=True)
    for hand in existing_hands:
        hand.blender_obj.hide_set(True)
        hand.blender_obj.hide_render = True



    left_hand_paths, left_hand_paths_txt = get_mano_hand_paths_pair(input_hands_root, "left")
    right_hand_paths, right_hand_paths_txt = get_mano_hand_paths_pair(input_hands_root, "right")

    left_hands = [load_mano_hand(left_hand_paths[n]) for n in range(num)]
    right_hands = [load_mano_hand(right_hand_paths[-n - 1]) for n in range(num)]

    for hand in left_hands:
        hand.set_location(mathutils.Vector((0, -1000, 0)), 0)
    
    for hand in right_hands:
        hand.set_location(mathutils.Vector((0, -1000, 0)), 0)

    left_hand_params_list = [load_mano_txt(left_hand_paths_txt[n]) for n in range(num)]
    right_hand_params_list = [load_mano_txt(right_hand_paths_txt[-n - 1]) for n in range(num)]



    print(f"Camera Positions Length: {len(camera_positions)}")
    print(f"Light Positions Length: {len(light_positions)}")
    print(f"Hand Poses Length: {num}")
    set_camera(camera)
    #camera.blender_obj.hide_set(True)

    i = 0
    for light_pos in light_positions:
        for camera_pos in camera_positions:
                for n in range(num):
                    light.set_location(light_pos[0], i)
                    light.set_rotation_euler(light_pos[1], i)

                    bproc.camera.add_camera_pose(camera_matrix(camera_pos), i)

                    left_hand = left_hands[n] # already handled indexing
                    left_hand.set_location(mathutils.Vector((0, 0, 0)), i)
                    left_hands[n - 1].set_location(mathutils.Vector((0, -1000, 0)), i) # set previous hand back

                    right_hand = right_hands[n] # already handled indexing
                    right_hand.set_location(mathutils.Vector((0, 0, 0)), i)
                    right_hands[n - 1].set_location(mathutils.Vector((0, -1000, 0)), i) # set previous hand back

                    left_hand_params = left_hand_params_list[n]
                    right_hand_params = right_hand_params_list[n]

                    master_list.append([i, light_pos, camera_pos, left_hand_params, right_hand_params])
                    i += 1



    print(f"Number of Keyframes: {bproc.utility.num_frames()}")
    configure_renderer(config)
    
    # bproc.world.set_world_background_hdr_img(os.path.join(input_root, "background", "office.jpeg"))


    input("\nReady to run\nPress anything to continue:")

    print(f"\nRunning Renderer...")

    # (not needed - can re-enable if wanted later)
    #bproc.renderer.enable_depth_output(False, output_dir=output_dep_dir, file_prefix=f"depth_")
    #bproc.renderer.enable_segmentation_output(default_values={"category_id": 0}, output_dir=output_seg_dir, file_prefix=f"segmap_")

    bproc.renderer.render(output_rgb_dir, verbose=True, file_prefix=f"rgb_", return_data=False)

    input("\nFinished Rendering\nEnter anything to continue:")
    return master_list







### MAIN RUN ###

print("Setting Up")
bproc.init()

config = ask_config()

num = MAX_HANDS
output_root = new_dir_folder(canvas_path, "output")
master_list = main_render(output_root, num, config)



import csv

with open(os.path.join(output_root, "params.csv"), "w") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["idx", "light_pos", "cam_pos", "left_hand_params", "right_hand_params"])
    for row in master_list:
        writer.writerow(row)

input("\nSaved results to file\nEnter anything to exit:")
