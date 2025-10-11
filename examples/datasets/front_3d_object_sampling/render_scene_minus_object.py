
#blenderproc run examples/datasets/front_3d_with_improved_mat/render_room_dataset_improved_mat.py /mnt/afs/liuyichen/data/_3D_FRONT/3D-FRONT/ /mnt/afs/liuyichen/data/_3D_FRONT/3D-FUTURE-model/ /mnt/afs/liuyichen/data/_3D_FRONT/3D-FRONT-texture/ /mnt/afs/liuyichen/data/_3D_FRONT/3D-FRONT/7f0b281d-ccbf-4d7c-91a0-103ff95f0a90.json ./resources/cctextures/ examples/datasets/front_3d_with_improved_mat/output
import blenderproc as bproc

import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("front_json", help="Path to the 3D front json file")
parser.add_argument("front_folder", help="Path to the 3D front file")
parser.add_argument("future_folder", help="Path to the 3D Future Model folder.")
parser.add_argument("front_3D_texture_path", help="Path to the 3D FRONT texture folder.")
parser.add_argument('output_dir', nargs='?', default="examples/datasets/front_3d_object_sampling/output",
                    help="Path to where the final files, will be saved")
args = parser.parse_args()

from pathlib import Path
def get_folders(args):
    front_dir = Path(args.front_folder)
    future_dir = Path(args.future_folder)
    front_3D_texture_dir = Path(args.front_3D_texture_path)
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    return front_dir, future_dir, front_3D_texture_dir, output_dir

front_dir, future_dir, front_3D_texture_dir, output_dir = get_folders(args)
front_json = front_dir.joinpath(args.front_json)
front_json_id = front_json.stem

if not future_dir.exists():
    raise Exception(f"Future folder does not exist: {future_dir}")

if not front_3D_texture_dir.exists():
    raise Exception(f"Front 3D texture folder does not exist: {front_3D_texture_dir}")

if not os.path.exists(args.front_json) or not os.path.exists(args.future_folder) or not os.path.exists(
        args.front_3D_texture_path):
    raise OSError("One of the three folders does not exist!")

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
bproc.init(compute_device='cuda:0', compute_device_type="CUDA")
# bproc.init()

# mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "blender_label_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# set the light bounces
bproc.renderer.set_light_bounces(diffuse_bounces=50, glossy_bounces=50, max_bounces=50,
                                 transmission_bounces=50, transparent_max_bounces=50)

# read 3d future model info
with open(future_dir.joinpath('model_info_revised.json'), 'r') as f:
    model_info_data = json.load(f)
model_id_to_label = {m["model_id"]: m["category"].lower().replace(" / ", "/") if m["category"] else 'others' for
                        m in
                        model_info_data}

# load the front 3D objects
room_objs = bproc.loader.load_front3d(
    json_path=args.front_json,
    future_model_path=args.future_folder,
    front_3D_texture_path=args.front_3D_texture_path,
    label_mapping=mapping,
    model_id_to_label=model_id_to_label
)

# # ==================Wall material improvement==================
# for obj in bproc.object.get_all_mesh_objects():
#     obj_name = obj.get_name().lower()
#     if "wall" in obj_name:
#         print(f"[Debug] ############### Improving material for wall object: {obj_name} ################")
#         materials = obj.get_materials()
#         if not materials:
#             mat = obj.new_material(name=f"{obj_name}_material")
#         else:
#             mat = materials[0]
        
#         # Add subtle texture variation using noise
#         principled_bsdf = mat.get_the_one_node_with_type("BsdfPrincipled")
#         noise_texture = mat.new_node("ShaderNodeTexNoise")
#         noise_texture.inputs["Scale"].default_value = 5.0
#         noise_texture.inputs["Detail"].default_value = 2.0
        
#         color_ramp = mat.new_node("ShaderNodeValToRGB")
#         color_ramp.color_ramp.elements[0].color = [0.88, 0.88, 0.88, 1.0]
#         color_ramp.color_ramp.elements[1].color = [0.95, 0.95, 0.95, 1.0]
        
#         mat.link(noise_texture.outputs["Fac"], color_ramp.inputs["Fac"])
#         mat.link(color_ramp.outputs["Color"], principled_bsdf.inputs["Base Color"])
        
#         principled_bsdf.inputs["Roughness"].default_value = 0.8

# # =========================================================


def cast_ray_for_camera_position(object_location, direction, target_object, max_distance=20.0, min_distance=0.5):
    """
    Cast a ray from the object location in a given direction to find optimal camera position.
    Returns the position where the ray hits something or reaches max_distance.
    Exclude the target obj itself
    """
    ray_start = object_location
    total_distance_traveled = 0.0
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations and total_distance_traveled < max_distance:
        remaining_distance = max_distance - total_distance_traveled
        ray_direction = direction * remaining_distance
        
        # Perform ray casting to find intersection
        result = bproc.object.scene_ray_cast(ray_start, ray_direction, float(remaining_distance))
        hit, hit_location, hit_normal, index, hit_object, matrix = result

        print(f"Ray casting iteration {iteration + 1}: hit={hit}, hit_location={hit_location}, hit_object={hit_object.get_name() if hit_object else 'None'}")

        if hit_location is not None:
            hit_distance = np.linalg.norm(hit_location - ray_start)
            total_distance_traveled += hit_distance
            
            # Check if we hit the target object itself
            if hit_object == target_object:
                print(f"Hit target object itself, continuing ray from hit point...")
                # Continue ray from hit point with small offset to avoid self-intersection
                ray_start = hit_location + direction * 0.05  # Small offset in the direction
                iteration += 1
                continue
            else:
                if total_distance_traveled > min_distance:
                    # avoid being too close to walls
                    camera_distance = max(min_distance, total_distance_traveled * 0.9)
                    camera_location = object_location + direction * camera_distance
                    return camera_location, total_distance_traveled
        
        # Ray didn't hit anything or we've exhausted our iterations
        break
    
    # Ray didn't hit anything valid or reached max distance
    camera_location = object_location + direction * max_distance
    return camera_location, max_distance


def find_optimal_camera_positions(target_object, num_cameras=16):
    """
    Find optimal camera positions around a target object using ray casting.
    """
    object_location = np.mean(target_object.get_bound_box(), axis=0)
    object_size = np.mean(np.max(target_object.get_bound_box(), axis=0) - np.min(target_object.get_bound_box(), axis=0))
    cam_fov = 40 #default?
    blocking_max_ratio = 0.7 # Max ratio of target object blocking allowed in one axis
    min_distance = object_size / (blocking_max_ratio * 2.8) / (np.tan(np.radians(cam_fov) / 2))
    print(f"[Debug] Object size: {object_size:.3f}, Min camera distance: {min_distance:.3f}")

    camera_poses = []

    # Generate multiple directions around the object
    # Use spherical coordinates to generate evenly distributed directions
    # elevations = np.linspace(10, 170, 4)
    elevations = [-45, -30, 30, 75]
    azimuths = [0, 45, 90, 135, 180, 225, 270, 315]

    for elevation in elevations:
        for azimuth in azimuths:
            # Convert spherical to cartesian coordinates
            elevation_rad = np.radians(elevation)
            azimuth_rad = np.radians(azimuth)
            direction = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            camera_location, distance = cast_ray_for_camera_position(object_location, direction, target_object, min_distance=min_distance)
            toward_direction = object_location - camera_location
            toward_direction += np.random.uniform(-0.1, 0.1, size=3) * object_size

            rotation_matrix = bproc.camera.rotation_from_forward_vec(toward_direction,
                                                                        inplane_rot=0)
            cam2world_matrix = bproc.math.build_transformation_mat(camera_location, rotation_matrix)

            # Check if the target object is visible from this camera position
            bvh_tree = bproc.object.create_bvh_tree_multi_objects(bproc.object.get_all_mesh_objects())
            visible_objects = bproc.camera.visible_objects(cam2world_matrix, sqrt_number_of_rays=20)

            if target_object in visible_objects:
                camera_poses.append(cam2world_matrix)
                print(f"Added camera at distance {distance:.2f}m from object")


    return camera_poses


# Select furniture objects that could be interesting to focus on
target_objects = []
for obj in room_objs:
    obj_name = obj.get_name().lower()
    # Select various furniture types as potential targets
    if any(furniture_type in obj_name for furniture_type in
           ["table", "desk", "chair", "sofa", "bed", "cabinet", "shelf", "tv"]):
        target_objects.append(obj)

if not target_objects:
    print("No suitable target objects found in the scene!")
    # target_objects = room_objs[:5]  # Fallback to first 5 objects
    import random
    target_objects = random.sample(room_objs, min(5, len(room_objs)))

if not target_objects:
    print("No objects found in the scene!")
    exit(1)


# For now, select the first suitable object, but you could implement other selection criteria
chosen_object = target_objects[0]
print(f"Selected target object: {chosen_object.get_name()}")

# Find optimal camera positions using ray casting
camera_poses = find_optimal_camera_positions(chosen_object, num_cameras=16)

if not camera_poses:
    print("No valid camera poses found!")
    exit(1)

import random
camera_poses = random.sample(camera_poses, min(3, len(camera_poses)))

# ====================DEBUG INFO=====================
print(f"Found {len(camera_poses)} valid camera poses")
obj_center = np.mean(chosen_object.get_bound_box(), axis=0)

sep = "=" * 60
print(f"\n{sep}")
print("CAMERA POSES DEBUG INFORMATION")
print(sep)
print(f"Target object: {chosen_object.get_name()}")
print(f"Number of camera poses found: {len(camera_poses)}")
print(sep)
for i, pose in enumerate(camera_poses, 1):
    loc = pose[:3, 3]
    dist = np.linalg.norm(loc - obj_center)
    print(f"Camera {i:02d}:")
    print(f"  Location : [{loc[0]:8.3f}, {loc[1]:8.3f}, {loc[2]:8.3f}]  | Distance: {dist:6.3f} m")
    print("  Matrix:")
    for r in range(4):
        print("    [" + ", ".join(f"{pose[r, c]:8.3f}" for c in range(4)) + "]")
    print()
print(f"{sep}\n")
#===============================================

bproc.camera.set_intrinsics_from_blender_params(
    lens=80, lens_unit='FOV',
    image_width=240, image_height=180,
)

for pose in camera_poses:
    bproc.camera.add_camera_pose(pose)

# ====================
png_dir = os.path.join(args.output_dir, front_json_id, "rgb_png")
os.makedirs(png_dir, exist_ok=True)

hdf5_dir = os.path.join(args.output_dir, front_json_id, "hdf5")
os.makedirs(hdf5_dir, exist_ok=True)

png_dir_without = os.path.join(args.output_dir, front_json_id, "rgb_png_without_object")
os.makedirs(png_dir_without, exist_ok=True)

hdf5_dir_without = os.path.join(args.output_dir, front_json_id, "hdf5_without_object")
os.makedirs(hdf5_dir_without, exist_ok=True)

# ==============================
bproc.renderer.set_cpu_threads(56)
bproc.renderer.set_output_format(file_format="PNG", color_depth=8)
data = bproc.renderer.render()

bproc.writer.write_hdf5(hdf5_dir, data)

print(f"Rendering With-Object Dataset complete. Output saved to {hdf5_dir}")

# Now render the same scene with the same cameras, but remove the target object
print(f"\nRemoving target object: {chosen_object.get_name()}")
chosen_object.hide()

os.makedirs(hdf5_dir_without, exist_ok=True)

data_without = bproc.renderer.render()

bproc.writer.write_hdf5(hdf5_dir_without, data_without)

print(f"Rendering Without-Object Dataset complete. Output saved to {hdf5_dir_without}")

def convert_hdf5_png(input_dir, output_dir):
    import h5py
    import glob
    import imageio
    all_hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    os.makedirs(output_dir, exist_ok=True)
    for hdf5_file in all_hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            for key in f.keys():
                if key == "colors":
                    # convert the colors to numpy array and save it as png
                    colors = f[key][:]
                    colors = colors.astype(np.uint8)
                    colors = colors.reshape(colors.shape[0], colors.shape[1], 3)
                    colors = colors.reshape(colors.shape[0], colors.shape[1], 3)
                    # save the colors by imageio
                    imageio.imwrite(os.path.join(output_dir, os.path.basename(hdf5_file).replace(".hdf5", ".png")), colors)
                    
convert_hdf5_png(hdf5_dir, png_dir)
convert_hdf5_png(hdf5_dir_without, png_dir_without)