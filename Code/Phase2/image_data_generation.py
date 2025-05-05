import bpy
import mathutils
import math
import Path
from PathGrapher import PathGrapher
import argparse
import numpy as np
import os
import sys

def env_setup():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--Outputs", default="Code/Phase2/Data/", type=str, help="Parent Directory for data files. Default:'Code/Phase2/Data/'")
    Parser.add_argument("--Path", default="straight_line", type=str, help="Which path object to use. ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid']")
    Args = Parser.parse_args()

    os.makedirs(Args.Outputs, exist_ok=True)
    os.makedirs(Args.Outputs+"Images/", exist_ok=True)
    return Args

def setup_scene():
    scene = bpy.context.scene
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100 

    for obj in list(bpy.data.objects):
        if obj.name != "Camera":
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.name = "Floor"
    plane.scale = (160, 160, 1)
    image = bpy.data.images.load(os.path.abspath("Code/Phase2/BlenderStuff/floor.webp"))
    mat = bpy.data.materials.new(name="FloorMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for node in nodes:
        nodes.remove(node)
    tex_image = nodes.new('ShaderNodeTexImage')
    tex_image.image = image
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    plane.data.materials.append(mat)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    # UNCOMMENT IF YOU WANT TO REMOVE CONSOLE PRINTING
    devnull = os.open(os.devnull, os.O_WRONLY) 
    os.dup2(devnull, 1)  # Redirect stdout (fd 1)
    os.dup2(devnull, 2)
    # Blender on the cluster required BLENDER_EVEE_NEXT, uncomment if running on cluster.
    # bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    return scene

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)
    bpy.data.images.remove(bpy.data.images['Render Result']) # to prevent memory backup

def generate_image_data(directory:str, times: np.ndarray, positions: np.ndarray, euler_angles: np.ndarray):
    scene = setup_scene()
    im_names = [f"{directory}Images/im_{(i):05}.png" for i in range(len(times))]
    camera_obj = bpy.data.objects['Camera']
    # a = 1628
    # iter = 0
    for (i_name, p, o) in zip(im_names, positions, euler_angles):
        # if iter < a:
        #     iter+=1
        #     continue
        new_p = mathutils.Vector(p)
        new_o = mathutils.Euler(o)
        print(new_o)
        camera_obj.location = new_p
        camera_obj.rotation_euler = new_o
        bpy.context.view_layer.update()
        render_scene(scene, i_name)
    # combines images into a video
    os.system(f"ffmpeg -framerate 200 -y -pattern_type glob -i '{directory}Images/im_*.png' -c:v libx264 -pix_fmt yuv420p {directory}video.mp4")