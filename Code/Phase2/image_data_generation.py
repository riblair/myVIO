import bpy
import mathutils
import math
import Path
from PathGrapher import PathGrapher
import argparse
import numpy as np
import os

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
    bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    return scene

def render_scene(scene, file_path):
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = file_path
    bpy.ops.render.render(write_still = 1)
    bpy.data.images.remove(bpy.data.images['Render Result']) # to prevent memory backup

def generate_image_data(args, path):
    scene = setup_scene()
    pg = PathGrapher(path)
    times, positions, orientations = pg._generate_ground_truth('euler')
    im_names = [f"{args.Outputs}Images/traj_{path.name}_{(i):05}.png" for i in range(len(times))]
    camera_obj = bpy.data.objects['Camera']
    for (i_name, p, o) in zip(im_names, positions, orientations):
        new_p = mathutils.Vector(p)
        new_o = mathutils.Euler(o)
        print(new_o)
        camera_obj.location = new_p
        camera_obj.rotation_euler = new_o
        bpy.context.view_layer.update()
        render_scene(scene, i_name)
    # combines images into a video
    os.system(f"ffmpeg -framerate 200 -y -pattern_type glob -i '{args.Outputs}Images/traj_{path.name}_*.png' -c:v libx264 -pix_fmt yuv420p {args.Outputs}_traj_{path.name}.mp4")

if __name__ == '__main__':
    args = env_setup()
    if args.Path == "straight_line":
        path = Path.STRAIGHT_LINE
    elif args.Path == 'circle':
        path = Path.CIRCLE
    elif args.Path == 'sinusoid':
        path = Path.SINUSOID
    elif args.Path == 'figure_eight':
        path = Path.FIGURE_EIGHT
    elif args.Path == 'hyperbolic_paraboloid':
        path = Path.HYPERBOLIC_PARABOLOID
    else:
        print(f"[ERROR] Wrong type given for '--Path' param. Expected ['straight_line', 'circle', 'sinusoid', 'figure_eight', 'hyperbolic_paraboloid'], given {args.Path}")
        exit(1)
    generate_image_data(args, path)