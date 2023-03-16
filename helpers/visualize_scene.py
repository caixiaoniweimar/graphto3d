import open3d as o3d
import numpy as np
from helpers.util import fit_shapes_to_box, params_to_8points, params_to_8points_no_rot
import json
import torch
from render.lineMesh import LineMesh


def render(predBoxes, predAngles=None, classes=None, classed_idx=None, shapes_pred=None, render_type='points',
           render_shapes=True, render_boxes=False, colors=None):

    if render_type not in ['meshes', 'points']:
        raise ValueError('Render type needs to be either set to meshes or points.')

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    edges = [0,1], [0,4], [0,3], [1,2],[1,5], [2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]
    #[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]

    valid_idx = []
    all_pcl = []
    for i in range(len(predBoxes)-1):#! ？-1是不是为了去掉0?
        shape = shapes_pred[i]#! 预测的点云
        do_render_shape = True
        if render_type == 'points':
            vertices = shape #! 点云中的顶点
        else:#! X
            do_render_shape = False
            if shape is not None:
                if len(shape) == 2:
                    vertices, faces = shape
                    did_fit = True
                else:
                    vertices, faces, did_fit = shape
                do_render_shape = True

        if predAngles is None:
            box_points = params_to_8points_no_rot(predBoxes[i])
            #print(f"predict: {predBoxes[i]}, {len(predBoxes[i])}")
            #print(f"box_points: {box_points}")
            #print()
        else:#!X
            box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
            box_points = params_to_8points(box_and_angle, degrees=True)
            raise ValueError("No predAngles")

        if do_render_shape:
            if predAngles is None:#!获得反归一化后的点云!!!!!!!!!!!!!!!!!!
                denorm_shape = fit_shapes_to_box(predBoxes[i], vertices, withangle=False)
            else:#!X
                box_and_angle = torch.cat([predBoxes[i].float(), predAngles[i].float()])
                denorm_shape = fit_shapes_to_box(box_and_angle, vertices)
                raise ValueError("No predAngles")

        valid_idx.append(i)
        if render_type == 'points':#! enter
            pcd_shape = o3d.geometry.PointCloud()
            pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
            all_pcl += denorm_shape.tolist()
            pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)
            if render_shapes:
                vis.add_geometry(pcd_shape)
        else:#! X
            mesh = o3d.geometry.TriangleMesh()
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.vertices = o3d.utility.Vector3dVector(denorm_shape)
            pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
            mesh.vertex_colors = o3d.utility.Vector3dVector(pcd_shape_colors)

            if did_fit:
                mesh = mesh.subdivide_loop(number_of_iterations=1)
                mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
            mesh.compute_vertex_normals()
            if render_shapes:
                vis.add_geometry(mesh)

        if render_boxes:
            line_colors = [colors[i % len(colors)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_points, edges, line_colors, radius=0.001)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

    #vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()
