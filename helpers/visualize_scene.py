import open3d as o3d
from helpers.util import fit_shapes_to_box, params_to_8points_no_rot
from render.lineMesh import LineMesh


def render(predBoxes, shapes_pred, render_shapes=True, render_boxes=False, colors=None):
    """ 
    param predBoxes: denormalized bounding box param6
    param shapes_pred: predicted point clouds
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    ren_opt = vis.get_render_option()
    ren_opt.mesh_show_back_face = True
    ren_opt.line_width = 50.

    edges = [0,1], [0,4], [0,3], [1,2],[1,5], [2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]

    for i in range(len(predBoxes)-1):#! -1为了最后去掉的属于0的预测

        vertices = shapes_pred[i] #! 点云中的顶点
        box_vertices = params_to_8points_no_rot(predBoxes[i])
        denorm_shape = fit_shapes_to_box(predBoxes[i], vertices, withangle=False)

        pcd_shape = o3d.geometry.PointCloud()
        pcd_shape.points = o3d.utility.Vector3dVector(denorm_shape)
        pcd_shape_colors = [colors[i % len(colors)] for _ in range(len(denorm_shape))]
        pcd_shape.colors = o3d.utility.Vector3dVector(pcd_shape_colors)
        if render_shapes:
            vis.add_geometry(pcd_shape)

        if render_boxes:
            line_colors = [colors[i % len(colors)] for _ in range(len(edges))]
            line_mesh = LineMesh(box_vertices, edges, line_colors, radius=0.001)
            line_mesh_geoms = line_mesh.cylinder_segments

            for g in line_mesh_geoms:
                vis.add_geometry(g)

    #vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 2]))
    vis.poll_events()
    vis.run()
    vis.destroy_window()
