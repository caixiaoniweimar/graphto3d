import numpy as np
import os
import trimesh
object_list = ['bowl', 'box', 'can', 'cup', 'cutlery', 'obstacle', 'pitcher', 'plate', 'teapot']
root_path = '/media/caixiaoni/xiaonicai-u/models'
vertices_mean = {}
""" for obj in object_list:
    all_vertices = []
    for file in os.listdir(root_path+'/'+obj):
        obj_path = os.path.join(root_path, obj, file)
        if os.path.isfile(obj_path) and obj_path.endswith('.obj'):
            obj_mesh = trimesh.load(obj_path)
            all_vertices.append(len(obj_mesh.vertices))
    
    all_vertices = np.array(all_vertices)
    vertices_mean[obj] = np.mean(all_vertices)

print(vertices_mean) """

#{'bowl': 7084.615384615385, 'box': 134176.95, 'can': 187560.73684210525, 'cup': 3545.0, 'cutlery': 433.0, 'obstacle': 17772.24, 'pitcher': 12611.5, 'plate': 2078.076923076923, 'teapot': 4388.5}
vertices_mean = {'bowl': 7084.615384615385, 'box': 134176.95, 'can': 187560.73684210525, 'cup': 3545.0, 'cutlery': 433.0, 'obstacle': 17772.24, 'pitcher': 12611.5, 'plate': 2078.076923076923, 'teapot': 4388.5}
means_of_mean = np.mean(np.array(list(vertices_mean.values())))
print(means_of_mean) #41072