import numpy as np
from plyfile import PlyData, PlyElement


def read_all_ply(filename):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()
    colors = np.stack((plydata['vertex']['red'], plydata['vertex']['green'], plydata['vertex']['blue'])).transpose()
    try:
        labels = plydata['vertex']['objectId']
    except:
        try:
            labels = plydata['vertex']['label']
        except:
            labels = np.array([])        
    try:
        faces = np.array(plydata['face'].data['vertex_indices'].tolist())
    except:
        faces = np.array([])

    file.close()

    return points, labels, colors, faces


def read_ply(filename, points_only=False):
    """ Reads a PLY file from disk.
    Args:
    filename: string
    
    Returns: np.array, np.array, np.array
    """
    file = open(filename, 'rb')
    plydata = PlyData.read(file)
    points = np.stack((plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z'])).transpose()

    if points_only:
        return points
    try:
        labels = plydata['vertex']['objectId']
    except:
        try:
            labels = plydata['vertex']['label']
        except:
            labels = np.array([])        
    try:
        faces = np.array(plydata['face'].data['vertex_indices'].tolist())
    except:
        faces = np.array([])

    try:
        masks = plydata['vertex']['mask']
    except:
        masks = np.array([])

    file.close()

    return points, labels, faces, masks


def write_ply(filename, points, mask=None, faces=None):
    """ Writes a set of points, optionally with faces, labels and a colormap as a PLY file to disk.
    Args:
    filename: string
    points: np.array
    faces: np.array
    labels: np.array
    colormap: np.array
    """
    colors = [[0, 0, 0], [0, 255, 0], [0, 128, 0], [0, 0, 255]]
    with open(filename, 'w') as file:

        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write('element vertex %d\n' % points.shape[0])
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if mask is not None:
            file.write('property ushort mask\n')
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')
        
        if faces is not None:
            file.write('element face %d\n' % faces.shape[0])
            file.write('property list uchar int vertex_indices\n')

        file.write('end_header\n')

        if mask is None:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f\n' % (points[point_i, 0], points[point_i, 1], points[point_i, 2]))
        else:
            for point_i in range(points.shape[0]):
                file.write('%f %f %f %i %i %i % i\n' % (points[point_i, 0], points[point_i, 1], points[point_i, 2], mask[point_i], colors[mask[point_i]][0], colors[mask[point_i]][1], colors[mask[point_i]][2]))

        if faces is not None:
            for face_i in range(faces.shape[0]):
                file.write('3 %d %d %d\n' % (
                    faces[face_i, 0], faces[face_i, 1], faces[face_i, 2]))

import json
def get_label_name_to_global_id(file):
    labelName2InstanceId = {} #{'teapot_2': 8}
    classId2ClassName = {} #{ 1: 'fork'}

    json_path = file.split('.')[0]+'_view-2.json'
    with open(json_path, 'r') as f:
        dict_out = json.load(f)
        for obj in dict_out['objects']:
            labelName2InstanceId[obj['name']] = obj['global_id']
            classId2ClassName[obj['global_id']] = obj['class']
    if not labelName2InstanceId or not classId2ClassName:
        raise ValueError(f"labelName2classId or classId2ClassName of {file} shouldn't be empty!")
    return labelName2InstanceId, classId2ClassName

def farthest_point_sampling(vertices: np.ndarray, num_samples: int) -> np.ndarray:
    vertices = np.array(vertices)
    num_vertices = vertices.shape[0]
    if num_samples >= num_vertices:
        return vertices
    centroids = np.zeros((num_samples, 3))
    centroids[0] = vertices[np.random.randint(num_vertices)]#!随机选择输入顶点集中的一个顶点作为第一个提取的顶点
    distance = np.linalg.norm(vertices - centroids[0], axis=1)#! 计算输入顶点集中的所有顶点与第一个提取的顶点之间的距离，并将结果存储在distance数组中
    for i in range(1, num_samples):#! 开始一个循环，从第二个提取的顶点开始，直到提取num_samples个顶点。
        centroids[i] = vertices[np.argmax(distance)] #! 找到距离数组distance中具有最大值的索引，并将对应的输入顶点作为下一个提取的顶点存储在centroids中。
        distance = np.minimum(distance, np.linalg.norm(vertices - centroids[i], axis=1))#! 更新distance数组，将每个顶点与当前提取的顶点之间的距离与原始距离相比较，保留较小的距离值。这一步用于确保在选择下一个提取的顶点时，我们找到的是距离当前已提取顶点集的最远顶点。
    #print(len(centroids))
    return centroids

def get_points_instances_from_mesh(file, labelName2InstanceId, num_samples=5625):
    """ 一行一行读obj, 若是v, 则是vertex, 若是o, 则是object的name, 每存储一个v, 则存储一个instance_id
    需要name_to_id的dict
    Input: file: e.g. .../7b4acb843fd4b0b335836c728d324152_0001_5_scene-X-bowl-X_mid.obj 
           labelName2InstanceId { "cup_1":50 ....}
    Output: vertices(num_of_vertices,3); instance(num_of_vertices, ) """    
    instances_points = {}

    with open(file, 'r') as f:
        for line in f:
            if line.startswith('o'):
                label_name = line.strip()[2:]
                instance_id = labelName2InstanceId[label_name]
                instances_points[instance_id] = []
            elif line.startswith('v'):
                vertex = list(map(float, line.split()[1:]))
                if len(vertex) == 3:
                    instances_points[instance_id].append(vertex)
    #利用fps采取点
    instances = np.array([])
    points = np.empty((0,3))
    for id in instances_points:
        instances_points[id] = farthest_point_sampling(instances_points[id], num_samples=num_samples)
        instances = np.concatenate(( instances, np.full(len(instances_points[id]),id) ))
        points = np.concatenate(( points,  instances_points[id] ))
    return points, instances
