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

def get_points_instances_from_mesh(file, labelName2InstanceId):
    """ 一行一行读obj, 若是v, 则是vertex, 若是o, 则是object的name, 每存储一个v, 则存储一个instance_id
    需要name_to_id的dict
    Input: file: e.g. .../7b4acb843fd4b0b335836c728d324152_0001_5_scene-X-bowl-X_mid.obj 
           labelName2InstanceId { "cup_1":50 ....}
    Output: vertices(num_of_vertices,3); instance(num_of_vertices, ) """    
    points = np.empty((0, 3))
    instances = np.empty((0, ))
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('o'):
                label_name = line.strip()[2:]
                instance_id = int(labelName2InstanceId[label_name])
            elif line.startswith('v'):
                vertex = list(map(float, line.split()[1:]))
                if len(vertex)==3:
                    #若vertex已经在points中, 需要检查拥有该vertex的instance_id是否相同
                    if vertex in points:
                        addVertex = True
                        indices = np.where( (points == vertex).all(axis=1))[0]
                        for index in indices:
                            if instances[index] == instance_id: #相同instance_id的vertex已经有了
                                addVertex = False
                                break #退出循环
                        if addVertex:
                            points = np.append(points, np.array([vertex]), axis=0) #! 一一对应
                            instances=np.append(instances, instance_id)
                    else:
                        points = np.append(points, np.array([vertex]), axis=0) #! 一一对应
                        instances=np.append(instances, instance_id)

    assert(points.shape[1]==3)
    assert(instances.ndim ==1)
    return points, instances