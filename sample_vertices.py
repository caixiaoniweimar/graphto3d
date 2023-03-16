import numpy as np

""" 
Farthest Point Sampling (FPS)：
FPS是一种贪婪算法，从输入点云中均匀地选择代表性的点。该方法首先随机选择一个点，然后在每次迭代中找到距离当前点集最远的点。
FPS能够产生较好的点云分布，有利于AtlasNet学习物体的形状特征。

Poisson Disk Sampling：这是一种基于蓝噪声采样的方法，能够在保持空间均匀性的同时，减少采样点之间的规律性。
这有助于提取更好的形状特征，因为它可以捕捉到物体的不同尺度和细节。

Monte Carlo Sampling：这种方法通过在物体表面随机选择点来生成采样点集。
尽管它的采样点可能不如其他方法均匀，但它可以很好地捕捉到物体的全局特征。可以通过增加采样点的数量来提高形状特征的学习效果。

Uniform Mesh Resampling：这种方法通过对原始mesh进行均匀采样来生成一个新的较低分辨率的mesh。
这有助于减少顶点的数量，同时保留物体的形状信息。可以使用一些开源工具（如MeshLab）实现这种采样方法。

总之，在选择采样方法时，您需要权衡采样点的均匀性、局部细节捕捉和计算复杂性等因素。可以尝试不同的采样方法，并使用验证集评估AtlasNet在学习物体特征方面的性能，从而找到最适合您任务的采样策略。
"""
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
    file = "/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/raw/2362ec480b3e9baa4fd5721982c508ad/2362ec480b3e9baa4fd5721982c508ad_0001_4_scene-X-bowl-X_type-X-in-X_goal.obj"
    labelName2InstanceId, _ = get_label_name_to_global_id(file)
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
    
    """     #! 含重复的点
        vertices = np.array(points_list)
        instances = np.array(instances_list)

        num_vertices = vertices.shape[0]
        if num_samples >= num_vertices:
            return vertices
        centroids = np.zeros((num_samples, 3))
        centroids[0] = vertices[np.random.randint(num_vertices)]
        distance = np.linalg.norm(vertices - centroids[0], axis=1)
        for i in range(1, num_samples):
            centroids[i] = vertices[np.argmax(distance)]
            distance = np.minimum(distance, np.linalg.norm(vertices - centroids[i], axis=1))
        return centroids """

