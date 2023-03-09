import torch
print(torch.cuda.is_available())
import numpy as np

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

def remove_duplicates(points, instances):
    """     # concatenate points and instance IDs along axis 1
        data = np.concatenate((points, instances.reshape(-1, 1)), axis=1)

        # get unique rows of data
        unique_data = np.unique(data, axis=0)

        # extract unique points and instance IDs from unique_data
        unique_points = unique_data[:, :-1]
        unique_instances = unique_data[:, -1].astype(int)

        return unique_points, unique_instances """
    # Convert points_list to a numpy array
    points = np.array(points)

    # Get the unique rows and their corresponding indices
    unique_rows, indices = np.unique(points, axis=0, return_index=True)

    # Update instances_list to remove instances corresponding to duplicate points
    instances_list = np.array(instances)[indices]

    # Update points_list to contain only unique points
    points = np.array([list(row) for row in unique_rows])

    # Convert instances_list to a numpy array
    instances = np.array(instances_list)
    return points, instances

def get_points_instances_from_mesh(file, labelName2InstanceId):
    points_list = []
    instances_list = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('o'):
                label_name = line.strip()[2:]
                instance_id = labelName2InstanceId[label_name]
            elif line.startswith('v'):
                vertex = list(map(float, line.split()[1:]))
                if len(vertex) == 3:
                    points_list.append(vertex)
                    instances_list.append(instance_id)

    points = np.array(points_list)
    instances = np.array(instances_list)

    """ existing_indices = np.where(np.isin(points, points, assume_unique=True, invert=True))[0]
    for i in existing_indices:
        indices = np.where(np.all(points[i] == points, axis=1))[0]
        if instances[i] in instances[indices]:
            points = np.delete(points, i, axis=0)
            instances = np.delete(instances, i) """
    points, instances = remove_duplicates(points, instances)

    assert points.shape[1] == 3
    assert instances.ndim == 1
    return points, instances

file_path = '/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/raw/7b4acb843fd4b0b335836c728d324152/7b4acb843fd4b0b335836c728d324152_0001_5_scene-X-bowl-X_mid.obj'

""" labelName2InstanceId, classId2ClassName=get_label_name_to_global_id(file_path) """
""" points,instances = get_points_instances_from_mesh(file_path, labelName2InstanceId)

print(points.shape, instances.shape) """
#import numpy; print(numpy.__version__)
import torch
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.enabled)