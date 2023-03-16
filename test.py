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
""" import torch
print(torch.backends.cudnn.version())
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.enabled) """

catfile = "/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/classes.txt"
class_choice = ['fork', 'knife', 'tablespoon','teaspoon', 'plate', 'bowl', 'cup', 'teapot', 'pitcher', 'can', 'box', 'support_table']
cat = {}
with open(catfile, 'r') as f:
    for line in f:
        category = line.rstrip()
        cat[category] = category
#{'_scene_': '_scene_', 'fork': 'fork', 'knife': 'knife', 'tablespoon': 'tablespoon', 'teaspoon': 'teaspoon', 'plate': 'plate', 'bowl': 'bowl', 'cup': 'cup', 'teapot': 'teapot', 'pitcher': 'pitcher', 'can': 'can', 'box': 'box', 'support_table': 'support_table'}
classes = dict(zip(cat, range(len(cat))))
#print(cat)
#print(classes)

if not class_choice is None: #剔除obstacle
    cat = {k: v for k, v in cat.items() if k in class_choice}
#{'fork': 'fork', 'knife': 'knife', 'tablespoon': 'tablespoon', 'teaspoon': 'teaspoon', 'plate': 'plate', 'bowl': 'bowl', 'cup': 'cup', 'teapot': 'teapot', 'pitcher': 'pitcher', 'can': 'can', 'box': 'box', 'support_table': 'support_table'}
    classes = {k:v for k,v in classes.items() if k in class_choice}
#{'fork': 1, 'knife': 2, 'tablespoon': 3, 'teaspoon': 4, 'plate': 5, 'bowl': 6, 'cup': 7, 'teapot': 8, 'pitcher': 9, 'can': 10, 'box': 11, 'support_table': 13}

""" print(f"after choice: {cat}")
print(f"after choice: {classes}")

points_classes = ['fork', 'knife', 'tablespoon','teaspoon', 'plate', 'bowl', 'cup', 'teapot', 'pitcher', 'can', 'box', 'support_table']
points_classes_idx = []
for pc in points_classes:
    if class_choice is not None:
        if pc in classes:
            points_classes_idx.append(classes[pc])
        else:
            points_classes_idx.append(0)
    else:
        points_classes_idx.append(classes[pc])

point_classes_idx = points_classes_idx + [0]       #! [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 0]
print(point_classes_idx)
point_classes_idx = list(classes.values()) +[0]
print(point_classes_idx)

sorted_cat_list = sorted(cat)
print(sorted_cat_list)
sorted_cat_list = list(cat.keys())
print(sorted_cat_list)

vocab = {}
import os 
with open(os.path.join("/media/caixiaoni/xiaonicai-u/test_pipeline_dataset", 'classes.txt'), "r") as f:
    vocab['object_idx_to_name'] = f.readlines() 
with open(os.path.join("/media/caixiaoni/xiaonicai-u/test_pipeline_dataset", 'relationships.txt'), "r") as f:
    vocab['pred_idx_to_name'] = f.readlines()
print(vocab)


instance2mask = {0: 0, 40: 1, 31: 2, 25: 3, 7: 4, 117: 5}
instances = np.array([40,31,25,7,7,7,7,117])
masks = np.array(list(map(lambda l: instance2mask[l] if l in instance2mask.keys() else 0, instances)),
                             dtype=np.int32)
print(masks)
cat = [40,31,25,7,11]
for i in range(len(cat)):
    print(np.where(masks == i + 1)) """

catfile = "/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/classes.txt"
with open(catfile, 'r') as f:
    for line in f:
        category = line.rstrip()
        if category != "obstacle":
            cat[category] = category

classes = dict(zip(sorted(cat), range(len(cat))))
print(classes)