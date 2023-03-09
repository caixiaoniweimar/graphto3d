import numpy as np
points=np.array([[1,2,3],[4,5,6],[1,2,3],[7,8,9],[10,11,12],[12,13,14],[1,2,3]])
instances = np.array([1,2,6,3,4,5,5])
def remove_duplicates(points, instances):
    # concatenate points and instance IDs along axis 1
    data = np.concatenate((points, instances.reshape(-1, 1)), axis=1)
    print("1", data)

    # get unique rows of data
    unique_data = np.unique(data, axis=0)
    print("2", unique_data)

    # extract unique points and instance IDs from unique_data
    unique_points = unique_data[:, :-1]
    unique_instances = unique_data[:, -1].astype(int)
    print(unique_points)

    return unique_points, unique_instances

def remove_duplicates2(points, instances):
    # Convert points_list to a numpy array
    points = np.array(points)

    # Get the unique rows and their corresponding indices
    unique_rows, indices = np.unique(points, axis=0, return_index=True)
    print(unique_rows, indices)

    # Update instances_list to remove instances corresponding to duplicate points
    instances = np.array(instances)[indices]
    print(instances, instances.shape, type(instances))

    # Update points_list to contain only unique points
    points_list = np.array([list(row) for row in unique_rows])

    # Convert instances_list to a numpy array
    instances = np.array(instances)
    return points_list, instances

points, instances = remove_duplicates2(points, instances)
print(points, instances)
