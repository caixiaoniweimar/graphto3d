from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import copy
import dataset.util as util
from tqdm import tqdm
import json
from helpers.psutil import FreeMemLinux
from helpers.util import normalize_box_params, denormalize_box_params, get_rotation
import random
import pickle


class RIODatasetSceneGraph(data.Dataset):
    def __init__(self, root, root_raw,
                 label_file, npoints=2500, class_choice=None,
                 split='train', data_augmentation=True, shuffle_objs=False,
                 pass_scan_id=False, use_points=True,
                 use_scene_rels=False, data_len=None,
                 with_changes=True, vae_baseline=False,
                 scale_func='diag', eval=False, eval_type='none',
                 atlas=None, path2atlas=None, with_feats=False,
                 seed=True, use_splits=False, large=False,
                 use_rio27=False, recompute_feats=False, use_canonical=False,
                 crop_floor=False, center_scene_to_floor=False):

        # options currently not used in the experiments
        # for partial scenes (use_splits), it crops the floor around the objects that are part of that scene fraction
        self.crop_floor = crop_floor
        self.center_scene_to_floor = center_scene_to_floor

        self.seed = seed
        self.with_feats = with_feats
        self.atlas = atlas
        self.path2atlas = path2atlas
        self.large = large
        self.recompute_feats = recompute_feats

        self.use_canonical = use_canonical

        if eval and seed:
            np.random.seed(47)
            torch.manual_seed(47)
            random.seed(47)

        self.scale_func = scale_func
        self.with_changes = with_changes
        self.npoints = npoints
        self.use_points = use_points
        self.root = root
        # list of class categories
        self.catfile = os.path.join(self.root, 'classes.txt')
        self.cat = {}
        self.scans = []
        self.data_augmentation = data_augmentation
        self.data_len = data_len
        self.vae_baseline = vae_baseline
        self.use_scene_rels = use_scene_rels

        self.fm = FreeMemLinux('GB')
        self.vocab = {}
        with open(os.path.join(self.root, 'classes.txt'), "r") as f:
            self.vocab['object_idx_to_name'] = f.readlines()
        with open(os.path.join(self.root, 'relationships.txt'), "r") as f:
            self.vocab['pred_idx_to_name'] = f.readlines()

        splitfile = os.path.join(self.root, '{}.txt'.format(split))

        filelist = open(splitfile, "r").read().splitlines()
        self.filelist = [file.rstrip() for file in filelist] #? 包含对应type的所有scene_names
        # list of relationship categories
        self.relationships = self.read_relationships(os.path.join(self.root, 'relationships.txt'))#?所有的relationship名称

        # uses scene sections of up to 9 objects (from 3DSSG) if true, and full scenes otherwise
        self.use_splits = use_splits
        if split == 'train_scenes': # training set
            splits_fname = 'relationships_train'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_train.json')

        else: # validation set
            splits_fname = 'relationships_validation'
            self.rel_json_file = os.path.join(self.root, '{}.json'.format(splits_fname))
            self.box_json_file = os.path.join(self.root, 'obj_boxes_validation.json')

        self.relationship_json, self.objs_json, self.tight_boxes_json = \
                self.read_relationship_json(self.rel_json_file, self.box_json_file)

        self.label_file = label_file

        self.padding = 0.2
        self.eval = eval

        self.pass_scan_id = pass_scan_id

        self.shuffle_objs = shuffle_objs

        self.root_raw = root_raw
        if self.root_raw == '':
            self.root_raw = os.path.join(self.root, "raw")

        # option to map object classes to a smaller class set from 3RScan
        # not used in the current paper results
        self.use_rio27 = use_rio27

        with open(self.catfile, 'r') as f:
            for line in f:
                category = line.rstrip()
                self.cat[category] = category
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
            raise ValueError("class_choice should be None!")

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))

        # we had to discard some underrepresented classes for the shape generation
        # either not part of shapenet, or limited and low quality samples in 3rscan
        if not large: #large=False进入
            points_classes = ['fork', 'knife', 'tablespoon','teaspoon', 'plate', 'bowl', 'cup', 'teapot', 'pitcher', 'can', 'box','obstacle', 'support_table']

        points_classes_idx = []
        for pc in points_classes:
            if class_choice is not None:
                if pc in self.classes:
                    points_classes_idx.append(self.classes[pc])
                else:
                    points_classes_idx.append(0)
                raise ValueError("class_choice should be None!")
            else:
                points_classes_idx.append(self.classes[pc])

        self.point_classes_idx = points_classes_idx + [0]
        self.sorted_cat_list = sorted(self.cat)
        self.files = {}
        self.eval_type = eval_type

        # check if all shape features exist. If not they get generated here (once)
        if with_feats:
            print('Checking for missing feats. This can be slow the first time.\nThis process needs to be only run once!')
            for index in tqdm(range(len(self))):
                self.__getitem__(index)
            self.recompute_feats = False

    def read_relationship_json(self, json_file, box_json_file):
        """ Reads from json files the relationship labels, objects and bounding boxes

        :param json_file: file that stores the objects and relationships
        :param box_json_file: file that stores the oriented 3D bounding box parameters
        :return: three dicts, relationships, objects and boxes
        """
        rel = {}
        objs = {}
        tight_boxes = {}

        with open(box_json_file, "r") as read_file:
            box_data = json.load(read_file)

        with open(json_file, "r") as read_file:
            data = json.load(read_file)
            for scene in data['scenes']:

                relationships = []
                for realationship in scene["relationships"]:
                    realationship[2] -= 1 #! id为6, 在txt中为5
                    relationships.append(realationship)

                # for every scene in rel json, we append the scene id
                rel[scene["scene_id"]] = relationships
                self.scans.append(scene["scene_id"])

                objects = {}
                boxes = {}
                for k, v in scene["objects"].items():
                    objects[int(k)] = v
                    try:
                        boxes[int(k)] = {}
                        boxes[int(k)]['param6'] = box_data[scene["scene_id"]][k]["param6"]
                        if self.use_canonical:#! 不进入
                            if "direction" in box_data[scene["scene_id"]][k].keys():
                                boxes[int(k)]['direction'] = box_data[scene["scene_id"]][k]["direction"]
                            else:
                                boxes[int(k)]['direction'] = 0
                            raise ValueError("use_canonical should be False")
                    except:
                        # probably box was not saved because there were 0 points in the instance!
                        continue
                objs[scene["scene_id"]] = objects
                tight_boxes[scene["scene_id"]] = boxes
        return rel, objs, tight_boxes

    def read_relationships(self, read_file):
        """load list of relationship labels

        :param read_file: path of relationship list txt file
        """
        relationships = []
        with open(read_file, 'r') as f:
            for line in f:
                relationship = line.rstrip().lower()
                relationships.append(relationship)
        return relationships

    def norm_tensor(self, p, params7=None, scale=False, center=True, rotation=False, scale_func='diag'):
#(obj_pointset, denormalize_box_params(tight_boxes[i]),scale=True, rotation=self.use_canonical, scale_func=self.scale_func)
        """ Given a set of points of an object and (optionally) a oriented 3D box, normalize points

        :param p: tensor of 3D pointset
        :param params7: bounding box parameters [W, L, H, Cx, Cy, Cy, Z]
        :param scale: boolean, if true apply scaling to pointset p according to scale_func
        :param center: boolean, if true normalize the center of points to 0,0,0
        :param rotation: boolean, if true rotate points based on the box rotation in param7
        :param scale_func: string specifying the function used for scaling. 'diag' normalizes the diagonal to length 1.
        'whl' sets each dimension to range [-1,1].
        :return: the normalized tensor of 3D pointset
        """
        if center:
            if params7 is None:
                # this is center of mass
                mean = torch.mean(p, dim=0)
            else:
                # get center from box center if available
                mean = torch.from_numpy(params7[3:6].astype("float32"))
            p -= mean.unsqueeze(0)
        if rotation and params7 is not None:
            p = (torch.from_numpy(get_rotation(-params7[-1], degree=False).astype("float32")) @ p.T).T
        if scale and params7 is not None:
            # first if needed rotate to canonical rotation
            # apply scaling
            # if needed rotate back
            if not rotation and len(params7)>6:#将点集 p 绕着 $z$ 轴逆时针旋转了 -params7[-1] 度
                p = (torch.from_numpy(get_rotation(-params7[-1], degree=False).astype("float32")) @ p.T).T
                raise ValueError("len(params7) should only be 6 without degree")
            if scale_func == 'diag':
                # OPTION 1: normalize diagonal = 1
                norm2 = np.linalg.norm(params7[:3].astype("float32"))
                p /= norm2
            elif scale_func == 'whl':
                # OPTION 2: normalize each axis by H, W, L
                norm2 = torch.from_numpy(params7[:3].astype("float32")).reshape(1, 3)
                min_p = p.min(0)[0]
                p = ((p - min_p) / norm2) * 2. - 1.  # between -1 and 1 in all directions
            elif scale_func == 'whl_after':
                norm2 = p.max(0)[0] - p.min(0)[0]
                min_p = p.min(0)[0]
                p = ((p - min_p) / norm2) * 2. - 1.  # between -1 and 1 in all directions
            else:
                raise NotImplementedError
            if not rotation and len(params7)>6: #将点集 p 绕着 $z$ 轴顺时针旋转了 params7[-1] 度
                p = (torch.from_numpy(get_rotation(params7[-1], degree=False).astype("float32")) @ p.T).T
                raise ValueError("len(params7) should only be 6 without degree")
        return p

    def __getitem__(self, index):
        scene_id = self.scans[index]
        #print(f"Current working on {scene_id}")
        #root_raw: ..../test_pipeline_dataset/raw/{table_id}/{scene_id}.obj
        file = os.path.join(self.root_raw, scene_id.split('_')[0], scene_id+self.label_file)
        if not os.path.exists(file):
            raise FileNotFoundError(f"Cannot find {scene_id}.obj file. Path: {file}")

        # instance2label, e.g. {1: 'floor', 2: 'wall', 3: 'picture', 4: 'picture'}
        labelName2InstanceId, instance2label = util.get_label_name_to_global_id(file)
        selected_instances = list(self.objs_json[scene_id].keys())
        keys = list(instance2label.keys())

        if self.shuffle_objs:
            random.shuffle(keys)
        feats_in = None
        # If true, expected paths to saved atlasnet features will be set here
        if self.with_feats and self.path2atlas is not None:
            _, atlasname = os.path.split(self.path2atlas)
            atlasname = atlasname.split('.')[0]

            if not self.large:#! 进入 因为large=False
                feats_path = os.path.join(self.root_raw, scene_id.split('_')[0],
                                          '{}_small_{}_{}.pkl'.format(atlasname,
                                                                      'splits' if self.use_splits else 'merged',
                                                                      scene_id))
            else:
                feats_path = os.path.join(self.root_raw, scene_id.split('_')[0],
                                          '{}_large_{}_{}.pkl'.format(atlasname,
                                                                      'splits' if self.use_splits else 'merged',
                                                                      scene_id))
            if self.recompute_feats:
                feats_path += 'tmp'
        # Load points if with features but features cannot be found or are forced to be recomputed
        # Loads points if use_points is set to true
        if (self.with_feats and (not os.path.exists(feats_path) or self.recompute_feats)) or self.use_points:
            if file in self.files: # Caching
                (points, instances) = self.files[file]
            else:
                points, instances = util.get_points_instances_from_mesh(file, labelName2InstanceId)

                if self.fm.user_free > 5:
                    self.files[file] = (points, instances)

        instance2mask = {}
        instance2mask[0] = 0

        cat = []
        tight_boxes = []

        counter = 0

        instances_order = []
        selected_shapes = []

        for key in keys: #{'1', 2}
            # get objects from the selected list of classes of 3dssg
            scene_instance_id = key
            scene_instance_class = instance2label[key]
            scene_class_id = -1
            if scene_instance_class in self.classes:
                scene_class_id = self.classes[scene_instance_class]
            
            if scene_class_id != -1 and key in selected_instances:
                instance2mask[scene_instance_id] = counter + 1
                counter += 1
            else:#! scene_class_id == -1 -> 不存在self.classes中
                print(key, selected_instances)
                instance2mask[scene_instance_id] = 0
                raise ValueError("scene_class_id should not be -1.")
            # mask to cat:
            if (scene_class_id >= 0) and (scene_instance_id > 0) and (key in selected_instances):
                if self.use_canonical:#! 不进入
                    direction = self.tight_boxes_json[scene_id][key]['direction']
                    if direction in [-1, 0, 6]:
                        # skip invalid point clouds with ambiguous direction annotation
                        selected_shapes.append(False)
                    else:
                        selected_shapes.append(True)
                cat.append(scene_class_id)
                bbox = self.tight_boxes_json[scene_id][key]['param6'].copy()

                if self.use_canonical:#! 不进入
                    if direction > 1 and direction < 5:
                        # update direction-less angle with direction data (shifts it by 90 degree
                        # for every added direction value
                        bbox[6] += (direction - 1) * np.deg2rad(90)
                        if direction == 2 or direction == 4:
                            temp = bbox[0]
                            bbox[0] = bbox[1]
                            bbox[1] = temp
                    # for other options, do not change the box
                instances_order.append(key)
                if not self.vae_baseline:#!network_type==shared!=sln则进入
                    bbox = normalize_box_params(bbox)
                tight_boxes.append(bbox)
        if self.with_feats:
            # If precomputed features exist, we simply load them
            if os.path.exists(feats_path):
                feats_dic = pickle.load(open(feats_path, 'rb'))

                feats_in = feats_dic['feats']
                feats_order = np.asarray(feats_dic['instance_order'])
                ordered_feats = []
                for inst in instances_order:
                    feats_in_instance = inst == feats_order
                    ordered_feats.append(feats_in[:-1][feats_in_instance])
                ordered_feats.append(np.zeros([1, feats_in.shape[1]]))
                feats_in = list(np.concatenate(ordered_feats, axis=0))
        # Sampling of points from object if they are loaded
        if (self.with_feats and (not os.path.exists(feats_path) or feats_in is None)) or self.use_points:
            masks = np.array(list(map(lambda l: instance2mask[l] if l in instance2mask.keys() else 0, instances)),
                             dtype=np.int32)
            num_pointsets = len(cat) + int(self.use_scene_rels)  # add zeros for the scene node
            obj_points = torch.zeros([num_pointsets, self.npoints, 3])

            for i in range(len(cat)):
                obj_pointset = points[np.where(masks == i + 1)[0], :]

                if len(obj_pointset) >= self.npoints:
                    choice = np.random.choice(len(obj_pointset), self.npoints, replace=False)
                else:
                    choice = np.arange(len(obj_pointset))
                    # use repetitions to fill some more points
                    choice2 = np.random.choice(len(obj_pointset), self.npoints - choice.shape[0], replace=True)
                    choice = np.concatenate([choice, choice2], 0)
                    random.shuffle(choice)

                obj_pointset = obj_pointset[choice, :]
                obj_pointset = torch.from_numpy(obj_pointset.astype(np.float32))

                if not self.vae_baseline:#! network_type=shared进入
                    obj_pointset = self.norm_tensor(obj_pointset, denormalize_box_params(tight_boxes[i]),
                                           scale=True, rotation=self.use_canonical, scale_func=self.scale_func)
                else:
                    obj_pointset = self.norm_tensor(obj_pointset, np.asarray(tight_boxes[i]),
                                                    scale=True, rotation=self.use_canonical, scale_func=self.scale_func)
            obj_points[i] = obj_pointset
        else:
            obj_points = None

        triples = []
        rel_json = self.relationship_json[scene_id]

        for r in rel_json: # create relationship triplets from data
            if r[0] in instance2mask.keys() and r[1] in instance2mask.keys():
                subject = instance2mask[r[0]] - 1
                object = instance2mask[r[1]] - 1
                predicate = r[2] + 1
                if subject >= 0 and object >= 0:
                    triples.append([subject, predicate, object])
            else:
                continue     
        if self.use_scene_rels:
            # add _scene_ object and _in_scene_ connections
            scene_idx = len(cat)
            for i, ob in enumerate(cat):
                triples.append([i, 0, scene_idx])
            cat.append(0)
            # dummy scene box
            tight_boxes.append([-1, -1, -1, -1, -1, -1])
        output = {}
        if self.use_points:#! X
            output['scene'] = points

        # if features are requested but the files don't exist, we run all loaded pointclouds through atlasnet
        # to compute them and then save them for future usage
        if self.with_feats and (not os.path.exists(feats_path) or feats_in is None) and self.atlas is not None:
            pf = torch.from_numpy(np.array(list(obj_points.numpy()), dtype=np.float32)).float().cuda().transpose(1,2)
            with torch.no_grad():
                feats = self.atlas.encoder(pf).detach().cpu().numpy()

            feats_out = {}
            feats_out['feats'] = feats
            feats_out['instance_order'] = instances_order
            feats_in = list(feats)

            assert self.path2atlas is not None
            path = os.path.join(feats_path)
            if self.recompute_feats:
                path = path[:-3]

            pickle.dump(feats_out, open(path, 'wb'))
        # prepare outputs
        output['encoder'] = {}
        output['encoder']['objs'] = cat
        output['encoder']['triples'] = triples
        output['encoder']['boxes'] = tight_boxes
        if self.use_points:
            output['encoder']['points'] = list(obj_points.numpy())

        if self.with_feats:
            output['encoder']['feats'] = feats_in

        output['manipulate'] = {}
        if not self.with_changes:
            output['manipulate']['type'] = 'none'
            output['decoder'] = copy.deepcopy(output['encoder'])
        else:
            if not self.eval:#! 进入 因为self.eval=False
                output['manipulate']['type'] = 'none'
                output['decoder'] = copy.deepcopy(output['encoder'])
            else:#!X
                output['manipulate']['type'] = self.eval_type
                output['decoder'] = copy.deepcopy(output['encoder'])
                if output['manipulate']['type'] == 'addition':
                    node_id = self.remove_node_and_relationship(output['encoder'])
                    if node_id >= 0:
                        output['manipulate']['added'] = node_id
                    else:
                        return -1
                elif output['manipulate']['type'] == 'relationship':
                    rel, pair, suc = self.modify_relship(output['decoder'], interpretable=True)
                    if suc:
                        output['manipulate']['relship'] = (rel, pair)
                    else:
                        return -1
        # torchify
        output['encoder']['objs'] = torch.from_numpy(np.array(output['encoder']['objs'], dtype=np.int64))
        output['encoder']['triples'] = torch.from_numpy(np.array(output['encoder']['triples'], dtype=np.int64))
        output['encoder']['boxes'] = torch.from_numpy(np.array(output['encoder']['boxes'], dtype=np.float32))
        if self.use_points:#!X
            output['encoder']['points'] = torch.from_numpy(np.array(output['encoder']['points'], dtype=np.float32))
        if self.with_feats:#!进入
            output['encoder']['feats'] = torch.from_numpy(np.array(output['encoder']['feats'], dtype=np.float32))

        output['decoder']['objs'] = torch.from_numpy(np.array(output['decoder']['objs'], dtype=np.int64))
        output['decoder']['triples'] = torch.from_numpy(np.array(output['decoder']['triples'], dtype=np.int64))
        output['decoder']['boxes'] = torch.from_numpy(np.array(output['decoder']['boxes'], dtype=np.float32))
        if self.use_points:#!X
            output['decoder']['points'] = torch.from_numpy(np.array(output['decoder']['points'], dtype=np.float32))
        if self.with_feats:#!进入
            output['decoder']['feats'] = torch.from_numpy(np.array(output['decoder']['feats'], dtype=np.float32))

        output['scene_id'] = scene_id
        output['instance_id'] = instances_order

        return output


    def remove_node_and_relationship(self, graph):
        """ Automatic random removal of certain nodes at training time to enable training with changes. In that case
        also the connecting relationships of that node are removed

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :return: index of the removed node
        """

        node_id = -1
        # dont remove layout components, like floor. those are essential
        if not self.use_rio27:
            excluded = [27, 58, 155]
        else:
            excluded = [1, 2, 15]
        trials = 0
        while node_id < 0 or graph['objs'][node_id] in excluded:
            if trials > 100:
                return -1
            trials += 1
            node_id = np.random.randint(len(graph['objs']) - 1)

        graph['objs'].pop(node_id)
        if self.use_points:
            graph['points'].pop(node_id)
        if self.with_feats:
            graph['feats'].pop(node_id)
        graph['boxes'].pop(node_id)

        to_rm = []
        for x in graph['triples']:
            sub, pred, obj = x
            if sub == node_id or obj == node_id:
                to_rm.append(x)

        while len(to_rm) > 0:
            graph['triples'].remove(to_rm.pop(0))

        for i in range(len(graph['triples'])):
            if graph['triples'][i][0] > node_id:
                graph['triples'][i][0] -= 1

            if graph['triples'][i][2] > node_id:
                graph['triples'][i][2] -= 1

        return node_id

    def modify_relship(self, graph, interpretable=False):
        """ Change a relationship type in a graph

        :param graph: dict containing objects, features, boxes, points and relationship triplets
        :param interpretable: boolean, if true choose a subset of easy to interpret relations for the changes
        :return: index of changed triplet, a tuple of affected subject & object, and a boolean indicating if change happened
        """

        # rels 26 -> 0
        '''26 hanging in' '25 lying in' '24 cover' '23 build in' '22 standing in' '21 belonging to'
         '20 part of' '19 leaning against' '18 connected to' '17 hanging on' '16 lying on' '15 standing on'
         '14 attached to' '13 same as' '12 same symmetry as' '11 lower than' '10 higher than' '9 smaller than'
         '8 bigger than' '7 inside' '6 close by' '5 behind' '4 front' '3 right' '2 left' '1 supported by'
         '0: none'''
        # subset of edge labels that are spatially interpretable (evaluatable via geometric contraints)
        interpretable_rels = [2, 3, 4, 5, 8, 9, 10, 11]
        inside_rel = [7, 22, 23, 24, 25, 26]
        # arm chair, basked, bathtub, bidet, bookshelf, box, chair,  commode, cupboard, trash bin, kettle, shelf,
        # wardrobe, sink
        inside_obj = [1, 7, 9, 13, 20, 22, 28, 37, 43, 67, 74, 119, 129, 156]

        hanging_rel = [14, 17]
        # backpack, bag, blinds, clock, curtain
        hanging_sub_allowed = [2, 3, 16, 30, 44]
        # wall
        hanging_obj_allowed = [155]

        did_change = False
        trials = 0
        excluded = [27]
        eval_excluded = [27, 58, 155]

        while not did_change and trials < 1000:
            idx = np.random.randint(len(graph['triples']))
            sub, pred, obj = graph['triples'][idx]
            trials += 1

            if pred == 0:
                continue
            if graph['objs'][obj] in excluded or graph['objs'][sub] in excluded:
                continue
            # sub_cl, obj_cl = graph['objs'][sub], graph['objs'][obj]
            if interpretable:
                if graph['objs'][obj] in eval_excluded or graph['objs'][sub] in eval_excluded: # don't use the floor
                    continue
                new_pred = interpretable_rels[np.random.randint(1, len(interpretable_rels))]
            else:
                new_pred = np.random.randint(1, 27)

            graph['triples'][idx][1] = new_pred
            did_change = True
        return idx, (sub, obj), did_change

    def __len__(self):
        if self.data_len is not None:
            return self.data_len
        else:
            return len(self.scans)


def collate_fn_vaegan(batch, use_points=False):
    """
    Collate function to be used when wrapping a RIODatasetSceneGraph in a
    DataLoader. Returns a dictionary
    """

    out = {}

    out['scene_points'] = []
    out['scene_id'] = []
    out['instance_id'] = []

    out['missing_nodes'] = []
    out['missing_nodes_decoder'] = []
    out['manipulated_nodes'] = []
    global_node_id = 0
    global_dec_id = 0

    for i in range(len(batch)):
        if batch[i] == -1:
            return -1
        # notice only works with single batches
        out['scene_id'].append(batch[i]['scene_id'])
        out['instance_id'].append(batch[i]['instance_id'])

        if batch[i]['manipulate']['type'] == 'addition':
            out['missing_nodes'].append(global_node_id + batch[i]['manipulate']['added'])
            out['missing_nodes_decoder'].append(global_dec_id + batch[i]['manipulate']['added'])
        elif batch[i]['manipulate']['type'] == 'relationship':
            rel, (sub, obj) = batch[i]['manipulate']['relship']
            out['manipulated_nodes'].append(global_dec_id + sub)
            out['manipulated_nodes'].append(global_dec_id + obj)

        if 'scene' in batch[i]:
            out['scene_points'].append(batch[i]['scene'])

        global_node_id += len(batch[i]['encoder']['objs'])
        global_dec_id += len(batch[i]['decoder']['objs'])

    for key in ['encoder', 'decoder']: 
        all_objs, all_boxes, all_triples = [], [], []
        all_obj_to_scene, all_triple_to_scene = [], []
        all_points = []
        all_feats = []

        obj_offset = 0

        for i in range(len(batch)):
            if batch[i] == -1:
                print('this should not happen')
                continue
            (objs, triples, boxes) = batch[i][key]['objs'], batch[i][key]['triples'], batch[i][key]['boxes']

            if 'points' in batch[i][key]:
                all_points.append(batch[i][key]['points'])
            if 'feats' in batch[i][key]:
                all_feats.append(batch[i][key]['feats'])

            num_objs, num_triples = objs.size(0), triples.size(0)

            all_objs.append(objs)
            all_boxes.append(boxes)

            if triples.dim() > 1:
                triples = triples.clone()
                triples[:, 0] += obj_offset
                triples[:, 2] += obj_offset

                all_triples.append(triples)
                all_triple_to_scene.append(torch.LongTensor(num_triples).fill_(i))

            all_obj_to_scene.append(torch.LongTensor(num_objs).fill_(i))

            obj_offset += num_objs

        all_objs = torch.cat(all_objs)
        all_boxes = torch.cat(all_boxes)

        all_obj_to_scene = torch.cat(all_obj_to_scene)

        if len(all_triples) > 0:
            all_triples = torch.cat(all_triples)
            all_triple_to_scene = torch.cat(all_triple_to_scene)
        else:
            return -1

        outputs = {'objs': all_objs,
                   'tripltes': all_triples,
                   'boxes': all_boxes,
                   'obj_to_scene': all_obj_to_scene,
                   'tiple_to_scene': all_triple_to_scene}

        if len(all_points) > 0:
            all_points = torch.cat(all_points)
            outputs['points'] = all_points

        if len(all_feats) > 0:
            all_feats = torch.cat(all_feats)
            outputs['feats'] = all_feats
        out[key] = outputs

    return out


def collate_fn_vaegan_points(batch):
    """ Wrapper of the function collate_fn_vaegan to make it also return points
    """
    return collate_fn_vaegan(batch, use_points=True)
