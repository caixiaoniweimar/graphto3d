from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset.dataset import RIODatasetSceneGraph, collate_fn_vaegan_points
from model.VAE import VAE
from model.atlasnet import AE_AtlasNet
from model.discriminators import BoxDiscriminator, ShapeAuxillary
from model.losses import bce_loss
from helpers.util import bool_flag

from model.losses import calculate_model_losses

import torch.nn.functional as F
import json

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser()
# standard hyperparameters, batch size, learning rate, etc
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--nepoch', type=int, default=101, help='number of epochs to train for')

# paths and filenames
parser.add_argument('--outf', type=str, default='checkpoint', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', required=False, type=str, default="/media/caixiaoni/xiaonicai-u/test_pipeline_dataset", help="dataset path")
parser.add_argument('--dataset_raw', type=str, default='/media/caixiaoni/xiaonicai-u/test_pipeline_dataset/raw', help="raw dataset path")
parser.add_argument('--label_file', required=False, type=str, default='.obj', help="label file name")
parser.add_argument('--logf', default='logs', help='folder to save tensorboard logs')
parser.add_argument('--exp', default='./experiments/graphto3d_test_nepoch_301_points_10000', help='experiment name')
parser.add_argument('--path2atlas', required=False, default="./experiments/atlasnet/model_70.pth", type=str)

# GCN parameters
parser.add_argument('--residual', type=bool_flag, default=True, help="residual in GCN")
parser.add_argument('--pooling', type=str, default='avg', help="pooling method in GCN")

# dataset related
parser.add_argument('--large', default=False, type=bool_flag, help='large set of shape class labels')
parser.add_argument('--use_splits', default=False, type=bool_flag, help='不需要splitSet to true if you want to use splitted training data')
parser.add_argument('--use_scene_rels', type=bool_flag, default=True, help="connect all nodes to a root scene node")
parser.add_argument('--with_points', type=bool_flag, default=False, help="with_feats为True即可if false and with_feats is false, only predicts layout."
                                                                         "If true always also loads pointsets. Notice that this is much "
                                                                         "slower than only loading the save features. Therefore, "
                                                                         "the recommended setting is with_points=False and with_feats=True.")
parser.add_argument('--with_feats', type=bool_flag, default=True, help="若为True但不存在,需要生成特征,若存在,不用点而用特征if true reads latent point features instead of pointsets."
                                                                       "If not existing, they get generated at the beginning.")
parser.add_argument('--shuffle_objs', type=bool_flag, default=True, help="shuffle objs of a scene")
parser.add_argument('--num_points', type=int, default=1024, help='number of points for each object')
parser.add_argument('--rio27', default=False, type=bool_flag)
parser.add_argument('--use_canonical', default=False, type=bool_flag)#!不需要有direction
parser.add_argument('--with_angles', default=False, type=bool_flag)#!不需要有angle
parser.add_argument('--num_box_params', default=6, type=int)
parser.add_argument('--crop_floor', default=False, type=bool_flag)

# training and architecture related
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--overfiting_debug', type=bool_flag, default=False)
parser.add_argument('--weight_D_box', default=0.1, type=float, help="Box Discriminator")
parser.add_argument('--with_changes', default=True, type=bool_flag)
parser.add_argument('--with_shape_disc', default=True, type=bool_flag)
parser.add_argument('--with_manipulator', default=True, type=bool_flag)

parser.add_argument('--replace_latent', default=True, type=bool_flag)
parser.add_argument('--network_type', default='shared', choices=['dis', 'sln', 'mlp', 'shared'], type=str)

args = parser.parse_args()
print(args)


def train():
    """ Train the network based on the provided argparse parameters
    """
    args.manualSeed = random.randint(1, 10000)  # optionally fix seed 7494
    print("Random Seed: ", args.manualSeed)

    print(torch.__version__)

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    # prepare pretrained AtlasNet model, later used to convert pointsets to a shape feature
    saved_atlasnet_model = torch.load(args.path2atlas)
    point_ae = AE_AtlasNet(num_points=1024, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_atlasnet_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()
    point_ae.eval()

    # instantiate scene graph dataset for training
    dataset = RIODatasetSceneGraph(
            root=args.dataset,
            root_raw=args.dataset_raw,
            label_file=args.label_file,
            npoints=args.num_points,
            path2atlas=args.path2atlas,
            split='train_scenes',
            shuffle_objs=(args.shuffle_objs and not args.overfiting_debug),
            use_points=args.with_points,
            use_scene_rels=args.use_scene_rels,
            with_changes=args.with_changes,
            vae_baseline=args.network_type == 'sln',
            with_feats=args.with_feats,
            large=args.large,
            atlas=point_ae,
            seed=False,
            use_splits=args.use_splits,
            use_rio27=args.rio27,
            use_canonical=args.use_canonical,
            crop_floor=args.crop_floor,
            center_scene_to_floor=args.crop_floor,
            recompute_feats=False)

    collate_fn = collate_fn_vaegan_points
    # instantiate data loader from dataset
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batchSize,
            collate_fn=collate_fn,
            shuffle=(not args.overfiting_debug),
            num_workers=int(args.workers))

    # number of object classes and relationship classes
    num_classes = len(dataset.classes)
    num_relationships = len(dataset.relationships)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass
    # instantiate the model
    model = VAE(type=args.network_type, vocab=dataset.vocab, replace_latent=args.replace_latent,
                with_changes=args.with_changes, residual=args.residual, gconv_pooling=args.pooling,
                with_angles=args.with_angles, num_box_params=args.num_box_params)
    if torch.cuda.is_available():
        model = model.cuda()
    # instantiate a relationship discriminator that considers the boxes and the semantic labels
    # if the loss weight is larger than zero
    # also create an optimizer for it
    if args.weight_D_box > 0:
        boxD = BoxDiscriminator(6, num_relationships, num_classes)#! box_dim, rel_dim, obj_dim
        optimizerDbox = optim.Adam(filter(lambda p: p.requires_grad, boxD.parameters()), lr=args.lr, betas=(0.9, 0.999))
        boxD.cuda()
        boxD = boxD.train()

    # instantiate auxiliary discriminator for shape and a respective optimizer
    #! 辅助神经网络模块，它的作用是对输入的形状编码进行判断，看其是否合理，同时为给定的形状编码预测一个类别标签。这个类继承了nn.Module，它包含两个主要部分：一个分类器和一个判别器。
    shapeClassifier = ShapeAuxillary(128, len(dataset.cat))
    shapeClassifier = shapeClassifier.cuda()
    shapeClassifier.train()
    optimizerShapeAux = optim.Adam(filter(lambda p: p.requires_grad, shapeClassifier.parameters()), lr=args.lr, betas=(0.9, 0.999))

    # initialize tensorboard writer
    writer = SummaryWriter(args.exp + "/" + args.logf)

    # optimizer for model
    params = filter(lambda p: p.requires_grad,list(model.parameters()) )
    optimizer = optim.Adam(params, lr=args.lr)

    print("---- Model and Dataset built ----")

    if not os.path.exists(args.exp + "/" + args.outf):
        os.makedirs(args.exp + "/" + args.outf)

    # save parameters so that we can read them later on evaluation
    with open(os.path.join(args.exp, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    print("Saving all parameters under:")
    print(os.path.join(args.exp, 'args.json'))

    optimizer.step()
    torch.autograd.set_detect_anomaly(True)
    counter = 0

    print("---- Starting training loop! ----")
    for epoch in range(0, args.nepoch):
        print('Epoch: {}/{}'.format(epoch, args.nepoch))

        for i, data in enumerate(dataloader, 0):
            # skip invalid data
            if data == -1:
                continue

            try:
                enc_objs, enc_triples, enc_tight_boxes, enc_objs_to_scene, enc_triples_to_scene = data['encoder']['objs'],\
                            data['encoder']['tripltes'], data['encoder']['boxes'], data['encoder']['obj_to_scene'], data['encoder']['tiple_to_scene']
                #! enc_objs: 所有的class_id
                #! enc_triples: 所有的边
                #! enc_tight_boxes: 所有的bbox -> param6
                #! enc_objs_to_scene: 每个对象所属的场景索引
                #! enc_triples_to_scene: 每个关系三元组所属的场景索引

                if args.with_feats:
                    encoded_enc_points = data['encoder']['feats']
                    encoded_enc_points = encoded_enc_points.cuda()
                elif args.with_points:
                    enc_points = data['encoder']['points']
                    enc_points = enc_points.cuda()

                dec_objs, dec_triples, dec_tight_boxes, dec_objs_to_scene, dec_triples_to_scene = data['decoder']['objs'],\
                            data['decoder']['tripltes'], data['decoder']['boxes'], data['decoder']['obj_to_scene'], data['decoder']['tiple_to_scene']

                if 'feats' in data['decoder']:
                    encoded_dec_points = data['decoder']['feats']
                    encoded_dec_points = encoded_dec_points.cuda()
                else:
                    if 'points' in data['decoder']:
                        dec_points = data['decoder']['points']
                        dec_points = dec_points.cuda()

                # changed nodes
                missing_nodes = data['missing_nodes']
                manipulated_nodes = data['manipulated_nodes']

            except Exception as e:
                print('Exception', str(e))
                continue

            enc_objs, enc_triples, enc_tight_boxes = enc_objs.cuda(), enc_triples.cuda(), enc_tight_boxes.cuda()
            dec_objs, dec_triples, dec_tight_boxes = dec_objs.cuda(), dec_triples.cuda(), dec_tight_boxes.cuda()

            if args.with_points:
                enc_points, dec_points = enc_points.cuda(), dec_points.cuda()

            # avoid batches with insufficient number of instances with valid shape classes
            #print(f"227 - {enc_objs}, {dec_objs}")
            mask = [ob in dataset.point_classes_idx for ob in dec_objs]
            if sum(mask) <= 1:
                continue
            #print(f"231 - {mask}")

            optimizer.zero_grad()
            optimizerShapeAux.zero_grad()

            model = model.train()

            if args.weight_D_box > 0:
                optimizerDbox.zero_grad()

            # if we are reading pointsets and not precomputed shape features, convert them to features
            # otherwise do nothing, as we already have the features
            if not args.with_feats and args.with_points:
                with torch.no_grad():
                    encoded_enc_points = point_ae.encoder(enc_points.transpose(2,1).contiguous())
                    encoded_dec_points = point_ae.encoder(dec_points.transpose(2,1).contiguous())

            # set all scene (dummy) nodes point encodings to zero
            enc_scene_nodes = enc_objs == 0
            dec_scene_nodes = dec_objs == 0
            encoded_enc_points[enc_scene_nodes] = torch.zeros([torch.sum(enc_scene_nodes), encoded_enc_points.shape[1]]).float().cuda()
            encoded_dec_points[dec_scene_nodes] = torch.zeros([torch.sum(dec_scene_nodes), encoded_dec_points.shape[1]]).float().cuda()

            if args.num_box_params == 6:
                # no angle. this will be learned separately if with_angle is true
                enc_boxes = enc_tight_boxes[:, :6]
                dec_boxes = dec_tight_boxes[:, :6]
            else:
                raise NotImplementedError

            # limit the angle bin range from 0 to 24
            enc_angles = None#enc_tight_boxes[:, 6].long() - 1
            """ enc_angles = torch.where(enc_angles > 0, enc_angles, torch.zeros_like(enc_angles))
            enc_angles = torch.where(enc_angles < 24, enc_angles, torch.zeros_like(enc_angles)) """
            dec_angles = None#dec_tight_boxes[:, 6].long() - 1
            """ dec_angles = torch.where(dec_angles > 0, dec_angles, torch.zeros_like(dec_angles))
            dec_angles = torch.where(dec_angles < 24, dec_angles, torch.zeros_like(dec_angles)) """

            attributes = None

            boxGloss = 0
            loss_genShape = 0
            loss_genShapeFake = 0
            loss_shape_fake_g = 0

            if args.with_manipulator:#! 进入
                model_out = model.forward_mani(enc_objs, enc_triples, enc_boxes, enc_angles, encoded_enc_points, attributes, enc_objs_to_scene,
                                                   dec_objs, dec_triples, dec_boxes, dec_angles, encoded_dec_points, attributes, dec_objs_to_scene,
                                                   missing_nodes, manipulated_nodes)

                mu_box, logvar_box, mu_shape, logvar_shape, orig_gt_box, orig_gt_angle, orig_gt_shape, orig_box, orig_angle, orig_shape, \
                dec_man_enc_box_pred, dec_man_enc_angle_pred, dec_man_enc_shape_pred, keep = model_out
                assert(orig_gt_angle==None)
                assert(orig_angle==None)
                assert(dec_man_enc_angle_pred==None)

                #mu, logvar, mu, logvar, orig_gt_boxes, None, orig_gt_shapes, orig_boxes, None, orig_shapes, boxes, None, shapes, keep
            else:#!X
                model_out = model.forward_no_mani(dec_objs, dec_triples, dec_boxes, encoded_dec_points, angles=dec_angles,
                                      attributes=attributes)

                mu_box, logvar_box, mu_shape, logvar_shape, dec_man_enc_box_pred, dec_man_encd_angles_pred, \
                  dec_man_enc_shape_pred = model_out

                orig_gt_box = dec_boxes
                orig_box = dec_man_enc_box_pred

                orig_gt_shape = encoded_dec_points
                orig_shape = dec_man_enc_shape_pred

                orig_angle = dec_man_encd_angles_pred
                orig_gt_angle = dec_angles

                keep = []
                for i in range(len(dec_man_enc_box_pred)):
                    keep.append(1)
                keep = torch.from_numpy(np.asarray(keep).reshape(-1, 1)).float().cuda()

            if args.with_manipulator and args.with_shape_disc and dec_man_enc_shape_pred is not None:
                #! 计算伪造形状编码（来自解码器输出）的类别预测（shape_logits_fake_d）和真实性概率（probs_fake_d）。mask用于过滤数据，只保留需要计算的部分。detach()方法用于将该张量从计算图中分离，以避免在计算梯度时影响它。
                shape_logits_fake_d, probs_fake_d = shapeClassifier(dec_man_enc_shape_pred[mask].detach())#! 让假的成为假的
                shape_logits_fake_g, probs_fake_g = shapeClassifier(dec_man_enc_shape_pred[mask])#! 让假的成为真的, 不会将输入张量从计算图中分离。这将用于计算生成器的损失，因此需要保留梯度信息。
                shape_logits_real, probs_real = shapeClassifier(encoded_dec_points[mask].detach())#! 计算真实形状编码的类别预测（shape_logits_real）和真实性概率（probs_real）

                # auxiliary loss. can the discriminator predict the correct class for the generated shape?]#! 伪造形状编码能够预测正确的class?
                #!计算真实形状（shape_logits_real）与目标对象（dec_objs[mask]）之间的交叉熵损失。这个损失衡量了辅助分类器在真实形状数据上的性能。
                loss_shape_real = torch.nn.functional.cross_entropy(shape_logits_real, dec_objs[mask])
                #!生成的形状是从解码器输出的形状数据中获得的，但在计算损失时使用了辅助分类器的detach()版本，这意味着辅助分类器的梯度不会在生成器的梯度计算中使用。这个损失衡量了辅助分类器在生成形状数据上的性能。
                loss_shape_fake_d = torch.nn.functional.cross_entropy(shape_logits_fake_d, dec_objs[mask])
                #!这里使用的是辅助分类器没有detach()的版本，这意味着辅助分类器的梯度会在生成器的梯度计算中使用。这个损失衡量了生成器在生成形状数据上的性能，同时也影响着辅助分类器的更新。
                #!计算了生成器生成的形状（shape）编码在辅助分类器（ShapeAuxiliary）上的性能。在这个损失计算中，使用了未经detach()的辅助分类器输出（shape_logits_fake_g），这意味着辅助分类器的梯度将在生成器的梯度计算中使用。
                #!这个损失衡量了生成器在生成形状数据上的性能，同时也影响着辅助分类器的更新。
                #!具体来说，损失函数是生成形状编码（shape_logits_fake_g）和实际形状类别（dec_objs[mask]）之间的交叉熵损失。当生成器生成出更接近真实形状的编码时，这个损失值会变得更小。
                #! shape_fake_g 小 -> 所生成的形状编码 能够预测出正确的形状类别
                loss_shape_fake_g = torch.nn.functional.cross_entropy(shape_logits_fake_g, dec_objs[mask])
                # standard discriminator loss
                #! 使用二进制交叉熵损失（bce_loss）计算辅助分类器在生成形状（probs_fake_g）上的性能。目标是使辅助分类器预测生成的形状为真实的（标签为1）。
                #! 它衡量了辅助分类器判断生成的形状编码（probs_fake_g）为真实的（即类别为 1）的概率。当生成器生成出更接近真实形状的编码时，辅助分类器将更有可能将其判断为真实的，这个损失值会变得更小。
                loss_genShapeFake = bce_loss(probs_fake_g, torch.ones_like(probs_fake_g))
                #! 计算辅助分类器在真实形状（probs_real）上的二进制交叉熵损失。目标是使辅助分类器预测真实形状为真实的（标签为1）。
                loss_dShapereal = bce_loss(probs_real, torch.ones_like(probs_real))
                #! 计算辅助分类器在生成形状（probs_fake_d）上的二进制交叉熵损失。目标是使辅助分类器预测生成的形状为假的（标签为0）。
                loss_dShapefake = bce_loss(probs_fake_d, torch.zeros_like(probs_fake_d))

                loss_dShape = loss_dShapefake + loss_dShapereal + loss_shape_real + loss_shape_fake_d #! 真实->真实; 假的->假 鉴别真伪能力
                loss_genShape = loss_genShapeFake + loss_shape_fake_g#! 假的->真 造假能力
                #! 目前的情况是生成的形状能够预测出正确的形状类别, 但是这个生成的形状无法被判断为真实的
                loss_dShape.backward()
                optimizerShapeAux.step()

            #!calculate_model_losses => 计算重建损失（rec_loss）为预测值（pred）与目标值（target）之间的L1损失 + KL散度损失，用于度量预测分布与目标分布之间的差异。
            vae_loss_box, vae_losses_box = calculate_model_losses(args,
                                                                    orig_gt_box,
                                                                    orig_box,
                                                                    name='box', withangles=args.with_angles, angles_pred=orig_angle,
                                                                    mu=mu_box, logvar=logvar_box, angles=orig_gt_angle,
                                                                    KL_weight=0.1, writer=writer, counter=counter)
            if dec_man_enc_shape_pred is not None:
                vae_loss_shape, vae_losses_shape = calculate_model_losses(args,
                                                                        orig_gt_shape,
                                                                        orig_shape,
                                                                        name='shape', withangles=False,
                                                                        mu=mu_shape, logvar=logvar_shape,
                                                                        KL_weight=0.1, writer=writer, counter=counter)
            else:
                # set shape loss to 0 if we are only predicting layout
                vae_loss_shape, vae_losses_shape = 0, 0
                raise ValueError("dec_man_enc_shape_pred shouldn't be None")

            if args.with_manipulator and args.with_changes:
                oriented_gt_boxes = torch.cat([dec_boxes], dim=1)
                boxes_pred_in = keep * oriented_gt_boxes + (1-keep) * dec_man_enc_box_pred

                if args.weight_D_box == 0:#! X
                    # Generator loss
                    boxGloss = 0
                    # Discriminator loss
                    gamma = 0.1
                    boxDloss_real = 0
                    boxDloss_fake = 0
                    reg_loss = 0
                    raise ValueError(f"args.weight_D_box shouldn't be 0")
                else: #! 进入
                    #! logits：这是一个形状为(batch_size, 1)的张量，其中每个元素是一个介于0到1之间的值。这个值表示给定的对象对之间关系的可信度。接近1表示关系很可能是真实的（或合理的），接近0表示关系不太可能是真实的（或不合理的）。
                    #! 通过判别器计算由生成器生成的边界框（boxes_pred_in）的可信度。logits是一个张量，包含了每个对象对之间关系的判别器输出值。
                    logits, _ = boxD(dec_objs, dec_triples, boxes_pred_in, keep)#! objs, triples, boxes, keeps
                    #! 计算生成的边界框（boxes_pred_in）的假（不真实）输出值（logits_fake）和相应的梯度惩罚（reg_fake）。
                    logits_fake, reg_fake = boxD(dec_objs, dec_triples, boxes_pred_in.detach(), keep, with_grad=True,
                                               is_real=False)
                    #! 计算真实边界框（oriented_gt_boxes）的真实（合理）输出值（logits_real）和相应的梯度惩罚（reg_real）。
                    logits_real, reg_real = boxD(dec_objs, dec_triples, oriented_gt_boxes, with_grad=True, is_real=True)
                    # Generator loss
                    boxGloss = bce_loss(logits, torch.ones_like(logits))
                    # Discriminator loss
                    gamma = 0.1
                    #! 计算真实边界框的判别器损失，将真实输出值（logits_real）与全1张量进行比较。
                    boxDloss_real = bce_loss(logits_real, torch.ones_like(logits_real))
                    #! 计算生成边界框的判别器损失，将假输出值（logits_fake）与全0张量进行比较。
                    boxDloss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
                    # Regularization by gradient penalty
                    #! 计算梯度惩罚的损失，这是真实边界框的梯度惩罚（reg_real）和生成边界框的梯度惩罚（reg_fake）的平均值。
                    reg_loss = torch.mean(reg_real + reg_fake)
                    #! 计算生成器和判别器的损失，以便在训练过程中更新它们的权重。这有助于生成器学会生成更合理的边界框，而判别器学会更好地区分合理的边界框与不合理的边界框。

                # gradient penalty
                # disc_reg = discriminator_regularizer(logits_real, in_real, logits_fake, in_fake)
                #! (gamma/2.0) * reg_loss：梯度惩罚项，用于增加模型稳定性并防止梯度爆炸。gamma是一个超参数，它用于调整梯度惩罚项在总损失中的权重。
                #! boxDloss_fake：生成边界框的判别器损失，用于衡量判别器如何区分生成的边界框与真实边界框。
                #! boxDloss_real: 真实边界框的判别器损失，用于衡量判别器如何区分真实边界框与生成的边界框。

                #!如果loss_genShapeFake的值一直增大，这可能表明以下几个问题：

                #生成器性能下降：生成器可能无法生成足够逼真的形状，使辅助分类器难以将生成的形状判断为真实的。这可能是因为生成器没有得到足够的训练，或者生成器的训练过程中遇到了困难。

                #辅助分类器过度拟合：辅助分类器可能在训练过程中过度拟合了真实数据，导致其在区分生成的形状和真实形状时表现过于出色。这可能意味着生成器无法生成具有足够逼真性的形状来欺骗辅助分类器。

                #不稳定的训练过程：生成对抗网络（GAN）的训练过程通常是不稳定的，生成器和辅助分类器之间的竞争可能导致训练过程难以收敛。这可能表明需要调整学习率、优化器设置或网络架构等超参数。

                #要解决这些问题，可以尝试以下方法：

                #调整训练过程中的超参数，例如学习率、批次大小或优化器设置。

                #修改网络架构，例如增加或减少层数，更改激活函数等。

                #使用更先进的训练技巧，例如梯度惩罚、谱归一化或其他稳定训练过程的方法。

                #使用预训练模型进行迁移学习，以便从现有知识中获得启示。
                
                boxDloss = boxDloss_fake + boxDloss_real + (gamma/2.0) * reg_loss
                optimizerDbox.zero_grad()
                boxDloss.backward()
                # gradient clip
                # torch.nn.utils.clip_grad_norm_(boxD.parameters(), 5.0)
                optimizerDbox.step()

            loss = vae_loss_box + vae_loss_shape + 0.1 * loss_genShape
            if args.with_changes:
                   loss = loss + args.weight_D_box * boxGloss #+ b_loss

            # optimize
            loss.backward()

            # Cap the occasional super mutant gradient spikes
            # Do now a gradient step and plot the losses
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and p.requires_grad and torch.isnan(p.grad).any():
                        print('NaN grad in step {}.'.format(counter))
                        p.grad[torch.isnan(p.grad)] = 0
            optimizer.step()
            counter += 1

            if counter:#counter % 100 == 0:
                print("loss at {}: box {:.4f}\tshape {:.4f}\tdiscr RealFake {:.4f}\t discr Classifcation "
                      "{:.4f}".format(counter, vae_loss_box, vae_loss_shape, loss_genShapeFake,
                                                              loss_shape_fake_g))
            writer.add_scalar('Train Loss BBox', vae_loss_box, counter)
            writer.add_scalar('Train Loss Shape', vae_loss_shape, counter)
            #! loss_genShapeFake应该是一个较小的值。这意味着生成的形状在辅助分类器的判断下更接近真实的形状，辅助分类器更难以区分生成的形状和真实的形状。
            #! 在训练过程中，生成器的目标是生成更高质量的形状，使其更难以被辅助分类器识别为伪造的。            
            writer.add_scalar('Train Loss loss_genShapeFake', loss_genShapeFake, counter)
            #! loss_shape_fake_g应该是一个较小的值。这意味着生成器生成的形状具有更高的质量，以至于辅助分类器在预测这些形状的类别标签时表现良好。
            #! 当生成器生成逼真的形状时，辅助分类器可以更准确地预测这些形状的类别
            writer.add_scalar('Train Loss loss_shape_fake_g', loss_shape_fake_g, counter)

        if epoch % 100 == 0:
            model.save(args.exp, args.outf, epoch)

    writer.close()


def main():
    train()


if __name__ == "__main__": main()
