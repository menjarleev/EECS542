import cv2
import numpy as np
import json
import os
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import RoIAlign
import pickle
import torch
import torch.nn as nn

from collections import OrderedDict


# transfer caffe model to pytorch which will match the layer name
def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights

def make_layers(block, no_relu_layers):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                 padding=v[2])
            layers.append((layer_name, layer))
        else:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if layer_name not in no_relu_layers:
                layers.append(('relu_'+layer_name, nn.ReLU(inplace=True)))

    return nn.Sequential(OrderedDict(layers))

class bodypose_model(nn.Module):
    def __init__(self):
        super(bodypose_model, self).__init__()

        # these layers have no relu layer
        no_relu_layers = ['conv5_5_CPM_L1', 'conv5_5_CPM_L2', 'Mconv7_stage2_L1', \
                          'Mconv7_stage2_L2', 'Mconv7_stage3_L1', 'Mconv7_stage3_L2', \
                          'Mconv7_stage4_L1', 'Mconv7_stage4_L2', 'Mconv7_stage5_L1', \
                          'Mconv7_stage5_L2', 'Mconv7_stage6_L1', 'Mconv7_stage6_L1']
        blocks = {}
        block0 = OrderedDict([
            ('conv1_1', [3, 64, 3, 1, 1]),
            ('conv1_2', [64, 64, 3, 1, 1]),
            ('pool1_stage1', [2, 2, 0]),
            ('conv2_1', [64, 128, 3, 1, 1]),
            ('conv2_2', [128, 128, 3, 1, 1]),
            ('pool2_stage1', [2, 2, 0]),
            ('conv3_1', [128, 256, 3, 1, 1]),
            ('conv3_2', [256, 256, 3, 1, 1]),
            ('conv3_3', [256, 256, 3, 1, 1]),
            ('conv3_4', [256, 256, 3, 1, 1]),
            ('pool3_stage1', [2, 2, 0]),
            ('conv4_1', [256, 512, 3, 1, 1]),
            ('conv4_2', [512, 512, 3, 1, 1]),
            ('conv4_3_CPM', [512, 256, 3, 1, 1]),
            ('conv4_4_CPM', [256, 128, 3, 1, 1])
        ])


        # Stage 1
        block1_1 = OrderedDict([
            ('conv5_1_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L1', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L1', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L1', [512, 38, 1, 1, 0])
        ])

        block1_2 = OrderedDict([
            ('conv5_1_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_2_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_3_CPM_L2', [128, 128, 3, 1, 1]),
            ('conv5_4_CPM_L2', [128, 512, 1, 1, 0]),
            ('conv5_5_CPM_L2', [512, 19, 1, 1, 0])
        ])
        blocks['block1_1'] = block1_1
        blocks['block1_2'] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        for i in range(2, 7):
            blocks['block%d_1' % i] = OrderedDict([
                ('Mconv1_stage%d_L1' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L1' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L1' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L1' % i, [128, 38, 1, 1, 0])
            ])

            blocks['block%d_2' % i] = OrderedDict([
                ('Mconv1_stage%d_L2' % i, [185, 128, 7, 1, 3]),
                ('Mconv2_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv3_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv4_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv5_stage%d_L2' % i, [128, 128, 7, 1, 3]),
                ('Mconv6_stage%d_L2' % i, [128, 128, 1, 1, 0]),
                ('Mconv7_stage%d_L2' % i, [128, 19, 1, 1, 0])
            ])

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        self.model1_1 = blocks['block1_1']
        self.model2_1 = blocks['block2_1']
        self.model3_1 = blocks['block3_1']
        self.model4_1 = blocks['block4_1']
        self.model5_1 = blocks['block5_1']
        self.model6_1 = blocks['block6_1']

        self.model1_2 = blocks['block1_2']
        self.model2_2 = blocks['block2_2']
        self.model3_2 = blocks['block3_2']
        self.model4_2 = blocks['block4_2']
        self.model5_2 = blocks['block5_2']
        self.model6_2 = blocks['block6_2']


    def forward(self, x):

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)

        return out6

class Body(object):
    def __init__(self, model_path, gpu_id):
        self.model = bodypose_model()
        device = torch.device('cuda: %d' % gpu_id)
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        map_location = {'cuda: 0' : 'cuda: %d' % gpu_id}
        model_dict = transfer(self.model, torch.load(model_path, map_location=map_location))
        self.model.load_state_dict(model_dict)
        self.model = DistributedDataParallel(self.model, device_ids=[gpu_id])
        self.model.eval()

    def __call__(self, oriImg, metadata, gpu_id):
        # scale_search = [0.5, 1.0, 1.5, 2.0]
        scale_search = [1.0]
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        # heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 19))
        # paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 38))

            # scale = multiplier[m]
            # imageToTest = cv2.resize(oriImg, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            # imageToTest_padded, pad = util.padRightDownCorner(imageToTest, stride, padValue)
        device = torch.device('cuda: %d' % gpu_id)
        im = np.transpose(np.float32(oriImg[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)

        data = torch.from_numpy(im).float().to(device)
        bbox = [torch.tensor(metadata['boxes'])[:, :-1].to(device)]
        # data = data.permute([2, 0, 1]).unsqueeze(0).float()
        feat = self.model(data)
        roi_align = RoIAlign(6, 1/8, -1, True).to(device)
        roi = roi_align(feat, bbox)
        roi = roi.reshape(roi.shape[0], -1)
        res_dict = {'pose_features': roi.detach().cpu().numpy()}
        return res_dict



def get_pose_feat(rank, world_size, vcr_images_dir, records, feature_extractor, save_path):
    part_size = len(records) // world_size
    sub_records = records[rank * part_size:min((rank + 1) * part_size, len(records))]
    print('proc [{}] will handle {} records'.format(rank, len(sub_records)))
    for idx, record in enumerate(sub_records):
        metadata_fn = record['metadata_fn']
        img_fn = record['img_fn']
        id = img_fn[img_fn.rfind('/') + 1: img_fn.rfind('.')]
        file = os.path.join(save_path, id + '.pkl')
        if not os.path.exists(file):
            print('proc [{}] now process {}'.format(rank, id))
            metadata_file = os.path.join(vcr_images_dir, metadata_fn)
            with open(metadata_file) as f:
                metadata = json.load(f)
            img_file = os.path.join(vcr_images_dir, img_fn)
            oriImg = cv2.imread(img_file)
            res = feature_extractor(oriImg, metadata, gpu_id=rank)
            with open(file, 'wb') as f:
                pickle.dump(res, f)



def dist_start(world_size, rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'
    init_process_group('nccl', rank=rank, world_size=world_size)

def dist_end():
    destroy_process_group()

def extract_data(rank, world_size, splits, vcg_dir, vcg_images_dir, save_path):
    dist_start(world_size, rank)
    feature_extractor = Body('./model/body_pose_model.pth', gpu_id=rank)
    for s, split in enumerate(splits):
        split_filename = '{}_annots.json'.format(split)
        input_file = os.path.join(vcg_dir, split_filename)
        with open(input_file) as f:
            records = json.load(f)
            get_pose_feat(rank, world_size, vcg_images_dir, records, feature_extractor, save_path)
    dist_end()


if __name__ == "__main__":
    vcg_dir = '/home/ubuntu/data/annots'
    vcg_images_dir = '/home/ubuntu/data/vcr1images'
    for split in ['train', 'val', 'test']:
        split_filename = '{}_annots.json'.format(split)
        assert os.path.exists(os.path.join(vcg_dir, split_filename))
    splits = ['train', 'val', 'test']
    save_path = '/home/ubuntu/data/pose'
    os.makedirs(save_path, exist_ok=True)
    world_size = 8
    processes = []
    for i in range(world_size):
        proc = mp.Process(target=extract_data, args=(i, world_size, splits, vcg_dir, vcg_images_dir, save_path))
        proc.start()
        processes.append(proc)
    for p in processes:
        p.join()




