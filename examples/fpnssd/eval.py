import argparse
import re

import path
from PIL import Image
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
from torchcv.models.fpnssd import FPNSSD512
from torchcv.models.fpnssd import FPNSSDBoxCoder


parser = argparse.ArgumentParser(description='PyTorch FPNSSD Eval')
parser.add_argument('--root', default=None, type=str, help='Dataset root.')
args = parser.parse_args()
args.root = path.Path(args.root)

print('Loading model..')
net = FPNSSD512(num_classes=21)
state_dict = {}
for k, v in torch.load('examples/fpnssd/checkpoint/ckpt.pth', map_location='cpu')['net'].items():
    state_dict[re.sub('module.', '', k)] = v
net.load_state_dict(state_dict)
net.cuda()
net.eval()

print('Preparing dataset..')
img_size = 512
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels

dataset = ListDataset(root=args.root/'voc_all_images',
                      list_file='torchcv/datasets/voc/voc07_test.txt',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
box_coder = FPNSSDBoxCoder()

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(d)

def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(inputs.cuda())
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cpu().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cpu().data,
            score_thresh=0.01)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print(voc_eval(
        pred_boxes, pred_labels, pred_scores,
        gt_boxes, gt_labels, gt_difficults,
        iou_thresh=0.5, use_07_metric=True))

with torch.no_grad():
    eval(net, dataset)
