import argparse
from models.densenet import DenseNet121
from models.vgg import VGG
from models.wideresnet import Wide_ResNet
import torch
from svc.datasets.loader import *
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
import matplotlib.pyplot as plt
import time
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to model weights.", type=str, default="")
parser.add_argument("-i", "--dataset", type=str, default='cifar', help="Which is the inlier set.")
parser.add_argument("-o", "--outliers", type=str, default='tiny-imagenet', help="Which is the inlier set.")
parser.add_argument("-s", '--supervisor', type=str, default='odin', help="Which supervisor method are u running.")
parser.add_argument("-t", "--subset", type=int, default=10000, help="Wanna run a subset of the dataset?")
transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

tiny_imagenet_transform = Compose([Resize(size=(32, 32)), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

from svc.eval.metricplot import MetricPlot






def experiment_setup(model_path, inlier_name, outlier_name, supervisor_name):
    if 'densenet' in model_path.lower(): net = DenseNet121()
    if 'vgg' in model_path.lower(): net = VGG('VGG16')
    if 'wrn28' in model_path.lower(): net = Wide_ResNet(28, 10, 0.3, 10)
    if 'wrn40' in model_path.lower(): net = Wide_ResNet(40, 10, 0.3, 10)

    if "cifar" in inlier_name.lower(): inlier_set = load_cifar10(batch_size=1, shuffle=False, transform=transform, train=False)
    if "tiny-imagenet" in outlier_name.lower(): outlier_set = load_tiny_imagenet(batch_size=1, shuffle=False, transform=tiny_imagenet_transform)

    if "odin" in supervisor_name.lower(): from svc.ood.ODIN import ODIN as supervisor
    if "softmax" in supervisor_name.lower(): from svc.ood.baseline import SoftmaxProb as supervisor

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    weights_path = '../pytorch-cifar/checkpoint/' + model_path + '.t7'
    checkpoint = torch.load(weights_path, map_location='cpu')

    net.load_state_dict(checkpoint['net'])
    net.to(device)
    net.eval()
    print('Loaded {}'.format(model_path))

    experiment_name = "./results/gridsearch_" + model_path + "_" + supervisor_name + '.txt'


    best_auroc = 0.0

    for temp in [50, 100, 200, 500, 700, 1000, 1500, 2000]:
        for grad in np.linspace(0, 0.004, 21):
            t0 = time.time()
            sv = supervisor(model=net, device=device, gradient_amplitude=grad, temperature=temp)
            print("running with grad: {}, and temp: {}".format(grad, temp))

            scores, predictions, labels = experiment(inlier_set, sv, device)
            sc_out, pre_out, _ = experiment(outlier_set, sv, device)

            anomaly_scores = np.concatenate([scores, sc_out])
            predictions = np.concatenate([predictions, pre_out])
            true_labels = np.concatenate([labels, -1*np.ones(len(sc_out,))])

            mp = MetricPlot(anomaly_scores, predictions, true_labels)
            metrics = mp.IST_Metrics()
            auroc = metrics['auroc']
            fpr95 = metrics['TPR95']

            with open(experiment_name, 'a+') as f:
                f.write("{}, {}, {}, {}\n".format(grad, temp, auroc, fpr95))
            print("{}, {}, {}, {}. Completed in {}\n".format(grad, temp, auroc, fpr95, time.time()-t0))




def experiment(dataloader, sv, device):
    scores = []
    outputs = []
    t0 = time.time()
    data_size = len(dataloader.dataset)
    labels = []
    for j, data in enumerate(dataloader):
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        [labels.append(x) for x in targets.detach().cpu()]
        score, predictions = sv.anomaly_score_list(inputs, return_predictions=True)
        [scores.append(x) for x in score]
        [outputs.append(x) for x in predictions]

        if (j + 1) % subset == 0:
            break

    return np.array(scores), np.array(outputs), np.array(labels)

if __name__ == "__main__":
    global subset
    args = parser.parse_args()
    subset = args.subset

    if not os.path.isdir('./results'):
        os.mkdir('./results')

    experiment_setup(args.model, args.dataset, args.outliers, args.supervisor)

