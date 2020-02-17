import argparse
import torch
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
import time
import numpy as np
import os

from models.densenet import DenseNet121
from models.vgg import VGG
from models.wideresnet import Wide_ResNet

from svc.datasets.loader import *
from svc.eval.metricplot import MetricPlot

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="path to model weights.", type=str, default="")
parser.add_argument("-i", "--dataset", type=str, default='cifar', help="Which is the inlier set.")
parser.add_argument("-o", "--outliers", type=str, default='tiny-imagenet', help="Which is the inlier set.")
parser.add_argument("-s", '--supervisor', type=str, default='odin', help="Which supervisor method are u running.")
parser.add_argument("-t", "--subset", type=int, default=10000, help="Wanna run a subset of the dataset?")

# CIFAR 10 normalization factors
transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

# Same transform but with Resize to CIFAR size.
outlier_transform = Compose([Resize(size=(32, 32)), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])


def experiment_setup(model_path, inlier_name, outlier_name, supervisor_name):
    # Pre-define net architecture
    if 'densenet' in model_path.lower(): net = DenseNet121()
    elif 'vgg' in model_path.lower(): net = VGG('VGG16')
    elif 'wrn28' in model_path.lower(): net = Wide_ResNet(28, 10, 0.3, 10)
    elif 'wrn40' in model_path.lower(): net = Wide_ResNet(40, 10, 0.3, 10)

    # Assign inlier and outlier sets.
    if "cifar" in inlier_name.lower():
        inlier_set = load_cifar10(batch_size=1, shuffle=False, transform=transform, train=False)
        inlier_training_set = load_cifar10(batch_size=1, shuffle=False, transform=transform, train=True)

    if "tiny-imagenet" in outlier_name.lower(): outlier_set = load_tiny_imagenet(batch_size=1, shuffle=False, transform=outlier_transform)
    elif "fake" in outlier_name.lower(): outlier_set = load_fake_data(batch_size=1, shuffle=False, transform=outlier_transform)
    elif "svhn" in outlier_name.lower(): outlier_set = load_svhn(batch_size=1, shuffle=False, transform=outlier_transform)

    # Import the supervisor
    if "odin" in supervisor_name.lower(): from svc.ood.ODIN import ODIN as supervisor
    elif "softmax" in supervisor_name.lower(): from svc.ood.baseline import SoftmaxProb as supervisor
    elif "openmax" in supervisor_name.lower(): from svc.ood.openmax import OpenMax as supervisor

    # Additional for the experiment
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    experiment_name = 'results/' + '_'.join([supervisor_name, inlier_name, outlier_name, model_path]) + '.txt'
    epoch_names = ['', '_best'] + ['-'+str(i*10) for i in range(1, 31)]

    # Experiments
    for name in epoch_names:
        t0 = time.time()
        load_path = '../pytorch-cifar/checkpoint/' + model_path + name + '.t7'
        if not os.path.isfile(load_path):
            continue
        checkpoint = torch.load(load_path, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        net.to(device)
        net.eval()
        if "openmax" in supervisor_name.lower():
            sv = supervisor(model=net, device=device, training_set=inlier_training_set)
        else:
            sv = supervisor(model=net, device=device)

        scores, predictions, labels = experiment(inlier_set, sv, device)
        sc_out, pre_out, _ = experiment(outlier_set, sv, device)

        anomaly_scores = np.concatenate([scores, sc_out])
        predictions = np.concatenate([predictions, pre_out])
        true_labels = np.concatenate([labels, -1*np.ones(len(sc_out,))])

        mp = MetricPlot(anomaly_scores, predictions, true_labels)
        metrics = mp.IST_Metrics()
        auroc = metrics['auroc']
        fpr95 = metrics['TPR95']
        cbpl = metrics['CBPL']
        carl = metrics['CARL']
        rarl = metrics['RARL']

        # Save the results to txt.
        with open(experiment_name, 'a+') as f:
            f.write("{}, {}, {}, {}, {}, {}\n".format(checkpoint['epoch'], auroc, fpr95, cbpl, carl, rarl))
        print("{}, {}, {}, {}, {}, {}. this took: {}\n".format(checkpoint['epoch'], auroc, fpr95, cbpl, carl, rarl, time.time()-t0))


def experiment(dataloader, sv, device):
    scores = []
    outputs = []
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
    experiment_setup(args.model, args.dataset, args.outliers, args.supervisor)
