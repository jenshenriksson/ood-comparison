import numpy as np
import time
from svc.eval.metricplot import MetricPlot
from svc.datasets.loader import load_tiny_imagenet, load_cifar10, load_cifar100

def demo():
    outlier_label = -1
    num_samples = 1000
    num_labels = 10

    # Randomize two distributions; one inlier and one outlier set.
    inlier_anomaly_score = np.random.normal(2, 2, num_samples)
    outlier_anomaly_score = np.random.normal(5, 2, num_samples)

    # Randomize labels for the distributions.
    inlier_predictions = np.random.randint(0, num_labels, num_samples)
    outlier_predictions = np.random.randint(0, num_labels, num_samples)
    # outlier_predictions = np.random.randint(-1, 1, num_samples)

    # Create true labels for the distributions. Assume 100% accuracy on the inlier.
    inlier_true_labels = inlier_predictions
    outlier_true_labels = outlier_label * np.ones(num_samples)

    # Concatenate the distributions.
    anomaly_score = np.concatenate([inlier_anomaly_score, outlier_anomaly_score])
    predictions = np.concatenate([inlier_predictions, outlier_predictions])
    true_labels = np.concatenate([inlier_true_labels, outlier_true_labels])
    # Test a MetricPlot object.
    mp = MetricPlot(anomaly_score, predictions, true_labels)
    return mp


def print_ai_metrics():
    mp = demo()
    print(mp.AI_Testing_metrics())


def visualize_example_data():
    mp = demo()
    mp.plot_all()


def download_datasets(dataset_name='cifar'):

    cifar10 = load_cifar10()
    print("CIFAR10 works")
    cifar100 = load_cifar100()
    print("CIFAR100 works")
    tiny = load_tiny_imagenet()
    print("Tiny ImageNet works")


if __name__=="__main__":
    # download_datasets()
    visualize_example_data()
    print_ai_metrics()
