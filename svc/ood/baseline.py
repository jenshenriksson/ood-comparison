import numpy as np
import torch
import torch.nn.functional as F


""""""
class SoftmaxProb:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def anomaly_score(self, image, return_predictions):
        output = self.model(image)
        softmax = F.softmax(output, dim=1)
        softmax_cpu = softmax.detach().cpu()
        if return_predictions:
            return 1 - softmax_cpu.max(), softmax_cpu.argmax()

        return 1 - softmax_cpu.max()

    def anomaly_score_list(self, inputs, return_predictions):
        outputs = self.model(inputs)
        softmax = F.softmax(outputs, dim=1)
        softmax_cpu = softmax.detach().cpu()
        max_prob, argmax_pos = softmax_cpu.max(dim=1)

        if return_predictions:
            return 1 - max_prob, argmax_pos

        return 1 - max_prob
