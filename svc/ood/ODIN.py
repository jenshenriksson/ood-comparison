import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


"""
https://github.com/facebookresearch/odin 

https://arxiv.org/abs/1706.02690 
"""
class ODIN:
    def __init__(self, model, device, gradient_amplitude=0.0008, temperature=50):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.gradient_amplitude = gradient_amplitude
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()


    def anomaly_score(self, image, return_predictions):

        # inputs = torch.FloatTensor([image]).to(self.device)
        inputs = image.to(self.device)
        inputs = Variable(inputs, requires_grad=True)

        # Scale the input with temperature.
        outputs = self.model(inputs)
        outputs = outputs / self.temperature

        outputs_cpu = outputs.data.cpu()
        outputs_cpu = outputs_cpu.numpy()
        max_index_temp = np.argmax(outputs_cpu)
        labels = Variable(torch.LongTensor([max_index_temp])).to(self.device)

        loss = self.criterion(outputs, labels)
        loss.backward()


        # Normalize the gradient to binary {0, 1}
        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)

        # Add small perturbation to the input image
        temp_inputs = torch.add(inputs.data, self.gradient_amplitude, gradient)
        outputs = self.model(Variable(temp_inputs))



        # Scale the output based on the temperature.
        outputs = outputs / self.temperature

        outputs_cpu = outputs.data.cpu().numpy()
        outputs_cpu = outputs_cpu[0]

        # Calculate the softmax of the image, and return the score of the most likely class.
        outputs_cpu = outputs_cpu - outputs_cpu.max()
        outputs_cpu = np.exp(outputs_cpu) / np.sum(np.exp(outputs_cpu))  # Create the SoftMax expression for the image
        score = 1 - np.max(outputs_cpu)

        if return_predictions is True:
            return score, np.argmax(outputs_cpu)
        return score

    def anomaly_score_list(self, inputs, return_predictions):
        scores = []
        predictions = []
        # labels_true = []
        for image in inputs:
            if return_predictions is True:
                score, pred = self.anomaly_score(image.unsqueeze_(0), return_predictions)
                scores.append(score)
                predictions.append(pred)
                # labels_true.append(label)
            else:
                scores.append(self.anomaly_score(image))

        if return_predictions is True:
            return np.array(scores), np.array(predictions)
        else:
            return np.array(scores)

