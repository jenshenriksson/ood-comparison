import scipy.spatial.distance as spd
import libmr
import numpy as np
import torch
import time
'''

https://github.com/abhijitbendale/OSDN

https://vast.uccs.edu/~abendale/papers/0348.pdf 

'''
class OpenMax:
    def __init__(self, model, device, tail_length=20, alpha_rank=10, training_set=None):
        self.weibull_models = None
        self.mean_activations = None
        self.eucos_dist = None
        self.tail_length = tail_length
        self.alpha_rank = alpha_rank
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.dataset = training_set
        self.update_model()

    def update_model(self):
        print('Training OpenMax meta recognition function')
        t0 = time.time()
        activations = []
        predictions = []
        labels = []
        for j, (images, targets) in enumerate(self.dataset):
            images, targets = images.to(self.device), targets.to(self.device)
            outputs = self.model(images)
            _, pred = outputs.max(1)

            activations.append(outputs[0].detach().cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())
            labels.append(targets.detach().cpu().numpy())

            if j > 10000:
                break


        activations = np.array(activations)
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()

        self.calculate_mean_activations(activations, predictions, labels)
        self.weibull_models = weibull_tailfitting(self.eucos_dist, self.mean_activations, self.tail_length)
        print('Fitted OpenMax Meta-recognition in {} seconds'.format(time.time()-t0))

    def calculate_mean_activations(self, activations, predictions, labels):
        self.mean_activations, self.eucos_dist = compute_mav_distances(activations, predictions, labels)


    def anomaly_score(self, image, return_predictions):
        activation = self.model(image)
        activation_cpu = activation.detach().cpu()
        max_prob, argmax_pos = activation_cpu.max(dim=1)

        openmax_probab = recalibrate_scores(self.weibull_models, activation_cpu[0], alpharank=self.alpha_rank)


        if np.argmax(openmax_probab, 0) == 10:  # Most probable class is outlier. Should be rejected.
            score = 1.0
        else:
            score = 1 - max_prob

        if return_predictions is True:
            return score, argmax_pos


    def anomaly_score_list(self, inputs, return_predictions=False):
        scores = []
        predictions = []
        for image in inputs:
            if return_predictions is True:
                score, pred = self.anomaly_score(image.unsqueeze_(0), return_predictions)
                scores.append(score)
                predictions.append(pred)
            else:
                scores.append(self.anomaly_score(image.unsqueeze_(0)))

        if return_predictions is True:
            return scores, predictions

        return scores

    def find_tail_length(self, outputs, predictions, labels, inlier_set, outlier_set):
        from utils.metrics import MetricPlots
        self.calculate_mean_activations(outputs, predictions, labels)

        tail_test_lengths = np.arange(10, 100, 1)
        auc_scores = np.zeros(len(tail_test_lengths))

        for i, tail_length in enumerate(tail_test_lengths):
            print("Fitting {}".format(i))
            self.weibull_models = weibull_tailfitting(self.eucos_dist, self.mean_activations, 20)
            anomalyScoreInlier = self.anomaly_score(torch.stack(inlier_set).cpu().numpy())
            anomalyScoreOutlier = self.anomaly_score(torch.stack(outlier_set).cpu().numpy())
            anomalyScores = np.concatenate([np.array(anomalyScoreInlier), np.array(anomalyScoreOutlier)])

            labels = np.concatenate(np.zeros(len(anomalyScoreInlier)), -1 * np.ones(len(anomalyScoreInlier)))
            mp = MetricPlots(anomalyScores, 0, labels, -1)
            auc_scores[i] = mp.auroc

        sorted_tails = [tail for auc, tail in sorted(zip(auc_scores, tail_test_lengths))]
        sorted_auc = [auc for auc, tail in sorted(zip(auc_scores, tail_test_lengths))]
        print("Best tails are: {}: {:.3f}, {}: {:.3f}, {}: {:.3f}".format(sorted_tails[-1], sorted_auc[-1],
                                                                          sorted_tails[-1], sorted_auc[-1],
                                                                          sorted_tails[-1], sorted_auc[-1]))

        return sorted_tails


def compute_open_max_probability(openmax_known_score, openmax_unknown_score):
    """
    Compute the OpenMax probability.
    :param openmax_known_score: Weibull scores for known labels.
    :param openmax_unknown_score: Weibull scores for unknown unknowns.
    :return: OpenMax probability.
    """

    prob_closed, prob_open, scores = [], [], []

    # Compute denominator for closet set + open set normalization.
    # Sum up the class scores.
    for category in range(10):
        scores += [np.exp(openmax_known_score[category])]
    total_denominator = np.sum(np.exp(openmax_known_score)) + np.exp(openmax_unknown_score)

    # Scores for image belonging to either closed or open set.
    prob_closed = np.array([scores / total_denominator])
    prob_open = np.array([np.exp(openmax_unknown_score) / total_denominator])

    probs = np.append(prob_closed.tolist(), prob_open)

    assert len(probs) == 11
    return probs


def recalibrate_scores(weibull_model, img_layer_act, alpharank=10):
    """
    Computes the OpenMax probabilities of an input image.
    :param weibull_model: pre-computed Weibull model.
                          Dictionary with [class_labels]['euclidean distances', 'mean_vec', 'weibull_model']
    :param img_layer_act: activations in penultimate layer.
    :param alpharank: number of top classes to revise/check.
    :return: OpenMax probabilities of image.
    """

    num_labels = 10
    # Sort index of activations from highest to lowest.
    ranked_list = np.argsort(img_layer_act)
    ranked_list = np.ravel(ranked_list)
    ranked_list = ranked_list[::-1]

    # Obtain alpha weights for highest -> lowest activations.
    alpha_weights = [((alpharank + 1) - i) / float(alpharank) for i in range(1, alpharank + 1)]
    ranked_alpha = np.zeros(num_labels)
    for i in range(0, len(alpha_weights)):
        ranked_alpha[ranked_list[i]] = alpha_weights[i]

    # Calculate OpenMax probabilities
    openmax_penultimate, openmax_penultimate_unknown = [], []
    for categoryid in range(num_labels):
        label_weibull = weibull_model[str(categoryid)]['weibull_model']  # Obtain the corresponding Weibull model.
        label_mav = weibull_model[str(categoryid)]['mean_vec']  # Obtain MAV for specific class.
        img_dist = spd.euclidean(label_mav, img_layer_act) / 200. + spd.cosine(label_mav, img_layer_act)

        weibull_score = label_weibull.w_score(img_dist)

        modified_layer_act = img_layer_act[categoryid] * (
                    1 - weibull_score * ranked_alpha[categoryid])  # Revise av.
        openmax_penultimate += [modified_layer_act]  # Append revised av. to a total list.
        openmax_penultimate_unknown += [
            img_layer_act[categoryid] - modified_layer_act]  # A.v. 'unknown unknowns'.

    openmax_closedset_logit = np.asarray(openmax_penultimate)
    openmax_openset_logit = np.sum(openmax_penultimate_unknown)

    # Transform the recalibrated penultimate layer scores for the image into OpenMax probability.
    openmax_probab = compute_open_max_probability(openmax_closedset_logit, openmax_openset_logit)

    return openmax_probab


def compute_mav_distances(activations, predictions, true_labels):
    """
    Calculates the mean activation vector (MAV) for each class and the distance to the mav for each vector.
    :param activations: logits for each image.
    :param predictions: predicted label for each image.
    :param true_labels: true label for each image.
    :return: MAV and euclidean-cosine distance to each vector.
    """

    correct_activations = list()
    mean_activations = list()
    eucos_dist = list()

    for cl in range(10):
        # Find correctly predicted samples and store activation vectors.
        i = (true_labels == predictions)
        i = i & (predictions == cl)
        act = activations[i, :]
        correct_activations.append(act)

        # Compute MAV for class.
        mean_act = np.mean(act, axis=0)
        mean_activations.append(mean_act)

        # Compute all, for this class, correctly classified images' distance to the MAV.
        eucos_dist_cl = np.zeros(len(act))
        for col in range(len(act)):
            eucos_dist_cl[col] = spd.euclidean(mean_act, act[col, :]) / 200 + spd.cosine(mean_act, act[col, :])
        eucos_dist.append(eucos_dist_cl)
    return mean_activations, eucos_dist


def weibull_tailfitting(eucos_dist, mean_activations, taillength=8):
    """
    Fits a Weibull model of the logit vectors farthest from the MAV.
    :param eucos_dist: the euclidean-cosine distance from the MAV.
    :param mean_activations: mean activation vector (MAV).
    :param taillength:
    :return: weibull model.
    """

    weibull_model = {}
    for cl in range(10):
        weibull_model[str(cl)] = {}
        weibull_model[str(cl)]['eucos_distances'] = eucos_dist[cl]
        weibull_model[str(cl)]['mean_vec'] = mean_activations[cl]
        weibull_model[str(cl)]['weibull_model'] = []
        mr = libmr.MR(verbose=True)
        tailtofit = sorted(eucos_dist[cl])[-int(taillength):]
        mr.fit_high(tailtofit, len(tailtofit))
        weibull_model[str(cl)]['weibull_model'] = mr

    return weibull_model


