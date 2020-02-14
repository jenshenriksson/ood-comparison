import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import seaborn as sns
sns.set()
# sns.set_style("darkgrid")

class MetricPlot:
	"""
	Creates an object that contains the true positive and false positive ratios needed to plot ROC-curve,
	PR-curve and anomaly distribution. Additionally, computes the complete risk-coverage for the given
	supervisor, by comparing prediction results to the true labels and risk as a function of coverage.

	Note that the supervisor defines outliers as Positives; i.e. a True Positive is an outlier correctly
	detected and removed. False positives

	Inputs:
		anomaly_score: <Vector, Mx1> The score how likely a sample is an outlier or not
		predictions: <Vector, Mx1> The predicted output class
		true_labels: <Vector, Mx1> The true label for the input sample. true_label == outlier_labels for outliers.

	outputs:
		MetricPlotObject. Contains all metrics defined below.

	"""
	def __init__(self, anomaly_score=None, predictions=None, true_labels=None, outlier_label=-1, bins=50):
		# Initial estimates
		self.anomaly_score = anomaly_score
		self.predictions = predictions
		self.true_labels = true_labels
		self.outlier_label = outlier_label
		self.bins = bins

		# Metrics
		self.auroc = None
		self.aupr = None
		self.thresholds = None
		self.risk = None
		self.coverage = None
		self.remaining_outliers = None
		self.tpr = None
		self.fpr = None
		self.precision = None
		self.recall = None
		self.metrics = {}

		# if values assigned during initialization: Update metrics
		if self.data_is_in_correct_format():
			self.update_metrics()

	def data_is_in_correct_format(self):
		# Check that data is of ND-Array format
		if not isinstance(self.anomaly_score, np.ndarray) or not isinstance(self.predictions, np.ndarray) or not isinstance(self.true_labels, np.ndarray):
			return False
		elif np.shape(self.anomaly_score) != np.shape(self.true_labels) and \
				np.shape(self.anomaly_score) != np.shape(self.predictions) and \
				np.shape(self.predictions) != np.shape(self.true_labels):
			return False
		else:
			return True

	def update_metrics(self):
		# Define the outlier as a true positives, i.e. asign to 1.
		labels = 1 - np.sign(self.true_labels + 1)

		# Compute FPR, TPR and corresponding threshold for the labels and the given anomaly score.
		self.fpr, self.tpr, self.thresholds = roc_curve(labels, self.anomaly_score)

		# Assign precision as correct if correctly detected outlier.
		precision_labels = [1 if (x == -1) else 0 for x in self.true_labels]

		# Compute precision, recall and corresponding thresholds fore the given anomaly score.
		self.precision, self.recall, _ = precision_recall_curve(precision_labels, self.anomaly_score)

		# Update ROC-metrics
		self.auroc = auc(self.fpr, self.tpr)
		self.aupr = auc(self.recall, self.precision)

		##
		## Compute Risk-Coverage metrics
		##

		bins = len(self.thresholds)
		self.risk = np.empty(bins)
		self.coverage = np.empty(bins)
		self.remaining_outliers = np.empty(bins)

		complete_set_length = len(self.anomaly_score)

		# Correct model predictions (Including outlier detection)
		correct_predictions = self.true_labels == self.predictions
		outliers = self.true_labels == self.outlier_label

		# Vary over thresholds to get risk as a function of coverage
		for i, threshold in enumerate(self.thresholds):
			# Find which samples are kept for a given threshold
			samples_covered = self.anomaly_score < threshold
			false_activations = (samples_covered != correct_predictions) & samples_covered

			# Update each metric for threshold i
			self.coverage[i] = np.sum(samples_covered) / complete_set_length
			self.risk[i] = np.sum(false_activations) / complete_set_length
			self.remaining_outliers[i] = (outliers & samples_covered).sum()

	def AI_Testing_metrics(self):
		if not self.data_is_in_correct_format():
			raise TypeError('Data is not properly set')

		# Compute the metrrics required.
		# Additional computations for Coverage Breakpoints
		inliers = (self.true_labels != self.outlier_label)
		baseline_risk = 1 - ((self.predictions == self.true_labels) & inliers).sum() / inliers.sum()

		# is there any coverage where we reach same error rate as achieved with only inliers?
		if any(x <= baseline_risk for x in self.risk):
			_, coverage_breakpoint_at_performance_level = [[x, y] for x, y in zip(self.risk, self.coverage) if x <= baseline_risk][0]
		else:
			coverage_breakpoint_at_performance_level = 0.0

		# Is there any coverage breakpoint where we detect all outliers?
		if any(x == 0 for x in self.remaining_outliers):
			idx = np.argmin(self.remaining_outliers)
			full_anomaly_detection = self.coverage[idx]
		else:
			full_anomaly_detection = 0.0

		self.metrics = {
			'auroc': self.auroc,
			'aupr': self.aupr,
			'TPR05': self.tpr[np.argmin(np.abs(self.fpr - 0.05))],
			'P95': self.precision[np.argmin(np.abs(self.tpr - 0.95))],
			'FPR95': 1 - self.tpr[np.argmin(np.abs(self.fpr - 0.95))],
			'CBPL': coverage_breakpoint_at_performance_level,
			'CBFAD': full_anomaly_detection
		}

		return self.metrics

	def IST_Metrics(self, accepted_risk=0.10):
		inliers = (self.true_labels != self.outlier_label)
		baseline_risk = 1 - ((self.predictions == self.true_labels) & inliers).sum() / inliers.sum()
		print("Baseline risk: {}".format(baseline_risk))
		# is there any coverage where we reach same error rate as achieved with only inliers?
		if any(x <= baseline_risk for x in self.risk):
			_, coverage_breakpoint_at_performance_level = [[x, y] for x, y in zip(self.risk, self.coverage) if x <= baseline_risk][0]
		else:
			coverage_breakpoint_at_performance_level = 0.0

		self.metrics = {
			'auroc': self.auroc,
			'aupr': self.aupr,
			'TPR95': self.fpr[np.argmin(np.abs(self.tpr - 0.95))],
			'CBPL': coverage_breakpoint_at_performance_level,
			'CARL': self.coverage[np.argmin(np.abs(self.risk - accepted_risk))],
			'RARL': self.risk[np.argmin(np.abs(self.risk - accepted_risk))],
		}
		return self.metrics


	def __plot_function(self, x, y, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, save_name=None, fancy=False):
		if not self.data_is_in_correct_format():
			raise TypeError('Data is not properly set')

		fig, ax1 = plt.subplots(figsize=(7, 4))
		plt.plot(x, y)

		if fancy: plt.fill_between([0, 1], [1, 1], [0.5, 0.5], color='red', alpha=0.1)
		if xlabel is not None: plt.xlabel(xlabel)
		if ylabel is not None: plt.ylabel(ylabel)
		if title is not None: plt.title(title)
		if ylim is not None: plt.ylim(ylim)
		if xlim is not None: plt.xlim(xlim)
		# plt.grid()
		if save_name is None:
			plt.draw()
		else:
			fig.savefig(save_file, dpi=400)

	def plot_risk_vs_coverage_curve(self, save_name=None, fancy=True):
		sns.set()
		self.__plot_function(
			self.coverage, self.risk, xlabel='coverage', ylabel='risk',
			title=None, save_name=save_name, xlim=(0, 1), ylim=(0, 1), fancy=fancy
		)

	def plot_roc_curve(self, save_name=None):
		self.__plot_function(
			self.fpr, self.tpr, xlabel='FPR', ylabel='TPR',
			title='Area under the curve: ' + str(round(self.auroc, 3)), save_name=save_name
		)

	def plot_precision_recall_curve(self, save_name=None):
		self.__plot_function(
			self.recall, self.precision, xlabel='recall', ylabel='precision',
			title='Area under the curve: ' + str(round(self.aupr, 3)), save_name=save_name
		)

	def plot_anomaly_distributions(self, bins=30, save_name=None):
		if not self.data_is_in_correct_format():
			raise TypeError('Data is not properly set')

		fig, ax1 = plt.subplots(figsize=(7, 4))
		plotting_legends = ['inliers', 'outliers']
		sns.distplot(
			self.anomaly_score[self.true_labels != self.outlier_label], ax=ax1,
			bins=bins, hist_kws={"label": plotting_legends [0]}
		)
		sns.distplot(
			self.anomaly_score[self.true_labels == self.outlier_label], ax=ax1,
			bins=bins, hist_kws={"label": plotting_legends[1]}
		)

		ax1.legend()
		ax1.set_xlabel('Anomaly score')
		# plt.grid()
		if save_name is None:
			plt.draw()
		else:
			fig.savefig(save_name, dpi=400)

	def plot_all(self, save_file=None):
		if save_file is not None:
			risk_cov = save_file + '_risk_coverage.pdf'
			distr = save_file + '_anomaly_distributions.pdf'
			roc_curve = save_file + '_roc_curve.pdf'
			pr_curve = save_file + '_pr_curve.pdf'
		else:
			risk_cov = None
			distr = None
			roc_curve = None
			pr_curve = None

		self.plot_risk_vs_coverage_curve(save_name=risk_cov)
		self.plot_anomaly_distributions(save_name=distr)
		self.plot_roc_curve(save_name=roc_curve)
		self.plot_precision_recall_curve(save_name=pr_curve)
