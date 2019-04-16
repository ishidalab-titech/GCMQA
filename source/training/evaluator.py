import copy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import six
import os
import numpy as np
from chainer import configuration
import chainer
from chainer.backend import cuda
from chainer import reporter
from chainer.training.extensions import Evaluator
from sklearn import metrics
import chainer.functions as F
from scipy.stats import pearsonr, spearmanr, kendalltau
from chainer import reporter as reporter_module
from chainer import function


def _get_1d_numpy_array(v):
    """Convert array or Variable to 1d numpy array
    Args:
        v (numpy.ndarray or cupy.ndarray or chainer.Variable): array to be
            converted to 1d numpy array
    Returns (numpy.ndarray): Raveled 1d numpy array
    """
    if isinstance(v, chainer.Variable):
        v = v.data
    return cuda.to_cpu(v).ravel()


def _to_list(a):
    """convert value `a` to list
    Args:
        a: value to be convert to `list`
    Returns (list):
    """
    if isinstance(a, (int, float)):
        return [a, ]
    else:
        # expected to be list or some iterable class
        return a


def plot_roc(y_true, y_score, out_name):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
    auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)' % auc)
    plt.legend()
    plt.title('ROC curve', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.grid(True)
    plt.savefig(out_name)


class GraphEvaluator(Evaluator):
    def __init__(self, iterator, target, comm, converter, local_loss_func, global_loss_func, config, device=None,
                 eval_hook=None, eval_func=None, name=None, pos_labels=1, ignore_labels=None):
        super(GraphEvaluator, self).__init__(
            iterator, target, converter=converter, device=device, eval_hook=eval_hook, eval_func=eval_func)
        self.name = name
        self.pos_labels = _to_list(pos_labels)
        self.ignore_labels = _to_list(ignore_labels)
        self.comm = comm
        self.local_loss_func = local_loss_func
        self.global_loss_func = global_loss_func
        self.config = config

    def __call__(self, trainer=None):
        """Executes the evaluator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        with reporter:
            with configuration.using_config('train', False):
                result = self.evaluate_roc(trainer=trainer)

        reporter_module.report(result)
        return result

    def evaluate_roc(self, trainer):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        updater = trainer.updater
        epoch = updater.epoch
        out_dir = trainer.out

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        local_score_total, global_score_total = [], []
        local_label_total, global_label_total = [], []
        local_mean_total = []
        for batch in it:
            in_arrays = self.converter(batch, self.device)

            with function.no_backprop_mode(), chainer.using_config('train', False):
                local_score, global_score = eval_func(*in_arrays[:-2])
                batch_indices = in_arrays[-3]
                local_label, global_label = in_arrays[-2], in_arrays[-1]
            local_score_total.extend(cuda.to_cpu(local_score.data))
            local_label_total.extend(cuda.to_cpu(local_label))
            global_score_total.extend(cuda.to_cpu(global_score.data))
            global_label_total.extend(cuda.to_cpu(global_label))
            for o, n in zip([0, *batch_indices[:-1]], batch_indices):
                local_mean_total.append(np.mean(cuda.to_cpu(F.sigmoid(local_score).data)[o:n], axis=0))

        local_score_total = np.array(local_score_total)
        local_label_total = np.array(local_label_total)
        global_score_total = np.array(global_score_total)
        global_label_total = np.array(global_label_total)
        local_mean_total = np.array(local_mean_total)
        observation = {}
        local_score_total = self.comm.gather(local_score_total)
        local_label_total = self.comm.gather(local_label_total)
        global_score_total = self.comm.gather(global_score_total)
        global_label_total = self.comm.gather(global_label_total)
        local_mean_total = self.comm.gather(local_mean_total)
        if self.comm.rank == 0:
            local_score_total = np.vstack(local_score_total)
            local_label_total = np.vstack(local_label_total)
            global_score_total = np.vstack(global_score_total)
            global_label_total = np.vstack(global_label_total)
            local_mean_total = np.vstack(local_mean_total)
            with reporter.report_scope(observation):

                # Evaluate Local metrics
                loss = 0
                if self.config['local_mode']:
                    if self.config['local_type'] == 'Classification':
                        local_loss = self.local_loss_func(local_score_total, local_label_total.astype(np.int32))
                    else:
                        local_loss = self.local_loss_func(F.sigmoid(local_score_total), local_label_total)
                    reporter.report({'local_loss': local_loss}, self._targets['main'])
                    all_auc = 0
                    all_local_pearson, all_local_spearman, all_local_kendall = 0, 0, 0
                    all_pearson, all_spearman, all_kendall = 0, 0, 0
                    for il, local_name in enumerate(self.config['local_label']):
                        local_true, local_score = local_label_total[:, il], F.sigmoid(local_score_total[:, il]).data

                        if self.config['local_type'] == 'Classification':
                            out_name = os.path.join(out_dir, str(epoch) + 'epoch_' + local_name + '_roc.pdf')
                            roc_auc = metrics.roc_auc_score(local_true, local_score)
                            all_auc += roc_auc
                            plot_roc(y_true=local_true, y_score=local_score, out_name=out_name)
                            reporter.report({local_name + '_auc': roc_auc}, self._targets['main'])
                        else:
                            local_pearson = pearsonr(local_true, local_score)[0]
                            local_spearman = spearmanr(local_true, local_score)[0]
                            local_kendall = kendalltau(local_true, local_score)[0]
                            reporter.report({'{}_local_pearson'.format(local_name): local_pearson},
                                            self._targets['main'])
                            reporter.report({'{}_local_spearman'.format(local_name, ): local_spearman},
                                            self._targets['main'])
                            reporter.report({'{}_local_kendall'.format(local_name, ): local_kendall},
                                            self._targets['main'])
                            all_local_pearson += local_pearson
                            all_local_spearman += local_spearman
                            all_local_kendall += local_kendall
                        for ig, global_name in enumerate(self.config['global_label']):
                            l_mean, g = local_mean_total[:, il], global_label_total[:, ig]
                            local_mean_pearson = pearsonr(l_mean, g)[0]
                            local_mean_spearman = spearmanr(l_mean, g)[0]
                            local_mean_kendall = kendalltau(l_mean, g)[0]
                            all_pearson += local_mean_pearson
                            all_spearman += local_mean_spearman
                            all_kendall += local_mean_kendall
                            reporter.report({'{}_{}_pearson'.format(local_name, global_name): local_mean_pearson},
                                            self._targets['main'])
                            reporter.report({'{}_{}_spearman'.format(local_name, global_name): local_mean_spearman},
                                            self._targets['main'])
                            reporter.report({'{}_{}_kendall'.format(local_name, global_name): local_mean_kendall},
                                            self._targets['main'])
                    loss += local_loss
                    all_pearson = all_pearson / (len(self.config['local_label']) * len(self.config['global_label']))
                    all_spearman = all_spearman / (len(self.config['local_label']) * len(self.config['global_label']))
                    all_kendall = all_kendall / (len(self.config['local_label']) * len(self.config['global_label']))
                    reporter.report({'local_mean_pearson': all_pearson}, self._targets['main'])
                    reporter.report({'local_mean_spearman': all_spearman}, self._targets['main'])
                    reporter.report({'local_mean_kendall': all_kendall}, self._targets['main'])

                    if self.config['local_type'] == 'Classification':
                        reporter.report({'local_auc': all_auc / len(self.config['local_label'])}, self._targets['main'])
                    else:
                        reporter.report({'local_pearson': all_local_pearson / len(self.config['local_label'])},
                                        self._targets['main'])
                        reporter.report({'local_spearman': all_local_spearman / len(self.config['local_label'])},
                                        self._targets['main'])
                        reporter.report({'local_kendall': all_local_kendall / len(self.config['local_label'])},
                                        self._targets['main'])

                # Evaluate Global metrics
                if self.config['global_mode']:
                    global_loss = self.global_loss_func(global_label_total, global_score_total)
                    reporter.report({'global_loss': global_loss}, self._targets['main'])
                    all_pearson, all_spearman, all_kendall = 0, 0, 0
                    for ig, global_name in enumerate(self.config['global_label']):
                        global_score, global_true = global_score_total[:, ig], global_label_total[:, ig]
                        global_pearson = pearsonr(global_score, global_true)[0]
                        global_spearman = spearmanr(global_score, global_true)[0]
                        global_kendall = kendalltau(global_score, global_true)[0]
                        reporter.report({'{}_global_pearson'.format(global_name): global_pearson},
                                        self._targets['main'])
                        reporter.report({'{}_global_spearman'.format(global_name): global_spearman},
                                        self._targets['main'])
                        reporter.report({'{}_global_kendall'.format(global_name): global_kendall},
                                        self._targets['main'])
                        all_pearson += global_pearson
                        all_spearman += global_spearman
                        all_kendall += global_kendall
                    all_pearson = all_pearson / len(self.config['global_label'])
                    all_spearman = all_spearman / len(self.config['global_label'])
                    all_kendall = all_kendall / len(self.config['global_label'])
                    reporter.report({'global_pearson': all_pearson}, self._targets['main'])
                    reporter.report({'global_spearman': all_spearman}, self._targets['main'])
                    reporter.report({'global_kendall': all_kendall}, self._targets['main'])

                    loss += global_loss
                reporter.report({'loss': loss}, self._targets['main'])
        return observation
