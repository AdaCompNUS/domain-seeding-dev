''' Util functions '''
import torch
import sys
import traceback
from datetime import datetime
import numpy as np


def error_handler(e):
    print(
        '\nError on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename, sys.exc_info()[-1].tb_lineno),
        type(e).__name__, e)
    print('Call-stack:')
    traceback.print_stack()
    sys.stdout.flush()


def error_handler_with_log(file, e):
    error_handler(e)
    log_flush(file,
        '\nError on file {} line {}'.format(sys.exc_info()[-1].tb_frame.f_code.co_filename,
                                            sys.exc_info()[-1].tb_lineno))


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


def log_flush(file, msg):
    try:
        file.write('{}: {}\n'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S"), msg))
        file.flush()
        print_flush(msg)
    except Exception as e:
        error_handler(e)


def to_batch(state, action, reward, next_state, done, device):
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    action = torch.FloatTensor([action]).view(1, -1).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)
    return state, action, reward, next_state, done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    # torch.autograd.set_detect_anomaly(True)
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    optim.step()


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def check_consistent_length(*arrays):
    lengths = [X.shape[0] for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric"):
    check_consistent_length(y_true, y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    return y_true, y_pred, multioutput

def explained_variance_score(y_true, y_pred,
                             sample_weight=None,
                             multioutput='uniform_average'):
    """Explained variance regression score function
    Best possible score is 1.0, lower values are worse.
    Read more in the :ref:`User Guide <explained_variance_score>`.
    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.
    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.
    multioutput : string in ['raw_values', 'uniform_average', \
                'variance_weighted'] or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.
    Returns
    -------
    score : float or ndarray of floats
        The explained variance or ndarray if 'multioutput' is 'raw_values'.
    Notes
    -----
    """
    try:
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        y_true, y_pred, multioutput = _check_reg_targets(
            y_true, y_pred, multioutput)

        y_diff_avg = np.average(y_true - y_pred, weights=sample_weight, axis=0)
        numerator = np.average((y_true - y_pred - y_diff_avg) ** 2,
                               weights=sample_weight, axis=0)

        y_true_avg = np.average(y_true, weights=sample_weight, axis=0)
        denominator = np.average((y_true - y_true_avg) ** 2,
                                 weights=sample_weight, axis=0)

        nonzero_numerator = numerator != 0
        nonzero_denominator = denominator != 0
        valid_score = nonzero_numerator & nonzero_denominator

        output_scores = np.ones(y_true.shape[1])

        output_scores[valid_score] = 1 - (numerator[valid_score] /
                                          denominator[valid_score])
        output_scores[nonzero_numerator & ~nonzero_denominator] = 0.
        if isinstance(multioutput, str):
            if multioutput == 'raw_values':
                # return scores individually
                return output_scores
            elif multioutput == 'uniform_average':
                # passing to np.average() None as weights results is uniform mean
                avg_weights = None
            elif multioutput == 'variance_weighted':
                avg_weights = denominator
        else:
            avg_weights = multioutput

        return np.average(output_scores, weights=avg_weights)
    except Exception as e:
        error_handler(e)

