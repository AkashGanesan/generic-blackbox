import torch
import torch.nn.functional as F
import torch.nn as nn


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Input:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.

    Output:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def kld_loss(mean,
             log_var):
    ''' KLD loss '''
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
    return KLD


def l2_loss(y_pred, y_true, mode='sum'):
    """
    Input:
    - y_pred: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - y_true: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    batch, _, seq_len = y_pred.size()
    loss = (y_true - y_pred).norm(dim=1)
    # if mode == 'sum':
    return torch.sum(loss, dim=1) / seq_len
    # elif mode == 'average':
    #     return torch.sum(loss) / torch.numel(loss_mask.data)
    # elif mode == 'raw':
    #     return loss.sum(dim=2).sum(dim=1)
