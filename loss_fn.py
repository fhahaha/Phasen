import logging

import torch


def loss_fn(pred, label, seq_len, loss_type):
    """
    Args:
        pred(torch.Tensor): [batch_size, 2, F, T]
        label(torch.Tensor): [batch_size, 2, F, T]
        seq_len(torch.Tensor): [batch_size]
        loss_type(str)
    """
    if loss_type == 'MSE':
        loss = torch.sum((pred - label).pow(2), 2)  # [batch_size, 2, T]
    elif loss_type == 'MAE':
        loss = torch.sum(torch.abs(pred - label), 2)
    else:
        logging.error('Not support {}!'.format(loss_type))
        raise ValueError
    loss = loss.mean(1)  # [batch_size, T]
    max_len = pred.size(-1)
    mask = torch.arange(0, max_len).unsqueeze(0).expand(seq_len.shape[0],
                                                        max_len)
    mask = mask <= torch.unsqueeze(seq_len, 1)
    loss = torch.where(mask, loss, torch.zeros_like(loss))  # [batch_size, T]
    loss = torch.sum(loss, [0, 1]) / torch.sum(seq_len)
    return loss
