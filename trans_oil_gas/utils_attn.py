"""Script for attention analysis investigation."""

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy

import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from typing import List, Tuple, Union


def get_inputs_for_calc_grad(
    batch: torch.Tensor, delta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Obtain batches with substracted delta and added delta.

    :param batch: initial batch (well-interval)
    :param delta: hyperparameter that should be substracted or added
    :return: a tuple of changed versions of the initial batch
    """
    batch_plus, batch_minus = [], []
    for i in range(batch.size()[-1]):
        a, b = deepcopy(batch.detach()), deepcopy(batch.detach())
        a[:, :, i] += delta
        batch_plus.append(a)

        b[:, :, i] -= delta
        batch_minus.append(b)

    return torch.stack(batch_plus), torch.stack(batch_minus)


def get_gradients(
    pair_intervals: List[torch.tensor], # each interval has shape [batch_size, sequence_len, n_features] 
    model: nn.Module,
    model_type: str,
    device: Union[str, torch.DeviceObjType] = "cuda",
    agg: bool = True,
):
    """Get the model's gradients with respect to the input intervals.

    :param pair_intervals: a pair of well-intervals
    :param model: model
    :param model_type: type of model
    :param device: device
    :param agg: if True, aggregates the obtained gradients
    :return: similarity_result, similarity_scores, gradients with respect to the 1st well-interval, gradients with respect to the 2nd well-interval
    """
    slice_1, slice_2 = deepcopy(pair_intervals[0]).to(device).requires_grad_(), deepcopy(pair_intervals[1]).to(device).requires_grad_()

    if "triplet" in model_type:
        embs_1 = model(slice_1)
        embs_2 = model(slice_2)
        similarity_scores = 1 / ((embs_2 - embs_1).pow(2).sum(dim=1).sqrt() + 1e-8)
    elif "siamese" in model_type:
        similarity_scores = model(
            (
                slice_1,
                slice_2,
            )
        )
    slice_1.retain_grad()
    slice_2.retain_grad()
    similarity_result = similarity_scores.sum()
    similarity_result.backward()
    
    if agg:
        return similarity_result, similarity_scores, slice_1.grad.sum(dim=-1), slice_2.grad.sum(dim=-1)
    
    return similarity_result, similarity_scores, slice_1.grad, slice_2.grad


def get_attn_score(el: torch.Tensor) -> torch.Tensor:
    """Calculate attention score.

    :param el: attention matrix
    :return: attention score calculated via summarizing elements on the diagonal of attention matrix
    """
    ans = []
    for i in range(el.shape[0]):
        t = torch.diag(el[i, 0, :, :])
        for j in range(1, el.shape[1]):
            t += torch.diag(el[i, j, :, :])
        ans.append(t.detach().cpu().numpy().tolist())
    return torch.tensor(ans, device=el.device)


def get_attention_scores(x: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """Calculate attention scores for each element in well-interval.

    :param x: well-interval
    :param model: model from which embeddings and attention matrix are obtained
    :return: attention scores
    """
    attn = model.get_attention_maps(x)
    attn_sum = torch.tensor([]).to(x.device)

    for el in attn:
        attn_sum = torch.cat([attn_sum, get_attn_score(el)[:, :, None]], dim=-1)

    return attn_sum.sum(dim=-1)


def calc_corr(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """Calculate correlation coefficient of two tensors.

    :param x1: the first tensor
    :param x2: the second tensor
    :return: pearson correlation coefficient of two tensors
    """
    return np.corrcoef(
        x1.detach().cpu().numpy().reshape(-1), x2.detach().cpu().numpy().reshape(-1)
    )[0][1]


def largest_smallest_indices(
    ary: np.array, n: int, mode: str = "largest"
) -> Tuple[np.array, np.array]:
    """Return the n largest (or smallest) indices from a numpy array.

    :param ary: initial array
    :param n: the number of the biggest (or smallest) elements we want to find
    :param mode: indicate which element should be found: the biggest or the smallest
    :return: tuple of indices in the following format: [i-ths (rows) indices], [j-ths (columns) indices]
    """
    assert mode in [
        "largest",
        "smallest",
    ], "mode should be either 'largest' or 'smallest'"
    flat = ary.flatten()

    if mode == "largest":
        indices = np.argpartition(flat, -n)[-n:]
    elif mode == "smallest":
        indices = np.argpartition(flat, n)[:n]

    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def change_interval_part_grads_attns(
    x: torch.Tensor,
    model: nn.Module,
    model_type: str, 
    fill_with: str = "zeros",
    p: float = 0.2,
    grads_abs: bool = False,
    attns_abs: bool = False,
    elim_mode_grads: str = "low",
    elim_mode_attns: str = "low",
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]:
    """Replace p % of elements in well-interval with the highest gradients and the lowest attention scores.

    :param x: initial well-interval
    :param model: transforer model
    :param model_type: name of model (with the loss type: siamese or triplet) 
    :param fill_with: elements with which elements from well-interval would be masked
    :param p: the percentage of eliminated values
    :param grads_abs: indicate whether to consider gradients (if False) or absolute values of gradients (if True)
    :param attns_abs: indicate whether to consider attention scores (if False) or absolute values of attention scores (if True)
    :param elim_mode_grads: indicate the criteria for gradients for elements elimination (smallest or largest)
    :param elim_mode_attns: indicate the criteria for gradients for elements elimination (smallest or largest)
    :return: tuple of intervals with masked elements with the highest gradients, the smallest attention scores
             (if add_baseline True also return interval with masked random parts)
    """
    assert fill_with in [
        "zeros",
        "rand",
    ], "Filling type should be either 'zeros' or 'rand'"
    _, _, gr1, gr2 = get_gradients(
        [x, x],
        model,
        model_type,
        x.device,
        False,
    )
    if grads_abs:
        gr = np.abs(gr1.detach().cpu().numpy()) + np.abs(gr2.detach().cpu().numpy())
    else:
        gr = gr1.detach().cpu().numpy() + gr2.detach().cpu().numpy()

    a = get_attention_scores(x, model.encoder).detach().cpu().numpy()
    if attns_abs:
        a = np.abs(a)

    new_batch_g, new_batch_a = [], []

    n_g, n_a = int(p * x.shape[1] * x.shape[2]), int(p * x.shape[1])

    for i in range(x.shape[0]):

        new_x_g, new_x_a, new_gr, new_a = (
            deepcopy(x[i, :, :]),
            deepcopy(x[i, :, :]),
            deepcopy(gr[i, :, :]),
            deepcopy(a[i, :]),
        )

        i_g, j_g = largest_smallest_indices(new_gr, n=n_g, mode=elim_mode_grads)
        i_a = largest_smallest_indices(new_a, n=n_a, mode=elim_mode_attns)

        if fill_with == "rand":
            rand_g = torch.randn(n_g).to(new_x_g.device)
            rand_a = torch.randn(n_a).to(new_x_a.device)

        for k, (ii_g, jj_g) in enumerate(zip(i_g, j_g)):
            new_x_g[ii_g, jj_g] = 0 if fill_with == "zeros" else rand_g[k]

        new_batch_g.append(new_x_g[None, :, :].detach().cpu())

        for k, ii_a in enumerate(i_a):
            new_x_a[ii_a, :] = 0 if fill_with == "zeros" else rand_a[k]

        new_batch_a.append(new_x_a[None, :, :].detach().cpu())

    return torch.cat(new_batch_g).to(x.device), torch.cat(new_batch_a).to(x.device)

def robustness_test(
    x: torch.Tensor,
    fill_with: str = "zeros",
    p: float = 0.2,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]:
    """Replace p % of random elements in well-interval.

    :param x: initial well-interval
    :param fill_with: elements with which elements from well-interval would be masked
    :param p: the percentage of eliminated values
    :return: interval with masked random parts
    """
    assert fill_with in [
        "zeros",
        "rand",
    ], "Filling type should be either 'zeros' or 'rand'"
    new_batch_r = []

    n_g = int(p * x.shape[1] * x.shape[2])

    for i in range(x.shape[0]):
        new_x_r = deepcopy(x[i, :, :])

        rand_r = torch.randn(n_g).to(new_x_r.device)

        i_r, j_r = (
            torch.randint(low=0, high=x.shape[1], size=(n_g,)), #.to(new_x_r.device), 
            torch.randint(low=0, high=x.shape[2], size=(n_g,)) #.to(new_x_r.device)
        )
        for k, (ii_r, jj_r) in enumerate(zip(i_r, j_r)):
            new_x_r[ii_r, jj_r] = 0 if fill_with == "zeros" else rand_r[k]

        new_batch_r.append(new_x_r[None, :, :].detach().cpu())

    return torch.cat(new_batch_r).to(x.device)


def get_acc(
    s1: torch.Tensor, s2: torch.Tensor, model: nn.Module, target: torch.Tensor
) -> float:
    """Calculate model's accuracy.

    :param s1: the first well-interval
    :param s2: the second well-interval
    :param model: transformer model
    :param target: target values: 1 if s1 and s2 belong to the same well otherwise 0
    :return: model's accuracy
    """
    return accuracy_score(
        target.detach().cpu().numpy(),
        (model((s1, s2)).detach().cpu().numpy() > 0.5).astype(int),
    )

def get_metrics(
    s1: torch.Tensor, s2: torch.Tensor, model: nn.Module, target: torch.Tensor
) -> float:
    """Calculate model's accuracy.

    :param s1: the first well-interval
    :param s2: the second well-interval
    :param model: transformer model
    :param target: target values: 1 if s1 and s2 belong to the same well otherwise 0
    :return: model's accuracy
    """
    proba = model((s1, s2)).detach().cpu().numpy()
    acc = accuracy_score(
        target.detach().cpu().numpy(),
        (proba > 0.5).astype(int),
    )
    roc_auc = roc_auc_score(
        target,
        proba,
    )
    pr_auc = average_precision_score(
        target,
        proba,
    )
    return acc, roc_auc, pr_auc
