import torch
import torch.nn.functional as F


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def relu_evidence(y: torch.tensor) -> torch.tensor:
    return F.relu(y)


def softplus_evidence(y: torch.tensor) -> torch.tensor:
    return F.softplus(y)


def kl_divergence(alpha: torch.tensor, num_classes: int, device: torch.device = None) -> torch.tensor:
    if not device:
        device = get_device()

    uniform = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)

    kl_d = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True) + torch.lgamma(uniform).sum(dim=1, keepdim=True) \
           - torch.lgamma(uniform.sum(dim=1, keepdim=True)) \
            + (alpha-uniform).mul(torch.digamma(alpha)-torch.digamma(sum_alpha)).sum(dim=1, keepdim=True)
    return kl_d


def loglikelihood_loss(y: torch.tensor, alpha: torch.tensor, device: torch.device = None) -> torch.tensor:
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    phat = alpha/S
    prediction_error = torch.sum((y-phat)**2, dim=1, keepdim=True)
    dirichlet_variance = torch.sum(phat*(1-phat)/(S+1), dim=1, keepdim=True)
    return prediction_error + dirichlet_variance


def mse_loss(y: torch.tensor,
             alpha: torch.tensor,
             epoch_num: int,
             num_classes: int,
             device: torch.device = None) -> torch.tensor:
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device)
    annealing_coeff = torch.min(1.0, epoch_num/10.0) #warning here
    kl_alpha = (alpha-1)*(1-y)+1
    kl_d = kl_divergence(kl_alpha, num_classes, device)
    return loglikelihood + annealing_coeff*kl_d


def edl_mse_loss(output: torch.tensor,
                 target: torch.tensor,
                 epoch_num: int,
                 num_classes: int,
                 device: torch.device = None) -> torch.tensor:
    if not device:
        device = get_device()

    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(mse_loss(target, alpha, epoch_num, num_classes, device=device))
    return loss

