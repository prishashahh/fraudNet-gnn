import torch


def accuracy(preds, labels):

    preds = torch.sigmoid(torch.tensor(preds)) > 0.5
    labels = torch.tensor(labels).float()

    return (preds.float() == labels).float().mean().item()


def precision(preds, labels):

    preds = torch.sigmoid(torch.tensor(preds)) > 0.5
    labels = torch.tensor(labels).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()

    return (tp / (tp + fp + 1e-8)).item()


def recall(preds, labels):

    preds = torch.sigmoid(torch.tensor(preds)) > 0.5
    labels = torch.tensor(labels).float()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    return (tp / (tp + fn + 1e-8)).item()