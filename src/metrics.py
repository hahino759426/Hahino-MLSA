import torch


def confusion(predictions, labels):
    valid = predictions >= 0
    return torch.bincount(labels[valid] * 10 + predictions[valid], minlength=100).reshape(10, 10).cpu().tolist()


def recalls(predictions, labels):
    return [float((predictions[labels == path] == path).float().mean()) for path in range(10)]
