import torch
import torch.nn.functional as F


@torch.no_grad()
def diagonal_sobel_retina(x, device=None, batch_size=512):
    target_device = torch.device(device or x.device)
    kernels = [
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]],
        [[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]],
    ]
    kernels = torch.tensor(kernels, device=target_device).reshape(4, 1, 3, 3)
    outputs = []
    for start in range(0, len(x), batch_size):
        raw = x[start : start + batch_size].to(target_device).float()
        images = raw.reshape(-1, 1, 28, 28)
        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        gradients = F.conv2d(padded, kernels)
        edges = (gradients.abs() > 0.5).float().reshape(-1, 4, 784)
        outputs.append(torch.cat([raw, edges.reshape(-1, 3136)], dim=1))
    return torch.cat(outputs)
