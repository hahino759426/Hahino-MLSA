import torch
import torch.nn.functional as F


@torch.no_grad()
def sobel_retina(x, device=None, batch_size=512):
    target_device = torch.device(device or x.device)
    kernel_x = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=target_device,
    ).reshape(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=target_device,
    ).reshape(1, 1, 3, 3)
    outputs = []
    for start in range(0, len(x), batch_size):
        raw = x[start : start + batch_size].to(target_device).float()
        images = raw.reshape(-1, 1, 28, 28)
        padded = F.pad(images, (1, 1, 1, 1), mode="reflect")
        grad_x = F.conv2d(padded, kernel_x)
        grad_y = F.conv2d(padded, kernel_y)
        edge_x = (grad_x.abs() > 0.5).float().reshape(-1, 784)
        edge_y = (grad_y.abs() > 0.5).float().reshape(-1, 784)
        outputs.append(torch.cat([raw, edge_x, edge_y], dim=1))
    return torch.cat(outputs)
