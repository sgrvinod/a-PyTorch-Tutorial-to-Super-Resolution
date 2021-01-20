import torch

from processing.jpeg_artifacts.model import ARCNN
from helper.color_convert import rgb_to_ycbcr, ycbcr_to_rgb


def denoise(t: torch.Tensor, denoiser: ARCNN) -> torch.Tensor:
    t = t.unsqueeze(0)
    t = t/2+0.5
    ycbcr_t = rgb_to_ycbcr(t)
    with torch.no_grad():
        result = denoiser(ycbcr_t[:, 0:1, :, :])
        result = torch.cat((result, ycbcr_t[:, 1:3, :, :]), 1)
        result = ycbcr_to_rgb(result)
    assert t.shape == result.shape
    result = (result * 2) - 1
    return result[0]