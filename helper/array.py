def tensor_to_np(x):
    array = (x.detach().cpu().numpy().transpose(1, 2, 0) + 1) * 0.5 * 255
    return array.astype('uint8')