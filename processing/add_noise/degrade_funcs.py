import io

import imageio
from PIL import Image

IMG = Image.Image


# Image is 2D numpy array, q is quality 0-100
def jpeg_blur(img: IMG, q: float) -> IMG:
    assert 0 <= q <= 100
    if q == 0:
        return img
    buf = io.BytesIO()
    imageio.imwrite(buf, img, format='jpg', quality=q)
    s = buf.getbuffer()
    return Image.fromarray(imageio.imread(s, format='jpg'))

