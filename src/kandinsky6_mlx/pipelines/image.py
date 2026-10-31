"""Reference-image preprocessing for image-to-video, as `Kandinsky6TI2VAPipeline.encode_image` does it.

`VideoProcessor.preprocess(image, height, width, resize_mode="crop")` (diffusers 0.42.1, `VaeImageProcessor` with its
defaults: `do_resize`, Lanczos resampling, no RGB conversion, `do_normalize`) for a PIL image: resize keeping the
aspect ratio so the image covers `width x height` (`_resize_and_crop`), paste it centred on a new RGB canvas (which
crops the excess), scale to [0, 1] in float32 and map to [-1, 1]. The same PIL calls are made here, so the pixels are
identical.
"""

import numpy as np
from PIL import Image


def resize_and_crop(image: Image.Image, width: int, height: int) -> Image.Image:
    ratio = width / height
    src_ratio = image.width / image.height
    src_w = width if ratio > src_ratio else image.width * height // image.height
    src_h = height if ratio <= src_ratio else image.height * width // image.width
    resized = image.resize((src_w, src_h), resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return canvas


def preprocess_image(image: Image.Image, height: int, width: int, vae_scale_factor: int = 8) -> np.ndarray:
    """PIL image -> `(1, 3, height, width)` float32 in [-1, 1]. Height and width are rounded down to multiples of
    `vae_scale_factor`, as `get_default_height_width` does."""
    height, width = height - height % vae_scale_factor, width - width % vae_scale_factor
    pixels = np.array(resize_and_crop(image, width, height)).astype(np.float32) / 255.0
    return (2.0 * pixels - 1.0).transpose(2, 0, 1)[None]
