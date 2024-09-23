import genstudio.plot as Plot
from PIL import Image
import numpy as np


def save_bw_image(src, dest, threshold=200):
    image = Image.open(src).convert("L")
    binary_array = np.array(image) >= threshold
    Image.fromarray((binary_array * 255).astype(np.uint8)).save(dest)


# Save a black and white copy of the image
save_bw_image("./genjax-logo.png", "./genjax-logo-bw.png")

#
(
    # Path from project root (for Jupyter environment)
    Plot.img(["docs/genjax-logo-bw.png"], src=Plot.identity, width=1530, height=330)
    + Plot.aspectRatio(1)
)

Plot.image
