import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pdf2image import convert_from_path


def print_bbox(file, rectangles, out_file):

    image = convert_from_path(file)
    x = np.array(image[0], dtype=np.uint8)
    plt.imshow(x)

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(x)

    for index, row in rectangles.iterrows():

        # Create a Rectangle patch
        rect = patches.Rectangle((row['left'], row['top']), row['right'] - row['left'], row['bottom']-row['top'], linewidth=0.3,
                             edgecolor='r', facecolor="none")

        # Add the patch to the Axes
        ax.add_patch(rect)

        ax.annotate(row._name, (row['left'], row['top']), color='blue', fontsize=1.5, ha='left', va='bottom')

    plt.savefig(out_file, dpi=400)

    plt.show()

