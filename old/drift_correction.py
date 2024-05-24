import numpy as np
import lumicks.pylake as lk
from skimage.transform import rescale
import tifffile
import os
from cv2 import warpAffine
from pathlib import Path
from picasso import io, postprocess
from json import JSONEncoder
import argparse
import shutil
import subprocess


def norm_image(image, inverse=False):
    amin = image.min()

    amax = image.max()

    if inverse:
        return 1 - (image - amin) / (amax - amin)

    else:
        return (image - amin) / (amax - amin)


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return JSONEncoder.default(self, obj)


# Don't run when in notebook mode

parser = argparse.ArgumentParser(
    description="""Scripts to align various channels based on reference beads.""",
    epilog="""""",
)
parser.add_argument("movie_file", help="Movie tif file")

parser.add_argument(
    "-o",
    "--output-directory",
    default="output",
    help="Output directory. Default=output/",
)
parser.add_argument(
    "-m", "--transform-matrix", help="Previously calculated matrix in .json format"
)
parser.add_argument(
    "-f", "--fit_method", default="lq", help="Fit method for picasso.  Default=lq"
)
parser.add_argument(
    "-b",
    "--box_size",
    default=31,
    help="Box sized for picasso. Caution=Not any size is allowed. Default=31",
)
parser.add_argument(
    "-g",
    "--min_gradient",
    default=6000,
    help="Minimum gradient for picasso. Default=6000",
)
parser.add_argument(
    "-e",
    "--max_pos_error",
    default=3.5,
    help="Maximum standard dev accepted for x and y position of spots. Default=3.5",
)
parser.add_argument("-p", "--max_photons", help="Maximum number of photons for spots.")
parser.add_argument(
    "-s",
    "--segmentation",
    default=300,
    help="Drift correction segmentation. Default=300",
)

args = parser.parse_args()

movie_path = args.movie_file
output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user

os.makedirs(output_path, exist_ok=True)

# Copy movie file to output and rename to tif

shutil.copy2(movie_path, output_path + os.path.basename(movie_path))

movie_path = output_path + os.path.basename(movie_path)

# rename tiff to tif files
if movie_path.endswith(".tiff"):
    os.rename(movie_path, movie_path[:-1])
    movie_path = movie_path[:-1]

movie = lk.ImageStack(movie_path)  # Loading a stack.

movie.export_tiff(
    output_path + Path(movie_path).stem + "_aligned.tif"
)  # Save aligned wt stack

# Get channel g
movie_r = movie.get_image(channel="red")
movie_g = movie.get_image(channel="green")
movie_b = movie.get_image(channel="blue")

movie_r_path = output_path + Path(movie_path).stem + "_r.tif"
movie_g_path = output_path + Path(movie_path).stem + "_g.tif"
movie_b_path = output_path + Path(movie_path).stem + "_b.tif"


# tifffile.imwrite(movie_r_path, movie_r)  #no need to export these
tifffile.imwrite(movie_g_path, movie_g)
# tifffile.imwrite(movie_b_path, movie_b)

# plt.imshow(movie_r[0], alpha=0.3, cmap="Reds")
# plt.imshow(movie_g[0], alpha=0.3, cmap="Greens")
# plt.imshow(movie_b[0], alpha=0.3, cmap="Blues")

run_string = (
    "python -m picasso localize "
    + movie_g_path
    + " --fit-method "
    + args.fit_method
    + " -b "
    + str(args.box_size)
    + " --gradient "
    + str(args.min_gradient)
)
subprocess.run(run_string)

# Get shift

locs_path = output_path + Path(movie_g_path).stem + "_locs.hdf5"
yaml_path = output_path + Path(movie_g_path).stem + "_locs.yaml"

locs, info = io.load_locs(locs_path)

post_results = postprocess.undrift(
    locs, info, segmentation=args.segmentation, display=False
)

# Transform using affine transform

undrifted_movie_r = np.copy(movie_r)
undrifted_movie_g = np.copy(movie_g)
undrifted_movie_b = np.copy(movie_b)


scaling_matrix = np.array([[1, 0], [0, 1]])

rotation_matrix = np.array([[1, 0], [0, 1]])

shearing_matrix = np.array([[1, 0], [0, 1]])


for frame in range(0, len(post_results[0])):
    # print(, post_results[frame][0][1])
    # undrifted_frame = np.copy(undrifted_)
    x_displacement = -post_results[0][frame][0]
    y_displacement = -post_results[0][frame][1]
    translation_matrix = np.array([[x_displacement], [y_displacement]])
    # Combine all transformations into a single transformation matrix
    transform_mat = np.hstack(
        (scaling_matrix.dot(rotation_matrix).dot(shearing_matrix), translation_matrix)
    )
    undrifted_frame_r = warpAffine(
        movie_r[frame],
        transform_mat,
        (movie_r[frame].shape[1], movie_r[frame].shape[0]),
    )
    undrifted_frame_g = warpAffine(
        movie_g[frame],
        transform_mat,
        (movie_g[frame].shape[1], movie_g[frame].shape[0]),
    )

    undrifted_frame_b = warpAffine(
        movie_b[frame],
        transform_mat,
        (movie_b[frame].shape[1], movie_b[frame].shape[0]),
    )
    undrifted_movie_r[frame] = undrifted_frame_r
    undrifted_movie_g[frame] = undrifted_frame_g
    undrifted_movie_b[frame] = undrifted_frame_b

# Output images

# movie_r_undrifted_path = output_path + Path(movie_r_path).stem + "_undrifted.tif"  #In case we want to export individual channels
# movie_g_undrifted_path = output_path + Path(movie_g_path).stem + "_undrifted.tif"  #In case we want to export individual channels
# movie_b_undrifted_path = output_path + Path(movie_b_path).stem + "_undrifted.tif"  #In case we want to export individual channels

# tifffile.imwrite(movie_r_undrifted_path, undrifted_movie_r)
# tifffile.imwrite(movie_g_undrifted_path, undrifted_movie_g)
# tifffile.imwrite(movie_b_undrifted_path, undrifted_movie_b)

stacked_image = np.stack(
    [undrifted_movie_r, undrifted_movie_g, undrifted_movie_b], axis=1
)  # Save stacked g and irm image

#  print(stacked_image.shape)

tifffile.imwrite(
    output_path + Path(movie_path).stem + "_undrifted.tif",
    np.float32(stacked_image),
    imagej=True,
    metadata={
        "Composite mode": "composite",  # This is whats needed for ImageJ to automatically merge the channels
    },
)

# Delete all temp files
movie._src.close()
os.remove(movie_path)
os.remove(locs_path)
os.remove(yaml_path)
os.remove(movie_g_path)
