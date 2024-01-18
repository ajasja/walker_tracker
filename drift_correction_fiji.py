# %%
import numpy as np
import subprocess
import lumicks.pylake as lk
from skimage.transform import rescale
import os
from pathlib import Path
import argparse

# %%

parser = argparse.ArgumentParser(
    description="""Script to drift-correct movies.""",
)
parser.add_argument("movie_file", help="Movie tif file")

parser.add_argument(
    "-o",
    "--output-directory",
    default="output/",
    help="Output directory. Default=output/",
)

parser.add_argument(
    "-f",
    "--path-to-fiji",
    default="C:/Program Files/Fiji.app/ImageJ-win64.exe",
    help="Path to fiji executable. Default='C:/Program Files/Fiji.app/ImageJ-win64.exe'",
)

parser.add_argument("--keep_uncorrected_movie")

args = parser.parse_args()

movie_path = args.movie_file
output_path = (
    args.output_directory + "/"
)  # The trailing slash is in case it wasn't added by the user

# %%
os.makedirs(output_path, exist_ok=True)

# %%
movie_filename = os.path.basename(movie_path)

# %%
movie = lk.ImageStack(movie_path)  # Loading a stack.
aligned_movie_path = output_path + Path(movie_path).stem + "_aligned.tiff"
aligned_movie_filename = Path(movie_path).stem + "_aligned.tiff"
movie.export_tiff(aligned_movie_path)  # Save aligned wt stack


# %%
# Write Fiji macro to file
correct_drift = True
current_dir = os.getcwd()
current_dir = current_dir.replace("\\", "/")

with open("temp_macro.ijm", "w") as f:
    f.write('open("{}/{}");\n'.format(current_dir, aligned_movie_path))
    f.write('run("Split Channels");\n')
    f.write(('selectImage("C3-{}");\n').format(aligned_movie_filename))
    f.write(
        (
            'run("F4DR Estimate Drift","time=100 max=10 reference=[first frame (default, better for fixed)] apply choose=[{}/{}output.njt]");\n'
        ).format(current_dir, output_path),
    )
    if correct_drift:
        f.write(('selectImage("C2-{}");\n').format(aligned_movie_filename))
        f.write(
            (
                'run("F4DR Correct Drift", "choose=[{}/{}outputDriftTable.njt]");\n'
            ).format(current_dir, output_path)
        )
        f.write(('selectImage("C1-{}");\n').format(aligned_movie_filename))
        f.write(
            (
                'run("F4DR Correct Drift", "choose=[{}/{}outputDriftTable.njt]");\n'
            ).format(current_dir, output_path)
        )
        f.write(('selectImage("C3-{}");\n').format(aligned_movie_filename))
        f.write(
            "close();\n",
        )
        f.write(('selectImage("C2-{}");\n').format(aligned_movie_filename))
        f.write(
            "close();\n",
        )
        f.write(('selectImage("C1-{}");\n').format(aligned_movie_filename))
        f.write(
            "close();\n",
        )
        f.write(
            (
                'run("Merge Channels...", "c1=[C1-{} - drift corrected] c2=[C2-{} - drift corrected] c3=[C3-{} - drift corrected] create");\n'
            ).format(
                aligned_movie_filename, aligned_movie_filename, aligned_movie_filename
            ),
        )
        f.write(
            ('saveAs("Tiff", "{}/{}_drift_corrected.tif");\n').format(
                current_dir, aligned_movie_path.replace(".tiff", "").replace(".tif", "")
            )
        )
        f.write('run("Quit");')

# %%
path_to_fiji = args.path_to_fiji  # How to find automatically?
run_string = '{} -macro "temp_macro.ijm"'.format(path_to_fiji)

# I think the drift correction plugin isn't happy with runnig headless
# run_string = '{} --headless --console -macro "temp_macro.ijm"'.format(path_to_fiji)
print("Begin fiji processing")
subprocess.run(run_string)
print("Finished fiji processing")

# %%
os.remove("{}/outputDriftTable.njt".format(output_path))
os.remove("temp_macro.ijm")
if not args.keep_uncorrected_movie:
    os.remove("{}".format(aligned_movie_path))
