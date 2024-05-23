
import subprocess
import lumicks.pylake as lk
import os
from pathlib import Path
import argparse


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

parser.add_argument(
    "-c",
    "--reference-channel",
    default="C3",
    help="Reference channel. C1=red C2=green C3=blue. Default=C3",
)

parser.add_argument("--keep-uncorrected-movie",
                    help="Keep intermediate aligned file",  action='store_true')

parser.add_argument("-d", "--dry-run",
                    help="Only produce the fiji script file",  action='store_true')

args = parser.parse_args()

channels = ["C1", "C2", "C3"]

if args.reference_channel not in channels:
    raise ValueError("Incorrect value for channel")
else:
    reference_channel = args.reference_channel


movie_path = args.movie_file
output_path = Path(args.output_directory)



os.makedirs(output_path, exist_ok=True)


movie_filename = os.path.basename(movie_path)

movie = lk.ImageStack(movie_path)  # Loading a stack.
aligned_movie_filename = Path(movie_path).stem + "_aligned.tiff"
aligned_movie_pathname = output_path / aligned_movie_filename
aligned_movie_pathname = aligned_movie_pathname.resolve()

if not args.dry_run:
    movie.export_tiff(aligned_movie_pathname)  # Save aligned wt stack



# Write Fiji macro to file
correct_drift = True

drift_table = aligned_movie_pathname.with_suffix('njt')

macro_path = f"{output_path / Path(movie_path).stem}_temp_macro.ijm"
with open(macro_path, "w") as f:
    f.write(f'open("{aligned_movie_pathname}");\n')
    f.write('run("Split Channels");\n')
    f.write(f'selectImage("{reference_channel}-{aligned_movie_filename}");')

    f.write(f'run("F4DR Estimate Drift","time=100 max=10 reference=[first frame (default, better for fixed)] apply choose=[{aligned_movie_pathname}_DriftTable.njt]");\n')
    if correct_drift:
        for channel in channels:
            if channel != reference_channel:
                f.write(f'selectImage("{channel}-{aligned_movie_filename}");\n')
                f.write(f'run("F4DR Correct Drift", "choose=[{drift_table}]");\n')
        for channel in channels:
            f.write(('selectImage("{}-{}");\n').format(channel, aligned_movie_filename))
            f.write(
                "close();\n",
            )

        f.write(  # This one can also be written as a loop if we expect to have more than the RGB channels.
            (
                'run("Merge Channels...", "c1=[C1-{} - drift corrected] c2=[C2-{} - drift corrected] c3=[C3-{} - drift corrected] create");\n'
            ).format(
                aligned_movie_filename, aligned_movie_filename, aligned_movie_filename
            ),
        )
        f.write(
            ('saveAs("Tiff", "{}/{}_drift_corrected.tif");\n').format(
                current_dir, aligned_movie_pathname.replace(".tiff", "").replace(".tif", "")
            )
        )
        f.write('run("Quit");')


path_to_fiji = args.path_to_fiji  # How to find automatically?
run_string = '{} -macro "{}"'.format(path_to_fiji, macro_path)
print(run_string)
# I think the drift correction plugin isn't happy with runnig headless
# run_string = '{} --headless --console -macro "temp_macro.ijm"'.format(path_to_fiji)

if args.dry_run:
    exit()

print("Begin fiji processing")
subprocess.run(run_string)
print("Finished fiji processing")


#os.remove(drift_table)
#os.remove(macro_path)
if not args.keep_uncorrected_movie:
    os.remove(aligned_movie_pathname)
