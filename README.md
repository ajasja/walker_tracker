# walker_tracker

Tracking single molecules of walkers on de novo fibers, including drift correction, localization fitting, and trajectory analysis.

This repository contains scripts and notebooks used to preprocess multichannel microscopy movies, correct sample drift using Fiji, and extract single-molecule trajectories for downstream analysis.

---

## Repository structure

- `utils.py`  
  Core utility functions for fibre masking, single-molecule fitting (Picasso), trajectory linking (Trackpy), diffusion/step statistics helpers, and movie rendering (BW/RGB overlays). 

- `drift_correction_fiji.py`  
  Command-line Python script that performs drift correction of multichannel TIFF movies by calling **Fiji/ImageJ** with the *F4DR* drift correction plugin. 

- `drift_correct_folder.ipynb`  
  Jupyter notebook to batch drift-correct all movies in a folder using the Fiji-based workflow.

- Analysis / simulation notebooks:
  - `2D_fit_diffusion_folder.ipynb` — batch 2D diffusion fitting over a folder
  - `2D_parametric_diffusion_folder.ipynb` — parametric diffusion analysis over a folder
  - `trajectory_analysis.ipynb` — trajectory/step-size analysis and plotting
  - `simulate_random_walk.ipynb` — random-walk simulations for sanity checks / comparisons
  - `00-generate-sythetic-2d-normal-dist.ipynb` — generate synthetic 2D normal samples


---

## Requirements

### Python

- Python ≥ 3.8
- `numpy`
- `pandas`
- `matplotlib`
- `tifffile`
- `scikit-image`
- `trackpy`
- `lumicks.pylake` (optional; some workflows use it)
- **Picasso** (for localization fitting; invoked via `python -m picasso localize ...`)

It is recommended to install dependencies using the same environment as **ImageAligner**, as this repository was developed alongside it.

### Fiji / ImageJ

- Fiji (ImageJ distribution, version 2.14.0)
- **F4DR** drift correction plugin

To install the plugin:
1. Download the plugin `.jar` file.
2. Open Fiji → `Plugins` → `Install...`
3. Select the `.jar` file and restart Fiji.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/walker_tracker.git
cd walker_tracker
```

Set up and activate your Python environment, then install required Python packages. If the ImageAligner environment has not been set up beforehand, follow the following instructions:

1. First create the conda environment
    >conda env create --name <ImageAligner> -f image_aligner_env.yml

2. Then activate it

    >conda activate <ImageAligner>

3. And then install picasso

    >pip install picassosr

The installation usually takes approximately half an hour.

---

## Drift correction (single movie)

Use `drift_correction_fiji.py` to drift-correct a single multichannel TIFF movie:

```bash
python drift_correction_fiji.py path/to/movie.tif \
    --output-directory output/ \
    --path-to-fiji "C:/path/to/Fiji.app/ImageJ-win64.exe" \
    --reference-channel C4
```

### Important arguments

- `movie_file` (required): Path to the input TIFF movie
- `-o, --output-directory`: Directory for corrected output (default: `output/`)
- `-f, --path-to-fiji`: Path to the Fiji executable
- `-c, --reference-channel`: Channel used for drift estimation (`C1`–`C4`)
- `--dry-run`: Only generate the Fiji macro without running it
- `--keep-uncorrected-movie`: Keep intermediate files

The script generates a temporary Fiji macro, runs drift estimation on the reference channel, applies the correction to all channels, and saves a drift-corrected TIFF.

---

## Batch drift correction

To drift-correct an entire folder of movies, use the notebook:

```text
drift_correct_folder.ipynb
```

This notebook loops over all TIFF files in a directory and applies the same Fiji-based drift correction workflow.

---

## Tracking workflow (overview)

A typical analysis pipeline is:

1. Drift-correct raw movies (Fiji)
2. Restrict the walker channel to fibre regions (masking)
3. Fit single-molecule localizations (Picasso)
4. Link localizations into trajectories (Trackpy)
5. Compute step sizes / diffusion parameters and plot results

Both `2D_fit_diffusion_folder.ipynb` and `2D_parametric_diffusion_folder.ipynb` will perform steps 2. to 5. on an entire directory of '.tif' files and generate a 'diff_info.csv' file that contains the following info for each file/movie:

1. diffusion coefficient in each dimension
2. the time it takes a particle to cover the length of 1 micrometre
3. the average dwell time of all particles in the analysed video

In addition to the 'diff_info.csv' file, the notebooks output the following:

1. A tif-stack with one channel containing only the walkers localized on the tracks
2. An '.hdf5' file containing particle localizations
3. A '.yaml' file from Picasso particle localization
4. A '.tray.csv' file containing particle IDs, particle coordinates and the length of their trajectories
5. A '.steps.csv' file containing the particle IDs, particle step sizes in each dimension and the length of their steps
6. A '.diff' file containing various diffusion data calculated in the pipeline
7. A 2D-Gaussian contour plotted on top of particle steps/histogram of particle steps

### Calculating the diffusion coefficients

The diffusion coefficients can be determined in two different ways:

1. By fitting a 2D Gaussian distribution to a 2D histogram of the particles' steps --> use notebook `2D_fit_diffusion_folder.ipynb`
2. By parametrically determining the coefficient --> use notebook `2D_parametric_diffusion_folder.ipynb`

---

## Additional analysis

The trajectories can be analysed further with the `trajectory_analysis.ipynb`. For better results (less noise) use several trajectory files (of the same particle type on the same type of track). For analysis of anisotropic movement, one should provide the trajectories calculated from files where the tracks are aligned along one of the axes (for now the only way to do this is manually). The notebook does the following:

1. Plots coordinates vs time
2. Plots histrograms of maximal trajectory span for both coordinates on the same plot
3. Plots histograms of net movement (final - initial position) for both coordinates on the same plot
4. Plots MSD for both dimensions and the overall MSD
5. Plots a particle trajectory (y vs x) and colors it time dependantly. The particle is chosen by the user.
6. Calculates mean step sizes (a '.steps.csv' has to be provided instead of a '.tray.csv')

The notebooks `simulate_random_walk.ipynb` and `00-generate-sythetic-2d-normal-dist.ipynb` were created to generate synthetic random walker samples (anisotropy) in order to compare them to experimental results.

---

## Instructions for running the demo

### Undrifting

1. Run the `drift_correct_folder.ipynb` notebook using the following input:
   - `in_folder`: path to the `example\undrifting` folder (containing aligned multichannel movie)

#### Expected output

The expected output is a `.tif` file with the following basename:  
`<timestamp>_WT_multichannel_aligned_drift_corrected.tif`

The output file can be opened in ImageJ. Check if the drift has been successfully corrected.

#### Expected runtime

10 seconds.

### Diffusion coefficient calculation by fitting 

1. Run the `2D_fit_diffusion_folder.ipynb` notebook using the following input:
   - `in_folder`: path to the `example` folder (containing aligned and undrifted multichannel movie)
   - leave other parameters as they are 

#### Expected output

The expected outputs are the following:

1. A tif-stack with one channel containing only the walkers localized on the tracks
2. An '.hdf5' file containing particle localizations
3. A '.yaml' file from Picasso particle localization
4. A '.tray.csv' file containing particle IDs, particle coordinates and the length of their trajectories
5. A '.steps.csv' file containing the particle IDs, particle step sizes in each dimension and the length of their steps
6. A '.diff' file containing various diffusion data calculated in the pipeline
7. A 2D-Gaussian contour plotted on top of a histogram of particle steps
8. A 'diff_info.csv' file 

#### Expected runtime

45 seconds.

### Parametric diffusion coefficient calculation

1. Run the `2D_parametric_diffusion_folder.ipynb` notebook using the following input:
   - `in_folder`: path to the `example` folder (containing aligned and undrifted multichannel movie)
   - leave other parameters as they are 

#### Expected output

The expected outputs are the following:

1. A tif-stack with one channel containing only the walkers localized on the tracks
2. An '.hdf5' file containing particle localizations
3. A '.yaml' file from Picasso particle localization
4. A '.tray.csv' file containing particle IDs, particle coordinates and the length of their trajectories
5. A '.steps.csv' file containing the particle IDs, particle step sizes in each dimension and the length of their steps
6. A '.diff' file containing various diffusion data calculated in the pipeline
7. A 2D-Gaussian contour plotted on top of a particles' step scatter plot
8. A 'diff_info.csv' file 

#### Expected runtime

45 seconds.

### Analysis of trajectories

1. Run the `trajectory_analysis.ipynb` notebook using the following input:
   - `in_folder`: path to the `example\analysing_rotated_trajectories` folder (containing multiple trajectories that have been generated from movies with tracks manually   aligned to the y-axis)
   - skip cell number 3 when analysing --> this one is meant to calculate average step sizes of the particles and cannot be run on trajectory files
   - leave other parameters as they are 

#### Expected output

The expected outputs are the following:

1. Plot of coordinates vs time
2. Histrogram of maximal trajectory span for both coordinates on the same plot
3. Histogram of net movement for both coordinates on the same plot
4. MSD plot for both dimensions and the overall MSD
5. A linearized MSD plot with a fitted linear function
6. Plot of a specific particle trajectory (y vs x)

#### Expected runtime

45 seconds.

---

## Notes

- The code was developed and tested on **Windows**.
- Long file paths may cause errors on Windows; keep directory paths short when possible.
- Fiji is called externally and **cannot run fully headless** for the F4DR plugin.

---

## Citation

If you use this code in academic work, please cite the associated publication:

**De-novo design of a random protein walker**
doi: https://doi.org/10.1101/2025.09.29.677966

---

## Contact

For questions or issues, please contact the repository maintainer or open an issue on GitHub.

