"""Utility methods for 2D tracking. (c) ajasja.ljubetic@ki.si, liza.ulcakar@ki.si"""

import numpy as np
import pandas as pd
import skimage.io as skio
import skimage
import matplotlib.pyplot as plt
import tifffile
import numpy as np
import os
from pathlib import Path
import trackpy as tp
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle


# Taken form https://medium.com/@DahlitzF/how-to-create-your-own-timing-context-manager-in-python-a0e944b48cf8
from contextlib import contextmanager
from time import time


@contextmanager
def timing(description: str) -> None:
    start = time()
    yield
    elapsed_time = time() - start

    print(f"{description}: {elapsed_time*1000*1000:.2f} us")


def take_only_walkers_on_fibre(
    fibre,
    walker,
    out_dir=None,
    fibre_background_radius=10,
    min_fibre_size=20,
    fibre_background_light=True,
    fibre_extend_radius=4,
):
    """Take only the part of the walker chanel that is close to the fibres"""

    fibre = skimage.exposure.rescale_intensity(fibre, in_range="image", out_range="uint8")
    fibre = subtract_background(fibre, radius=fibre_background_radius, light_bg=fibre_background_light)

    #thresh = skimage.filters.threshold_otsu(fibre)
    thresh = skimage.filters.threshold_triangle(fibre)
    binary = fibre > thresh

    # Remove small parts
    binary = skimage.morphology.remove_small_objects(binary, min_fibre_size)

    # extend the treshold region
    # binary= skimage.filters.gaussian(binary)
    binary = skimage.morphology.binary_dilation(binary, footprint=skimage.morphology.disk(fibre_extend_radius))

    #plt.imsave(f'{out_dir}_mask.png', binary)
    #plt.show()

    # blur borders
    binary = skimage.filters.gaussian(binary)

    walker = subtract_background(walker, radius=10)

    take = walker * binary

    # plt.imshow(take)
    # plt.show()

    return take


def take_only_walkers_on_fibre_trajectory(
    in_file,
    out_dir=None,
    walker_channel=0,
    fibre_chanel=3,
    fibre_background_radius=10,
    min_fibre_size=20,
    fibre_background_light=True,
    fibre_extend_radius=4,
):
    """Takes an input file or a stack in the form of TYXC and saves it to out file. Writes the shape of TYX""" 

    if isinstance(in_file, str):
        # Could also be a path object, but that would now fail. perhaps change to file exists, or test if in_file is an array
        stack = skio.imread(in_file)
    else:
        stack = in_file

    dims = stack.shape
    # skip the channel?
    new_dims = (dims[0], dims[1], dims[2])
    out = np.zeros(new_dims, dtype=np.uint16)
    for i in range(dims[0]):
        if i % 100 == 0:
            print(f"Processing frame {i+1}/{dims[0]}")
        frame_fibre = stack[i, :, :, fibre_chanel]
        frame_walker = stack[i, :, :, walker_channel]
        new = take_only_walkers_on_fibre(
            frame_fibre,
            frame_walker,
            out_dir=out_dir,
            fibre_background_radius=fibre_background_radius,
            min_fibre_size=min_fibre_size,
            fibre_background_light=fibre_background_light,
            fibre_extend_radius=fibre_extend_radius,
        )
        out[i] = new
    out_file = f'{out_dir}.tif'

    tifffile.imwrite(
        out_file,
        out,
        ome=True,
        dtype=np.uint16,
        photometric="minisblack",
        metadata={"axes": "TYX"},
    )


# taken from https://forum.image.sc/t/background-subtraction-in-scikit-image/39118/4
def subtract_background(image, radius=50, light_bg=False):
    from skimage.morphology import white_tophat, black_tophat

    str_el = skimage.morphology.rectangle(
        radius, radius
    )  # you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)


def fit_single_molecules(out_dir, basename,
                          fit_method = "lq",
                          box_side_length = 9,
                          drift = 0,
                          min_gradient = 1000,
                          px_to_nm = 72):
    """Fit the positions in the walker channel. Currently uses picasso that is expected to be available in the path of the environment.
       Returns the            
    """
    #os.environ['HDF5_DISABLE_VERSION_CHECK']='0'
    #TOOD add out parameter
    cmd = f"python -m picasso localize {out_dir/basename} --fit-method {fit_method} --box-side-length {box_side_length}  --gradient {min_gradient} --drift {drift} --pixelsize {px_to_nm}"
    print(cmd)
    os.system(cmd)
    basename_noext, ext = os.path.splitext(basename)
    # Hack -- for now just rename the out file. This is dangerous in multithreaded environment.
    out_locs = out_dir / (basename_noext + "_locs.hdf5")
    new_suffix = f"__locs_{fit_method}_box{box_side_length}_grad{min_gradient}_drift{drift}.hdf5"
    new_out_locs = out_dir / (basename_noext + new_suffix)
    #print(new_out_locs)
    if os.path.exists(new_out_locs):
        os.remove(new_out_locs)
    if os.path.exists(new_out_locs.with_suffix(".yaml")):
        os.remove(new_out_locs.with_suffix(".yaml"))
    out_locs.with_suffix(".yaml").rename(new_out_locs.with_suffix(".yaml"))
    out_locs.rename(new_out_locs)
    return new_out_locs


def link_trajectory(locs_path,
                          out_dir, 
                          basename_noext,
                          max_link_displacement_px = 2,
                          max_gap = 2,
                          min_tray_length = 3,
                         ):
    """Takes a locs file and links the trajectories. Exclude very short trays.
    Must be larger or equal min_tray_length 
    Returns a pandas dataframe of trajectories, the path of the trajectories, pandas dataframe of steps, path of the steps.     
    """
    locs = pd.read_hdf(locs_path, "locs")
    locs["mass"] = locs.photons

    tray = tp.link(locs, search_range=max_link_displacement_px, memory=max_gap)

    # count the length of trajectories
    tray_by_particle = tray.groupby(["particle"])
    tray["length"] = tray_by_particle["particle"].transform("count")


    # Exclude very short trays
    tray = tray.query(f"length>={min_tray_length}")

    steps = tray.groupby(["particle"]).apply(get_steps_from_df)
    steps["step_len"] = np.sqrt(steps.dx**2 + steps.dy**2)

    tray_out = Path(f'{out_dir}/{basename_noext}_link{max_link_displacement_px}_maxgap{max_gap}_traylen{min_tray_length}.tray.csv')
    tray.to_csv(tray_out)
    steps_out = Path(f'{out_dir}/{basename_noext}.steps.csv')
    steps.to_csv(steps_out)
    return tray, tray_out, steps, steps_out


def get_steps_from_df(df):
    """Input is dataframe with x and y columns and frame.
    Returns the difference of coordinates (dx, dy) between frames"""
    # get the diffusion
    # diffusion equation <x^2> = 2*D*t # two for 2D
    dx = np.diff(df.x.values)
    dy = np.diff(df.y.values)
    dframe = np.diff(df.frame.values)
    # print(df.frame.values)
    return pd.DataFrame(dict(dframe=dframe, dx=dx, dy=dy))


'''def get_diff_from_steps(steps_df, dim=1, dt=1, step_to_length=1):
    """Data frame must have dx and dy field. 
      Returns diffusion coefficient, optionally scaled by dimension (1, 2, 3), time units and length units (step_to_length)"""
    return np.mean(steps_df.dx**2+steps_df.dy**2)/(2*dim*dt)*step_to_length*step_to_length'''


def write_yaml(data, filename):
    import yaml

    with open(filename, "w") as yaml_file:
        yaml.dump(data, yaml_file)


# Generate random colormap
def rand_cmap(
    nlabels, type="bright", first_color_black=True, last_color_black=False, verbose=True
):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if type not in ("bright", "soft"):
        print('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print("Number of labels: " + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == "bright":
        randHSVcolors = [
            (
                np.random.uniform(low=0.0, high=1),
                np.random.uniform(low=0.2, high=1),
                np.random.uniform(low=0.9, high=1),
            )
            for i in range(nlabels)
        ]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(
                colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2])
            )

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors = [
            (
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
                np.random.uniform(low=low, high=high),
            )
            for i in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list(
            "new_map", randRGBcolors, N=nlabels
        )

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(
            ax,
            cmap=random_colormap,
            norm=norm,
            spacing="proportional",
            ticks=None,
            boundaries=bounds,
            format="%1i",
            orientation="horizontal",
        )

    return random_colormap


# This is taken from scipy 1.13
def fit_2d_normal(self, x, fix_mean=None, fix_cov=None):
    """Fit a multivariate normal distribution to data.

    Parameters
    ----------
    x : ndarray (m, n)
        Data the distribution is fitted to. Must have two axes.
        The first axis of length `m` represents the number of vectors
        the distribution is fitted to. The second axis of length `n`
        determines the dimensionality of the fitted distribution.
    fix_mean : ndarray(n, )
        Fixed mean vector. Must have length `n`.
    fix_cov: ndarray (n, n)
        Fixed covariance matrix. Must have shape `(n, n)`.

    Returns
    -------
    mean : ndarray (n, )
        Maximum likelihood estimate of the mean vector
    cov : ndarray (n, n)
        Maximum likelihood estimate of the covariance matrix

    """
    # input validation for data to be fitted
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("`x` must be two-dimensional.")

    n_vectors, dim = x.shape

    # parameter estimation
    # reference: https://home.ttic.edu/~shubhendu/Slides/Estimation.pdf
    if fix_mean is not None:
        # input validation for `fix_mean`
        fix_mean = np.atleast_1d(fix_mean)
        if fix_mean.shape != (dim,):
            msg = (
                "`fix_mean` must be a one-dimensional array the same "
                "length as the dimensionality of the vectors `x`."
            )
            raise ValueError(msg)
        mean = fix_mean
    else:
        mean = x.mean(axis=0)

    if fix_cov is not None:
        # input validation for `fix_cov`
        fix_cov = np.atleast_2d(fix_cov)
        # validate shape
        if fix_cov.shape != (dim, dim):
            msg = (
                "`fix_cov` must be a two-dimensional square array "
                "of same side length as the dimensionality of the "
                "vectors `x`."
            )
            raise ValueError(msg)
        # validate positive semidefiniteness
        # a trimmed down copy from _PSD
        s, u = scipy.linalg.eigh(fix_cov, lower=True, check_finite=True)
        eps = _eigvalsh_to_eps(s)
        if np.min(s) < -eps:
            msg = "`fix_cov` must be symmetric positive semidefinite."
            raise ValueError(msg)
        cov = fix_cov
    else:
        centered_data = x - mean
        cov = centered_data.T @ centered_data / n_vectors
    return mean, cov

class MultivariateNormal(object):
    def __init__(self):
        self.u_ = None
        self.sig_ = None
    @staticmethod
    def redimx(x): return x[...,np.newaxis] if x.ndim == 2 else x
    def fit(self, x):
        x = self.redimx(x)
        self.u_ = x.mean(0)
        self.sig_ = np.einsum('ijk,ikj->jk', x-self.u_, x-self.u_)/ (x.shape[0]-1)
    def prob(self, x):
        x = self.redimx(x)
        f1 = (2*np.pi)**(-self.u_.shape[0]/2)*np.linalg.det(self.sig_)**(-1/2)
        f2 = np.exp((-1/2)*np.einsum('ijk,jl,ilk->ik', x-self.u_, np.linalg.inv(self.sig_), x-self.u_))
        return f1*f2

def gaussian_2d(xy, x0, y0, sigmax, sigmay, sigmaxy):
    if xy.shape[-1] != 2 or len(xy.shape) != 2:
        raise ValueError("XY data should have shape (n, 2).")
    mu = np.array([x0, y0])
    sigma = np.array([[sigmax, sigmaxy], [sigmaxy, sigmay]])
    normalization = (((2 * np.pi) ** 2) * np.abs(np.linalg.det(sigma))) ** (-1 / 2)
    bell = np.exp(-0.5 * np.sum(((xy - mu) @ np.linalg.inv(sigma)) * (xy - mu), axis=1))
    return normalization * bell

def cov_to_axes_and_rotation(cov, sorted=True):
    """"Takes a covariance matrix and returns the principle axes and a rotation"""
    (e1, e2), eigen_vec = np.linalg.eig(cov)
    V1,V2 = eigen_vec.T
    
    # Eigenvectors are assumed to be unit and orthogonal
    # print(np.linalg.norm(V1))
    e1 = np.real(e1) # sometimes tiny imaginary components are returned. 
    e2 = np.real(e2) # sometimes tiny imaginary components are returned. 
    if np.isclose(e1, e2):
        # the angle is not well defined
        theta = 0

    else:
        # Are eigenvectors always normalized? 
        theta = np.degrees(np.arctan2(V1[1], V1[0]))
    #if theta>90 and theta <180:
    #     theta = 180 - theta
    #print(theta)

    if sorted: #make sure e1 is always the largest
        if e2>e1:
            (e1, e2) = (e2, e1)
            theta = theta + 90

    return e1, e2, np.mod(theta, 360)

def walker_traj_movie_BW(
    tif_file,
    tray,
    out_dir,
    fps=10, 
    range_min=0, 
    range_max=200, 
    frame_to_s=0.25,
    scale_bar_length=1, 
    pixelsize=0.072,
):
    basename = os.path.basename(tif_file)
    basename_noext, _ = os.path.splitext(basename)
    stack = skio.imread(tif_file)

    fig = plt.figure("FRAMES", dpi=300, frameon=False)
    ax = fig.add_subplot(111)
    frame_for_shape = ax.imshow(stack[0].astype('uint16'), cmap = 'Greys_r')
    image_array = frame_for_shape.get_array().data  # Retrieve the image array
    image_height, image_width = image_array.shape[:2]
    subset = stack[range_min:range_max+1]
    v_min = np.min(subset) #probably 0 for all cases anyway
    #v_max = np.max(subset)
    v_max = np.mean(np.partition(subset.ravel(), -100)[-100:])
    
    cmap_bright = plt.colormaps.get_cmap("Set1")
    set1_dict = { particle:cmap_bright(i%9) for i, particle in enumerate(list(set(tray.particle)))}
    ims = [] 

    for n in range(range_min, range_max):
        frame = ax.imshow(stack[n].astype('uint16'), cmap = 'Greys_r', vmin = v_min, vmax = v_max)
        number = ax.annotate(f'{round((n-range_min)*frame_to_s)} s',(1,5), color='white', fontsize='16')
        scale_bar = Rectangle(
            (image_width - 15, image_height - 4),  # Position (10 pixels from left, 20 pixels from bottom)
            int(scale_bar_length / pixelsize),  # Width of the scale bar in pixels
            1,  # Height of the scale bar in pixels
            color='white'
        )
        ax.add_patch(scale_bar)

        # Add text for the scale bar length
        scale_text = ax.text(
            image_width - 12, image_height - 5,  # Position slightly above the scale bar
            f'{scale_bar_length} µm',
            color='white',
            fontsize=10
        )

        artist_obj = [frame, number, scale_bar, scale_text]

        artist_obj.append(frame)
        artist_obj.append(number)
        artist_obj.append(scale_bar)
        artist_obj.append(scale_text)
        if n in list(set(tray.frame)):
            for k in list(set(tray[(tray["frame"]==n)].particle)):
                bright_color = set1_dict[k]
                line, = ax.plot(  # Note the comma to unpack the single Line2D object
                    tray[(tray["frame"] <= n) & (tray["particle"] == k)].x,
                    tray[(tray["frame"] <= n) & (tray["particle"] == k)].y,
                    linewidth=1,
                    color=bright_color
                )
                plt.axis('off')
                artist_obj.append(line)
                x = tray[(tray["frame"] == n) & (tray["particle"] == k)].x.values[0]
                y = tray[(tray["frame"] == n) & (tray["particle"] == k)].y.values[0]

        # Add particle ID label near position
                particle_id_text = ax.text(x-10, y, str(k), fontsize=6, color=bright_color)
                artist_obj.append(particle_id_text)    

        ims.append(artist_obj)

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat_delay=500)

    output_path = f'{out_dir}\{basename_noext}_trajectories.mp4'

    ani.save(output_path, fps=fps)
    plt.close(fig)

def colorize_channel(channel_stack, color="green", clip_percentiles=(1, 99)):
    """
    Map a 3D channel stack (frames, H, W) to RGB using a specified color.
    Returns float32 RGB array in [0,1], shape (frames, H, W, 3)
    
    color options: "red", "green", "blue", "magenta", "cyan", "yellow"
    """
    # Normalize across entire channel
    low, high = np.percentile(channel_stack, clip_percentiles)
    norm = np.clip((channel_stack.astype(np.float32) - low) / (high - low + 1e-8), 0, 1)

    # Prepare RGB array
    rgb = np.zeros((*norm.shape, 3), dtype=np.float32)

    # Map color
    if color == "red":
        rgb[..., 0] = norm
    elif color == "green":
        rgb[..., 1] = norm
    elif color == "blue":
        rgb[..., 2] = norm
    elif color == "magenta":
        rgb[..., 0] = norm
        rgb[..., 2] = norm
    elif color == "cyan":
        rgb[..., 1] = norm
        rgb[..., 2] = norm
    elif color == "yellow":
        rgb[..., 0] = norm
        rgb[..., 1] = norm
    else:
        raise ValueError("Unsupported color")

    return rgb

def stretch_contrast(channel_stack, low_pct=1, high_pct=99): 
    """ Stretch contrast for a 3D channel stack (frames, H, W) using min/max-based clipping """ 
    low, high = np.percentile(channel_stack, (low_pct, high_pct)) 
    stretched = np.clip((channel_stack - low) / (high - low + 1e-8), 0, 1) 
    return stretched

def walker_traj_movie_RGB(
    tif_stack,
    out_dir,
    tray=None,
    draw_traj=False,
    walker=0,
    fibre=3,
    fps=10,
    walker_color='magenta',
    fibre_color='green',
    alpha_walker=0.9,
    alpha_fibre=0.5,
    walker_low_pct=60,
    walker_high_pct=99.9,
    fibre_low_pct=10,
    fibre_high_pct=99, 
    range_min=0, 
    range_max=200, 
    frame_to_s=0.25,
    scale_bar_length=1, 
    pixelsize=0.072,    
):
    basename = os.path.basename(tif_stack)
    basename_noext, _ = os.path.splitext(basename)
    stack = skio.imread(tif_stack)

    stack = skio.imread(tif_stack)
    walker_channel = stack[:, :, :, walker].astype(np.float32)
    fibre_channel = stack[:, :, :, fibre].astype(np.float32)
    fibre_channel = np.max(fibre_channel) - fibre_channel  #inverts LUT
    fibre_channel = colorize_channel(fibre_channel, color=fibre_color)
    walker_channel = colorize_channel(walker_channel, color=walker_color)
 
    fig = plt.figure("FRAMES", dpi=300, frameon=False)

    ax= fig.add_subplot(111)

    frame_for_shape = ax.imshow(walker_channel[0].astype(np.float32))
    image_array = frame_for_shape.get_array().data  # Retrieve the image array
    image_height, image_width = image_array.shape[:2]
    walker_channel = stretch_contrast(walker_channel, low_pct=walker_low_pct, high_pct=walker_high_pct)
    fibre_channel = stretch_contrast(fibre_channel, low_pct=fibre_low_pct, high_pct=fibre_high_pct)

    cmap_bright = plt.colormaps.get_cmap("Set1")
    ims = [] 

    for n in range(range_min, range_max): 
        frame_fibre = ax.imshow(fibre_channel[n], alpha=alpha_fibre)
        frame_walker = ax.imshow(walker_channel[n], alpha=alpha_walker) 
        number = ax.annotate(f'{round((n-range_min)*0.25)} s',(1,5), color='white', fontsize='16')
        scale_bar = Rectangle(
            (image_width - 15, image_height - 4),  # Position (10 pixels from left, 20 pixels from bottom)
            int(scale_bar_length / pixelsize),  # Width of the scale bar in pixels
            1,  # Height of the scale bar in pixels
            color='white'
        )
        ax.add_patch(scale_bar)

        # Add text for the scale bar length
        scale_text = ax.text(
            image_width - 12, image_height - 5,  # Position slightly above the scale bar
            f'{scale_bar_length} µm',
            color='white',
            fontsize=10
        )

        artist_obj = []

        artist_obj.append(frame_fibre)
        artist_obj.append(frame_walker)
        artist_obj.append(number)
        artist_obj.append(scale_bar)
        artist_obj.append(scale_text)

        if draw_traj == True:
            set1_dict = { particle:cmap_bright(i%9) for i, particle in enumerate(list(set(tray.particle)))}
            if n in list(set(tray.frame)):
                for k in list(set(tray[(tray["frame"]==n)].particle)):
                    bright_color = set1_dict[k]
                    line, = ax.plot(  # Note the comma to unpack the single Line2D object
                        tray[(tray["frame"] <= n) & (tray["particle"] == k)].x,
                        tray[(tray["frame"] <= n) & (tray["particle"] == k)].y,
                        linewidth=1,
                        color=bright_color
                    )
                    plt.axis('off')
                
                    artist_obj.append(line)

        ims.append(artist_obj)

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat_delay=500)

    output_path = f'{out_dir}/{basename_noext}_trajectories_4.mp4'

    ani.save(output_path, fps=fps)
    plt.close(fig)