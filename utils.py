import numpy as np
import pandas as pd
import skimage.io as skio
import skimage
import matplotlib.pyplot as plt
import tifffile
import numpy as np


def take_only_walkers_on_fibre(fibre, walker):
    """Take only the part of the walker chanel that is close to the fibres"""
    fibre = substract_background(fibre, radius=100)
    fibre = skimage.exposure.rescale_intensity(fibre, in_range='image', out_range='uint8')
    
    thresh = skimage.filters.threshold_otsu(fibre)
    binary = fibre > thresh
    #extend the treshold region
    for i in range(1):
        binary= skimage.filters.gaussian(binary)
        binary = skimage.morphology.binary_dilation(binary)
    #blur borders
    binary=skimage.filters.gaussian(binary)
    #plt.imshow(binary)
    #plt.show()
    walker = substract_background(walker, radius=20)
    take = walker*binary
    take = skimage.exposure.rescale_intensity(take, in_range='image', out_range='uint8')

    return take

#taken from https://forum.image.sc/t/background-subtraction-in-scikit-image/39118/4
def substract_background(image, radius=50, light_bg=False):
        from skimage.morphology import white_tophat, black_tophat
        str_el = skimage.morphology.rectangle(radius, radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
        if light_bg:
            return  black_tophat(image, str_el)
        else:
            return  white_tophat(image, str_el)

def get_steps_from_df(df):
    """Input is dataframe with x and y columns and frame. 
    Returns the difference of coordinates (dx, dy) between frames"""
    #get the diffusion
    # diffusion equation <x^2> = 2*D*t # two for 2D
    dx = np.diff(df.x.values)
    dy = np.diff(df.y.values)
    dframe = np.diff(df.frame.values)
    #print(df.frame.values)
    return pd.DataFrame(dict(dframe=dframe, dx=dx, dy=dy))

def get_diff_from_steps(steps_df, dim=1, dt=1, step_to_length=1):
    """Data frame must have dx and dy field. 
      Returns diffusion coefficient, optionally scaled by dimension (1, 2, 3), time units and length units (step_to_length)"""
    return np.mean(steps_df.dx**2+steps_df.dy**2)/(2*dim*dt)*step_to_length*step_to_length

def write_yaml(data, filename):
    import yaml
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file)

# Generate random colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
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

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap