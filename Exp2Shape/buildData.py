import os
import glob
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops_table
from scipy.stats import zscore
from tqdm import tqdm


path = "CRC_Dataset"
patients = len(glob.glob(f'P*'))
cells_dfs = []
for patient in patients:
    channel_files = glob.glob(f'{path}patient/C*.tif')
    labeled_image = imread(f'{path}patient/Segmented.tif')
    # Initialize a dictionary to store the channel data
    channel_data = {}

    # Load each channel file and store the data in the dictionary
    for channel_file in channel_files:
        channel_name = channel_file.split()[-1].split('.')[0]
        channel_image = imread(os.path.join(path, channel_file))
        channel_data[channel_name] = channel_image
    print('Now it will take a while..')
    # Extract cell properties from the segmented image
    cell_props = regionprops_table(labeled_image, properties=[
        'label','area', 'centroid', 
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "equivalent_diameter_area",
        "area_convex",
        "extent",
        "orientation",
        "perimeter_crofton",
        "solidity"

    ])
    cell_props_df = pd.DataFrame(cell_props)
    normalized_data = pd.DataFrame()
    for channel_name, channel_image in tqdm(channel_data.items()):
        cell_intensities = regionprops_table(labeled_image, intensity_image=channel_image, properties=['mean_intensity'])
        cell_intensities_df = pd.DataFrame(cell_intensities)
        ### Normalize protein expression values by cell size
        cell_intensities_df['normalized_intensity'] = cell_intensities_df['mean_intensity'] / cell_props_df['area']
        ### Apply arctan transformation
        cell_intensities_df['arctan_intensity'] = np.arctan(cell_intensities_df['normalized_intensity'])
        # Calculate Z-scores
        cell_intensities_df['z_score'] = zscore(cell_intensities_df['arctan_intensity'])
        # Add the normalized values to the DataFrame
        normalized_data[channel_name] = cell_intensities_df['z_score']
    # Merge the cell properties and normalized data into a single DataFrame
    cells_data = pd.concat([normalized_data,cell_props_df], axis=1)
    cells_dfs.append(cells_data)
# Save the final data to a CSV file
cellsData = pd.concat(cells_dfs, axis = 0)
cellsData.to_csv('cellsDataWithShape.csv', index=False)