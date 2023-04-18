import numpy as np
import pandas as pd

data_df = pd.read_csv('test_data.csv')

z_max = 40000 # the upper bound for the positiion of the 2nd detector
resolution = 1 # m
half_detector_height = 2 # m
histogram_array = np.zeros(int(z_max * 1/resolution))

def make_histogram_array():
    # data_df = data_df.reset_index()  # make sure indexes pair with number of rows
    for _, row in data_df.iterrows():
        dK = row['decay lenght K+']
        a1 = row['angle pi+']
        a2 = row['angle pi0']

        # distance (on the z axis) when paricle leaves detectable range (on the y axis)
        #       /|
        #      / | 2m
        # ____/__|         2/z = tan(a) --> z = 2/tan(a)
        #  dK  z

        z1 = half_detector_height/np.tan(a1)
        z2 = half_detector_height/np.tan(a2)

        # the Pions don't exist before the decay of the Kaon
        lower_end = int(np.round(dK * 1/resolution,0))

        # both Pions need to be detected so stop when the first one leave the detectable range
        upper_end = int(np.round(np.min([dK+z1, dK+z2, z_max]) * 1/resolution)) 

        histogram_array[lower_end:upper_end] += 1

    return histogram_array

print(max(histogram_array))
    
