import numpy as np
import monai
from multiprocessing import Pool





def calculate_adc(dwimin_data,dwimax_data,voxel,b_values):
    
    '''
     calculate_adc function
        In:
        - dwimin_data: 3D numpy array representing the DWI image at minimum b-value
        - dwimax_data: 3D numpy array representing the DWI image at maximum b-value
        - voxel: A tuple representing the coordinates (i, j, k) of a voxel in the image
        - b_values: Tuple of two floats, representing the b-values for dwimin_data and dwimax_data

        Out:
        - Returns the Apparent Diffusion Coefficient (ADC) value for the given voxel. 
        - If signal intensities are invalid, returns 0.
    '''
    
    
    # Get signal intensities from all images
    s_min = dwimin_data[voxel]
    s_max = dwimax_data[voxel]
    #print(s_min)
    # Check if signal is positive and non-zero
    if s_min > 0 and s_max > 0 and s_min>=s_max:
        # Take natural logarithm of signal intensities
        ln_smin = np.log(s_min)
        ln_smax = np.log(s_max)

        
        # Define b-values as a numpy array
        b_values = np.array(b_values)
        
        # Define log-signal intensities as a numpy array
        ln_s_values = np.array([ln_smin, ln_smax])
        
        # Perform linear regression
        #slope, intercept, r_value, p_value, std_err = linregress(b_values, ln_s_values)
        
        # Calculate ADC value as negative slope
        #adc = -slope
        adc=(-(ln_s_values[1]-ln_s_values[0])/(b_values[1]-b_values[0]))*1e6
        
        # Return ADC value
        return adc
    else:
        # Return zero if signal is invalid
        return 0



def get_save_adc(dwi_path,dwi_bval_pos):
    '''
    In:
    - dwi_path: String, the path to the DWI image to be processed
    - dwi_bval_pos: List of b-values associated with each volume in the DWI sequence

    Out:
    - adc_map: 3D numpy array representing the ADC map calculated from the DWI image
    - dwi_img: 4D numpy array of the original DWI image loaded from dwi_path
    
    '''

    
    
    dwi_img=monai.transforms.LoadImage(image_only=True)(dwi_path)

    if dwi_img.ndim>3:
    
        max_bval=np.argmax(dwi_bval_pos)
        min_bval=np.argmin(dwi_bval_pos)
        
        # Get image data as numpy arrays
        dwimin_data = dwi_img[:,:,:,min_bval]
        dwimax_data = dwi_img[:,:,:,max_bval]

        # Get image dimensions
        nx, ny, nz = dwimin_data.shape

        # Create a list of all voxels as tuples of indices
        voxels = [(i, j, k) for i in range(nx) for j in range(ny) for k in range(nz)]
        
        b_values=(dwi_bval_pos[min_bval],dwi_bval_pos[max_bval])
        
        #print(b_values)
        args = [(dwimin_data, dwimax_data, voxel, b_values) for voxel in voxels]


        # # Create a Pool object with 4 worker processes
        # pool = Pool(4)

        # Use the map method to apply the calculate_adc function to all voxels in parallel
        #with multiprocessing.Pool(processes=8) as pool:
        with Pool(8) as pool:
            adc_values = pool.starmap(calculate_adc, args)
        #adc_values = calculate_adc(dwimin_data,dwimax_data,voxels,b_values)

        # Convert the list of ADC values to a numpy array with the same shape as the images
        adc_map = np.array(adc_values).reshape((nx, ny, nz))
        
        return adc_map,dwi_img
    else:
        return None,dwi_img