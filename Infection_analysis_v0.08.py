# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:42:48 2023

@author: leaga
Credits to: @DigitalSreeni
"""

import czifile
import numpy as np
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage import io, color, measure, img_as_ubyte, exposure
from skimage.io import imsave
from skimage.segmentation import clear_border
from skimage.measure import label
import pyclesperanto_prototype as cle
import pandas as pd
from skimage import io, filters, img_as_ubyte
#from ij.plugin import ZProjector as zp


# =============================================================================
# Please specify your channel and pixel size
# =============================================================================

DAPI = 0
NP = 2
pixels_to_um = 0.2930000 #size of pixel

# =============================================================================
# Please specify your conditions for plotting
# =============================================================================

# Define condition names for labels and actual condition strings
# if you do not have 2 belonging conditions (e.g. sex), you can name them here 
# Important: Make sure that the value is exaclty like this in your filename!

condition_definitions = {
    'Female 6h': 'Female_H7N7_MOI0.05_6h',
    'Male 6h': 'Male_H7N7_MOI0.05_6h',
    'Female 24h': 'Female_H7N7_MOI0.05_24h',
    'Male 24h': 'Male_H7N7_MOI0.05_24h',
}


###############################################################################
# Functions
###############################################################################

# =============================================================================
# The following functions need to be altered for different users as the naming
# of your files is important here 
# =============================================================================     
            
def ConditionFinder (filename):
    condition = filename.split("_")[1:5]
    condition = '_'.join(condition[0:])
    
    return condition

    
def RoundFinder(filename):
    cell_round = filename.split("_")[0]
    
    return cell_round

# =============================================================================
# The following functions do not need to be altered for different users
# =============================================================================  

def enhance_contrast(img):
    # Convert the image to grayscale if it's not already
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization
    img = cv2.convertScaleAbs(img)
    img_equalized = cv2.equalizeHist(img)

    return img_equalized

def ClaheContrast(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1

def NucleiRecognition (img, int_mean, filename, image_path, results_path):
    
    #img = ClaheContrast(img)
    
    #img = img - int_mean
    #img = img - 0.3 * img 
    
    
    img = cv2.convertScaleAbs(img) #for dist_transform uint 8 is needed with 1 Channel
    #plt.imshow(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #grayscale to color conversion, needed for watershed

    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (5, 5), 3)
    blurred_img_uint8 = cv2.convertScaleAbs(blurred_img)

    # Thresholding on the blurred image
    dynamic_threshold_blurred = np.mean(img) + 1 * np.std(blurred_img_uint8)
    ret, thresh = cv2.threshold(blurred_img_uint8, dynamic_threshold_blurred, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((1,1),np.uint8)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    sure_bg = cv2.dilate(closing,kernel,iterations=10) # dilating pixes a few times increases cell boundary to background. --> This way whatever is remaining for sure will be background. 
    
    
    
    dist_transform = cv2.distanceTransform(closing,cv2.DIST_L2,0) # Finding sure foreground area using distance transform and thresholding
    #intensities of the points inside the foreground regions are changed to 
    #distance their respective distances from the closest 0 value (boundary)
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0) #Let us threshold the dist transform by starting at 1/2 its max value.
    
    # Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    
    #Now we create a marker and label the regions inside. 
    # For sure regions, both foreground and background will be labeled with positive numbers.
    # Unknown regions will be labeled 0. 
    #For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)
    #One problem rightnow is that the entire background pixels is given value 0.
    #This means watershed considers this region as unknown.
    
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.
    
    #Now we are ready for watershed filling. 
    markers = cv2.watershed(img_RGB,markers) #expects the source image to have a data type of CV_8UC3 (an 8-bit unsigned 3-channel image, i.e., a color image) and the destination image to have a data type of CV_32SC1 (a 32-bit signed single-channel image, typically used for labeling)
    #The boundary region will be marked -1
    #Let us color boundaries in red 
    img_RGB[markers == -1] = [255, 0, 0]
    img2 = color.label2rgb(markers, bg_label=0, bg_color=[0, 0, 0])
    #plt.imshow(img2)
    
    # Save the img2 as a TIFF image
    tiff_filename = filename + 'DAPI_labelled.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    img2_uint8 = img_as_ubyte(img2)
    # Save the TIFF image
    io.imsave(tiff_path, img2_uint8)
    
    # Save the img with overlay as a TIFF image
    tiff_filename = filename +  '_DAPI_original_label.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    img_uint8 = img_as_ubyte(img_RGB)
    # Save the TIFF image
    io.imsave(tiff_path, img_uint8)
    
    # Save the threshold image
    tiff_filename = filename + '_DAPI_threshold.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    thresh_uint8 = img_as_ubyte(thresh)
    # Save the TIFF image
    io.imsave(tiff_path, thresh_uint8)
    
    # Save the sure_fg
    tiff_filename = filename + '_sureFG_threshold.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    sure_fg_uint8 = img_as_ubyte(sure_fg)
    # Save the TIFF image
    io.imsave(tiff_path, sure_fg_uint8)
    
    # Count the number of unique labels excluding the background label (-1)
    unique_labels = np.unique(markers)
    count = 0

    # Iterate through each unique label
    for label in unique_labels:
        if label == -1:  # Skip background label
            continue

        # Create a binary mask for the current label
        label_mask = np.zeros_like(markers, dtype=np.uint8)
        label_mask[markers == label] = 255

        # Use regionprops to get properties of the labeled region
        props = measure.regionprops(label_mask)

        # Check the size of the region (in pixels)
        region_size = props[0].area

        # Check if the region size is within the desired range (20 to 100 pixels)
        if 3 <= region_size <= 5000:
            count += 1
        
    return markers, count


def NPRecognition(img, mean_int, filename, image_path, results_path):
    # Preprocessing
    #img_blurred = filters.gaussian(img, sigma=5)

    #img = img - 0.5 * (img - img_blurred)
    img = img - 0.2 * mean_int
    img = cv2.convertScaleAbs(img)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Dynamic Thresholding
    mean_value = np.mean(img)
    std_dev = np.std(img)
    dynamic_threshold = mean_value + 1.5 * std_dev
    ret, thresh = cv2.threshold(img, dynamic_threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=5)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=4)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    
    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Create markers
    ret3, markers = cv2.connectedComponents(sure_fg)
    #print("Markers prior adding 10: ", np.unique(markers))
    #So let us add 10 to all labels so that sure background is not 0, but 10
    markers = markers+10
    #print("Markers after adding 10: ", np.unique(markers))
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    #plt.imshow(markers, cmap='jet')   #Look at the 3 distinct regions.
    
    #Now we are ready for watershed filling. 
    markers = cv2.watershed(img_RGB,markers) #expects the source image to have a data type of CV_8UC3 (an 8-bit unsigned 3-channel image, i.e., a color image) and the destination image to have a data type of CV_32SC1 (a 32-bit signed single-channel image, typically used for labeling)
    #The boundary region will be marked -1
    #Let us color boundaries in red 
    img_RGB[markers == -1] = [255, 0, 0]
    img2 = color.label2rgb(markers, bg_label=0, bg_color=[0, 0, 0])
    #plt.imshow(img2)
    
    # Save the img2 as a TIFF image
    tiff_filename = filename + 'NP_labelled.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    img2_uint8 = img_as_ubyte(img2)
    # Save the TIFF image
    io.imsave(tiff_path, img2_uint8)
    
    # Save the img with overlay as a TIFF image
    tiff_filename = filename +  '_NP_original_label.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    img_uint8 = img_as_ubyte(img_RGB)
    # Save the TIFF image
    io.imsave(tiff_path, img_uint8)
    
    # Save the threshold image
    tiff_filename = filename + '_NP_sureFG.tif'
    tiff_path = os.path.join(image_path, tiff_filename)
    # Convert the img2 to uint8 format (required for TIFF)
    thresh_uint8 = img_as_ubyte(sure_fg)
    # Save the TIFF image
    io.imsave(tiff_path, thresh_uint8)
    
    # Count the number of unique labels excluding the background label (-1)
    unique_labels = np.unique(markers)
    # Initialize count
    count = 0

    # Iterate through each unique label
    for label in unique_labels:
        if label == -1:  # Skip background label
            continue

        # Create a binary mask for the current label
        label_mask = np.zeros_like(markers, dtype=np.uint8)
        label_mask[markers == label] = 255

        # Use regionprops to get properties of the labeled region
        props = measure.regionprops(label_mask)

        # Check the size of the region (in pixels)
        region_size = props[0].area

        # Check if the region size is within the desired range (20 to 100 pixels)
        if 2 <= region_size <= 5000:
            count += 1
    
    
    return markers, count


def cell_type_count(img_intensity_1, img_intensity_2, labels, mean_1, mean_2, results_path, DAPI_count):
    
    regions_1 = measure.regionprops(labels, intensity_image=img_intensity_1)
    regions_2 = measure.regionprops(labels, intensity_image=img_intensity_2)
    cell_count_1 = 0
    cell_count_2 = 0
    for region_1, region_2 in zip(regions_1, regions_2):
        region_intensity_1 = region_1.mean_intensity
        region_intensity_2 = region_2.mean_intensity
        if (region_intensity_1 >            mean_1):
            cell_count_1 += 1
        elif (region_intensity_2 > (mean_2)):
            cell_count_2 += 1
        else:
            continue

    try:
        cell_count_percentage_1 = (cell_count_1/DAPI_count) * 100
        cell_count_percentage_2 = (cell_count_2/DAPI_count) * 100
    except:
        cell_count_percentage_1 = 0
        cell_count_percentage_2 = 0
    
    return cell_count_1, cell_count_percentage_1, cell_count_2, cell_count_percentage_2
    
               

def MaxIntensityProjection(img):
    # Perform the maximum intensity projection along the Z-axis
    # Assuming CZI image has shape (Z, Y, X)
    num_slices, height, width = img.shape
    max_intensity_projection = np.zeros((height, width), dtype=img.dtype)
    
    for z in range(num_slices):
        max_intensity_projection = np.maximum(max_intensity_projection, img[z])
    
    
    return max_intensity_projection

def PropertiesFinder (img, markers, results_path, pixels_to_um): 
    regions = measure.regionprops(markers, intensity_image=img)
    """
    opening = clear_border(opening) #Remove edge touching grains
    #Check the total regions found before and after applying this.
    """
    #save data in csv
    csv_filename = filename + "prop_data.csv"
    csv_file_path = os.path.join(results_path, csv_filename)
    propList = ["Area", 
                "equivalent_diameter", 
                "orientation", 
                "MajorAxisLength", 
                "MinorAxisLength", 
                "Perimeter", 
                "MinIntensity", 
                "MeanIntensity", 
                "MaxIntensity"]
    output_file = open(csv_file_path, "w")
    output_file.write("#" + "," + "," + ",".join(propList)+"\n")
    
    grain_number = 1
    for region_props in regions:
        #output_file.write(str(cluster_props["Label"]))
        output_file.write(str(grain_number)+",")
        for i, prop in enumerate(propList):
            if(prop == "Area"):
                to_print = region_props[prop]*pixels_to_um**2
            elif(prop == "orientation"):
                to_print = region_props[prop]*57.2958 # convert to degrees from radians
            elif (prop.find("Inensity") <0):
                to_print = region_props[prop]*pixels_to_um
            else:
                to_print = region_props[prop]
            output_file.write("," + str(to_print))
        output_file.write("\n")
        grain_number +=1
    output_file.close()

    """
    for prop in regions:
        print('Label: {} Area: {}'.format(prop.label, prop.area))
    """
    
    statistics = cle.statistics_of_labelled_pixels(img, (markers-10))
    stats_table = pd.DataFrame(statistics) 
    output_file = r'/{}stats_table.csv'.format(filename)
    stats_table.to_csv(results_path + output_file, index=False, header=True)
    count = stats_table.shape[0]
    
    return count
            
  
###############################################################################
# MAIN
###############################################################################

# =============================================================================
# where are your data located? 
# =============================================================================
root_path = filedialog.askdirectory(title="Please select your folder with images to analyze")

#create results path for data
results_path = os.path.join(root_path,"Analysis")
if not os.path.exists(results_path):
    os.makedirs(results_path)

#create results path for images (control instance)
results_path_images = os.path.join(root_path,"Analysis_Images")
if not os.path.exists(results_path_images):
    os.makedirs(results_path_images)

#initialize lists of needed output
filename_list = []
condition_list = []
cell_round_list = []

DAPI_count_list = []
count_NP_list = []
count_NP_percent_list = []



#extracts all neccessary data from czi file
for file in os.listdir(root_path):
    if file == "Analysis":
        continue
    if file.endswith(".png"):
        continue
    if file.endswith(".zip"):
        continue
    if file.endswith(".czi"):
        filename = file
        file_directory = os.path.join(root_path,filename)
        img = czifile.imread(file_directory, order='TCZYX') # time channel stacksize Rows Columns RGB (RGB =1 --> grayscale)
        #print(filename)
        #print(img.shape) # time channel Rows Columns RGB
        #load in your single channels as np.array
        
        condition = ConditionFinder(filename)
        cell_round = RoundFinder(filename)
        cell_round_list.append(cell_round)
        
        try:
            # if images are stacks
            img_DAPI = MaxIntensityProjection(img[0, DAPI, :, :, :, 0] )
            img_NP = MaxIntensityProjection(img[0, NP, :, :, :, 0] )

                
        except:
            # if images are no stacks
            img_DAPI = img[0,DAPI,:,:,0] 
            img_NP = img[0,NP,:,:,0] 

        #plt.imshow(img_DAPI)
        #build the mean intensity of all channels (despite DAPI) as float (more precise), this will be your threshold if the staining is positive
        NP_mean = np.mean(img_NP, dtype = np.float32)
        DAPI_mean = np.mean(img_DAPI, dtype = np.float32)
        #analyze DAPI staining
        label_NP, count_NP = NPRecognition(img_NP, NP_mean, filename, results_path_images, results_path)

        label, DAPI_count = NucleiRecognition(img_DAPI, DAPI_mean, filename, results_path_images, results_path)


        
        #fill your lists
        if DAPI_count > 0:
            filename_list.append(filename)
            condition_list.append(condition)
            
            DAPI_count_list.append(DAPI_count)
            count_NP_list.append(count_NP)

            try:
                count_NP_percent_list.append((count_NP/DAPI_count)*100)
            except:
                count_NP_percent_list.append("")

        else:
            continue
        
    else:
        continue
        
# =============================================================================
# Prepare for saving
# =============================================================================

#initialize dictionaries
dict_all = {}

#fill your dicts
dict_all['filename'] = filename_list
dict_all['condition'] = condition_list
dict_all['round'] = cell_round_list
dict_all['DAPI_count'] =DAPI_count_list
dict_all['NP_count'] =count_NP_list
dict_all['NP_%'] =count_NP_percent_list


'''
# Iterate through the dictionary and show the length of each value
for key, value in dict_all.items():
    print(f"The length of the value for key '{key}' is: {len(value)}")
'''

df_all = pd.DataFrame(dict_all)
data_all_sort = df_all.sort_values(by=['condition', 'round'], ascending=[True, False])


# write the dataframe to a CSV file with headers
data_all_sort.to_csv(results_path + r'\analysis_all.csv', index=False, header=True)

#calculate your data per rounds
#generate a new dataframe without the filename column
df_all.drop(columns=["filename"], inplace=True)
# Ensure all percentage values are treated as floats
df_all['NP_%'] = pd.to_numeric(df_all['NP_%'], errors='coerce')
# Only aggregate numeric columns
numeric_cols = df_all.select_dtypes(include=[np.number]).columns
# Now group and calculate the mean
df_sum = df_all.groupby(['round', 'condition'])[numeric_cols].mean().reset_index()
df_sum.to_csv(results_path + r'\analysis_summary.csv', index=False, header=True)

# =============================================================================
# Plotting
# =============================================================================

#create results path for graphs
results_path_graphs = os.path.join(root_path,"Analysis_Graphs")
if not os.path.exists(results_path_graphs):
    os.makedirs(results_path_graphs)

import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


# 1. Dynamically determine groups (e.g., by sex)
group_names = sorted(set(label.split()[0] for label in condition_definitions))  # e.g., ['Female', 'Male']

# 2. Assign unique colors dynamically
color_palette = cycle(plt.get_cmap('tab10').colors)  # you can also try 'Set2' or 'Dark2'
color_map = {group: next(color_palette) for group in group_names}

# Containers for plotting
plot_labels = []
means = []
sems = []
bar_colors = []

# 3. Loop through all conditions and extract values
for label, condition in condition_definitions.items():
    filtered = df_all[df_all['condition'] == condition]
    values = filtered['NP_%'].astype(float)  # Ensure it's numeric
    
    means.append(values.mean())
    sems.append(values.std() / (len(values) ** 0.5))
    plot_labels.append(label)
    
    # Determine group dynamically (e.g., 'Female' or 'Male') for color
    group = label.split()[0]
    bar_colors.append(color_map[group])

# 4. Dynamic bar plot
x_pos = np.arange(len(plot_labels))

fig, ax = plt.subplots()
ax.bar(x_pos, means, yerr=sems, align='center', alpha=0.6, ecolor='black',
       capsize=10, color=bar_colors)

ax.set_ylabel('Infection Rate [%]')
ax.set_title('Total infection rate')

# X-axis labels
ax.set_xticks(x_pos)
ax.set_xticklabels(plot_labels, rotation=45, ha='right')

# Style the plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.tick_params(axis='y', which='both', width=1.5)
ax.set_ylim(0, max(means) + 10)

# Save the plot
plt.tight_layout()
save_path = os.path.join(results_path_graphs, "Total_Infections.png")
plt.savefig(save_path)
#plt.show()
