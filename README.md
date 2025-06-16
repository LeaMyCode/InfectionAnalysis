# Infection Analysis
With the help of the code you can determine the number of cells within the image using the DAPI channel and the number of infected cells using the virus-specific (nucleoprotein)-stained channel. The channels are binarized utilizing the OpenCV library (version 4.8.1.78) with dynamic thresholding and the cell nuclei channel is prior processed 
with a Gaussian blur filter using the cv2.GaussianBlur function. Additionally, the principle of finding sure for- and background of the cells or nucleoprotein staining is performed to improve the thresholding (Bhattiprolu, 2020). Afterwards, segmentation is performed using watershed 
segmentation. The czifile library (version 2019.7.2) is used for image handling, data handling is performed using pandas (version 2.1.4) and NumPy is used for data processing (version 1.26.2). 

## Example of nuclei and nucleoprotein counted with the code (original image with segmenation mask)
![Methods_Infectionrate](https://github.com/user-attachments/assets/523bca2a-f5e1-4781-9009-d5ef872140b0)


## Please cite the following if you use the code
>Gabele, L., Bochow, I., Rieke, N., Sieben, C., Michaelsen-Preusse, K., Hosseini, S., et al. (2024). H7N7 viral infection elicits pronounced, sex-specific neuroinflammatory responses in vitro. Front Cell Neurosci 18. doi: 10.3389/fncel.2024.1444876
