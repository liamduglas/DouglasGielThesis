# A U-Net to Identify Deforested Areas in Satellite Imagery of the Amazon

Deforestation in the Amazon rainforest has the potential to have devastating effects on ecosystems on both a local and global scale and is, therefore, one of the most environmentally threatening phenomena occurring today. In order to minimize deforestation in the Amazon and its subsequent consequences, it is helpful to analyze its occurrence using machine learning architectures such as the U-Net. The U-Net is a type of Fully Convolutional Network that has shown significant capability in performing semantic segmentation. It is built upon a symmetric series of downsampling and upsampling layers which propagates feature information into higher spatial resolutions, allowing for precise identification of features on the pixel scale. Such an architecture is well-suited for identifying features in satellite imagery.  In this repository, we present a U-Net to identify deforested areas in satellite imagery of the Amazon through semantic segmentation. The structure of this U-Net can be seen below.

<p align="center">
<img width="406" alt="Screenshot 2024-04-05 at 3 20 22 PM" src="https://github.com/liamduglas/DouglasGielThesis/assets/126018139/f3cbe752-77f1-4617-b85c-949d1b179815">
</p>

## Data Sources

Data is collected via the Google Earth Engine API. Our input images are from the Landsat 8 satellite and our ground-truth images are from the [Global Forest Change](https://glad.earthengine.app/view/global-forest-change#bl=off;old=off;dl=1;lon=20;lat=10;zoom=3;) dataset. Examples of these input and ground-truth images can be seen below. 

<p align="center">
<img width="225" alt="Screenshot 2024-04-05 at 3 23 00 PM" src="https://github.com/liamduglas/DouglasGielThesis/assets/126018139/ccefc0d2-213e-450d-b174-241eeda3c4c8">
</p>

500 512x512 images of 60 meter resolution (which automatically get partitioned into 6400 128x128 images via `Data_Driver.py`) are available on [DropBox](https://www.dropbox.com/scl/fo/90y2x8ez9xtsafozqd2bo/h?rlkey=ssgeie6bwo6mtdr7yfm23o6vl&dl=0).

Images of other specifications (e.g. year, threshold, resolution, source) can be downloaded via `Image_Exporter.py`.

## Running the Model

To run the model, you can use either our pre-trained `weights.weights.h5` folder or generate your own weights. 

