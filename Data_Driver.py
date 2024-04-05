from patchify import patchify
import tifffile as tiff
import numpy as np
import os.path

all_img_patches = []

# grab all sat images into set
images = []
images_path = '250_epoch_500_img_run/landsat_images_year_2022_res_60_vis_ee' #'BigData/landsat_images_year_2020_res_60'
valid_images = [".tif"]

for f in sorted(os.listdir(images_path)):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = tiff.imread(images_path+ "/" + f)
    patches_img = patchify(image, (128, 128, 3), step=128)  # Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
         for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = (single_patch_img.astype('float32')) / 255.
            images.append(single_patch_img)

images_x = np.array(images)

# grab all sat images into set
images = []
images_path = '250_epoch_500_img_run/hansen_images_year_2020_res_60_thresh_80'
valid_images = [".tif"]
for f in sorted(os.listdir(images_path)):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    image = tiff.imread(images_path + "/" + f)
    patches_img = patchify(image, (128, 128, 3), step=128)  # Step=256 for 256 patches means no overlap

    for i in range(patches_img.shape[0]):
         for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, :, :]
            single_patch_img = (single_patch_img.astype('float32')) / 255.
            single_patch_img = single_patch_img.squeeze()
            images.append(single_patch_img[:,:, 0])

images_y = np.array(images)
