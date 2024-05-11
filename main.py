'''user should give:
1. the path to the folder containing the five folders
2. what folders to process out of the five folders
3. path to save the created dataset which will have:
 full_masks, patches/train, patches/val, train and val folders will include images and masks
 '''

import patch


data_dir = 'data'
dirs = ['Breast3_ihc'] # dirs to process, give one or as many as you want
save_dir = 'full_dataset' # where to save the dataset


patch.mask_patch(data_dir, dirs=dirs, save_dir=save_dir, clear=False, create_mask=True, patch_size=768)

