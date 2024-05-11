from fastai.vision.all import load_learner, PILImage
from pathlib import Path
import openslide
from patch import extract_patches
import os
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
from fastai.vision.core import PILImage
from pathlib import Path


def patch_test_image(slide_path, save_dir='temp'):
    slide = openslide.OpenSlide(slide_path)
    save = os.path.join(save_dir, 'images')
    extract_patches(slide, save, name=slide_path, level=0, patch_size=1024*3, train=False)

def get_y_fn(x):
    return Path(str(x).replace("images", "masks"))

def create_save(path):
    #save_dir, save/masks, save/images
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, 'masks')):
        os.makedirs(os.path.join(path, 'masks'))
    if not os.path.exists(os.path.join(path, 'images')):
        os.makedirs(os.path.join(path, 'images'))
    
    
save_dir = 'test'
create_save(save_dir)
patch_test_image('sample_data/Breast1_he/TCGA-A2-A3XV-01Z-00-DX1.svs',
                 save_dir=save_dir)
learn = load_learner('train 2/images/3072-model.pkl', cpu=True)
# predict each patch
for f in tqdm(os.listdir(os.path.join(save_dir, 'images'))):
    img = PILImage.create(os.path.join(save_dir, 'images', f))
    pred = learn.predict(img)
    # image = Image.fromarray(pred[0].numpy())
    image = Image.fromarray(pred[0].numpy().astype('uint8'))
    image.save(os.path.join(save_dir, 'masks', f))

    # show for 10 ms each
    cv2.imshow('patch', np.array(img))
    cv2.imshow('predicted mask', np.array(image)*255)
    cv2.waitKey(10)




