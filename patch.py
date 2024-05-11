import xml.etree.ElementTree as ET
import openslide
from PIL import Image
import numpy as np
import os
import cv2
from tqdm import tqdm, trange
from scipy.interpolate import splev, splprep
import sys

FILE_ENDS = ('.svs', '.ndpi', '.tiff', '.tif', '.mrxs')
DIRS = ['Breast3_ihc', 'LymphNode1_he', 'Breast1_he',  'Breast2_he']

def extract_patches(slide, save_dir, name=None,train=True, mask=None, mask_path=None, patch_size=1280, level=0):
    if mask_path is not None:
        mask = Image.open(mask_path)
    elif mask is not None:
        mask = Image.fromarray(mask)

    if mask is not None:
        assert slide.level_dimensions[level][0] == mask.size[0]
        assert slide.level_dimensions[level][1] == mask.size[1]

    f_name = name.split('/')[-1].split('.')[0]

    for x in trange(0, slide.level_dimensions[level][0], patch_size):
        for y in range(0, slide.level_dimensions[level][1], patch_size):
            if mask is not None:
                mask_patch = mask.crop((x, y, x + patch_size, y + patch_size))
                if np.sum(mask_patch) == 0 and train:
                    continue 
                mask_patch = mask_patch.resize((256, 256))
                mask_patch.save(os.path.join(save_dir, 'masks', '{}_{}_{}.png'.format(f_name, x, y)))

                patch = slide.read_region((x, y), level, (patch_size, patch_size))
                patch = patch.resize((256, 256))
                patch.save(os.path.join(save_dir, 'images', '{}_{}_{}.png'.format(f_name, x, y)))
            else:
                patch = slide.read_region((x, y), level, (patch_size, patch_size))
                patch = patch.resize((256, 256))
                patch.save(os.path.join(save_dir, '{}_{}_{}.png'.format(f_name, x, y)))

            cv2.imshow('patch', np.array(patch))
            if mask is not None:
                cv2.imshow('mask', np.array(mask_patch)*255)
            cv2.waitKey(1)
    slide.close()

def scale_coordinates(coords, orig_size, new_size):
    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]
    return [(int(x * scale_x), int(y * scale_y)) for x, y in coords]

def parse_coordinate(coord_string):
    return float(coord_string.replace(',', '.'))

def extract_mask(slide, xml_path, save_dir, name='', level=3, write=False):
    level_dimensions = slide.level_dimensions[level]
    mask = np.zeros(level_dimensions[::-1], dtype=np.uint8)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for annotation in root.find('Annotations').findall('Annotation'):
        coordinates = []
        annotation_type = annotation.get('Type')
        for coordinate in annotation.find('Coordinates').findall('Coordinate'):
            x = parse_coordinate(coordinate.get('X'))
            y = parse_coordinate(coordinate.get('Y'))
            coordinates.append((x, y))

        scaled_coordinates = scale_coordinates(coordinates, slide.dimensions, level_dimensions)

        if annotation_type == 'Polygon' or annotation_type == 'Spline':
            scaled_coordinates = np.array(scaled_coordinates, np.int32)
            scaled_coordinates = scaled_coordinates.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [scaled_coordinates], 1)  # 1 for white
        elif annotation_type == 'Spline':
            x_coords, y_coords = zip(*scaled_coordinates)  # Separate x and y coordinates
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)

            # Prepare and interpolate the spline
            tck, u = splprep([x_coords, y_coords], s=0)
            unew = np.linspace(0, 1, 1000)  # Number of points can be adjusted
            out = splev(unew, tck)

            spline_points = np.array(out).T.round().astype(np.int32)
            cv2.polylines(mask, [spline_points], isClosed=True, color=1, thickness=1)
            cv2.fillPoly(mask, [spline_points], 1)

    if write:
        write_path = os.path.join(save_dir, 'full_masks', name+'.png')
        cv2.imwrite(write_path, mask*255)
        print('Full mask saved to {}'.format(write_path))
    return mask

def create_data_path(root='dataset', clear=False):
    if clear:
        os.system('rm -rf {}'.format(root))
    os.makedirs(root, exist_ok=True)

    os.makedirs(os.path.join(root, 'patches', 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'patches', 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(root, 'patches', 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(root, 'patches', 'test', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(root, 'full_masks'), exist_ok=True)


def extract_mask_patch(images, data_dir, save_dir, subset, create_mask=False, file_ends=FILE_ENDS, level=0, clear=False, patch_size=1024*3):
    patch_save_dir = os.path.join(save_dir, 'patches', subset)
    for f in tqdm(images):
        f_name = f.split('.')[0]
        svs_path = os.path.join(data_dir, f)
        img_name = f_name.split('/')[-1]
        slide = openslide.OpenSlide(svs_path)
        print('Patching {} slide'.format(f))
        if create_mask:
            xml_path = os.path.join(data_dir, f_name+'.xml')
            save_dir = os.path.join(save_dir)
            mask = extract_mask(slide, xml_path, save_dir, name=img_name, write=True, level=level)
            extract_patches(slide, name=f_name, mask=mask, save_dir=patch_save_dir,
                            level=level, patch_size=patch_size)
            print()
            print('Done patching {} ...'.format(f))
        else:
            extract_patches(svs_path, 'dataset/train', level=level, patch_size=patch_size)


def mask_patch(data_dir, dirs, save_dir, create_mask=False, file_ends=FILE_ENDS, level=0, clear=False, patch_size=1024*3):
    create_data_path(root=save_dir, clear=clear)
    wsi_images = []
    #slides to ignore
    slides_ignore = ['04_HER2.ndpi', '13_HER2.ndpi', '25_HER2.ndpi', '36_HER2.ndpi', '42_HER2.ndpi', '52_HER2.ndpi', '73_HER2.ndpi', '79_HER2.ndpi','HER2.ndpi', '15_HER2.ndpi', '28_HER2.ndpi', '40_HER2.ndpi', '47_HER2.ndpi', '64_HER2.ndpi', '75_HER2.ndpi', '81_HER2.ndpi',
                        '11_HER2.ndpi', '17_HER2.ndpi', '32_HER2.ndpi', '41_HER2.ndpi', '51_HER2.ndpi', '69_HER2.ndpi', '77_HER2.ndpi']
    for d in dirs:
        for f in os.listdir(os.path.join(data_dir, d)):
            if f in slides_ignore:
                continue
            if f.endswith(file_ends):
                wsi_images.append(os.path.join(d, f))
    #choose the last 8 as test
    train_indices = list(range(len(wsi_images)-8))
    test_indices = list(range(len(wsi_images)-8, len(wsi_images)))

    print(f'processing {len(train_indices)} train images...')
    extract_mask_patch([wsi_images[i] for i in train_indices], data_dir, save_dir,
                        'train', create_mask=create_mask, level=level,
                        clear=clear, patch_size=patch_size)
    
    print()
    print('------- Done processing train images ------- ')
    print(f'processing {len(test_indices)} test images...')
    extract_mask_patch([wsi_images[i] for i in test_indices], data_dir,
                        save_dir, 'test', create_mask=create_mask, level=level,
                        clear=clear, patch_size=patch_size)
        
mask_patch('data', DIRS, 'temp2', create_mask=True, clear=True)