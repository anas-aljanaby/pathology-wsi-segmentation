import os
from PIL import Image
import openslide
from tqdm import tqdm

patch_dir = 'working/test_predictions'
patch_files = [f for f in os.listdir(patch_dir) if os.path.isfile(os.path.join(patch_dir, f))]

save_dir = 'dataset/test_pred'
print(patch_files[0].split("_")[0])


slide = openslide.OpenSlide(f'sample_data/Breast1_he/{patch_files[0].split("_")[0]}.svs')
wsi_width = slide.level_dimensions[0][0]
wsi_height = slide.level_dimensions[0][1]

reassembled_image = Image.new('RGB', (wsi_width, wsi_height))

for patch_file in tqdm(patch_files):

    parts = patch_file.split('_')
    x_coord = int(parts[1])
    y_coord = int(parts[2].split('.')[0])

    patch_path = os.path.join(patch_dir, patch_file)
    patch = Image.open(patch_path)


    patch = patch.resize((3072, 3072))

    reassembled_image.paste(patch, (x_coord, y_coord))


# reassembled_image.save('reassembled_slide.png')
#save small version
reassembled_image = reassembled_image.resize((wsi_width//4, wsi_height//4))
reassembled_image.save('reassembled_slide_small.png')
# reassembled_image.show()

