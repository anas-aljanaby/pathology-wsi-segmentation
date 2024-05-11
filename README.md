# Histopathology Whole Slide Image Patching for Deep Learning Segmentation
These script are designed to process Histopathology Whole Slide Images (WSIs), which often span hundreds of thousands of pixels making them too large for deep learning models. To address this, the scripts employ a widely-used technique: it segments these extensive images into manageable patches (e.g., 256x256 pixels). These smaller, segmented patches are then used as training data for deep learning models, facilitating effective learning and analysis.

Included in this repository is example_fastai_model.ipynb, which demonstrates training a model using the FastAI library on the patched data prepared by the scripts.

Processing a Whole Slide Image, extracting 256x256 patches and a binary mask.
![WSI image patches](WSI_segmentation.gif)
