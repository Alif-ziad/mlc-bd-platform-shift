# MLC-BD platform shift

A leave-one-platform-out evaluation of two image classifiers on MLC-BD, a land-cover dataset of 10,683 images from Bangladesh. This is a course project (Computer Vision and Pattern Recognition, AIUB, 2026), done in a group of three. It has not been peer reviewed.

## Question

MLC-BD collects its images from three mapping platforms: ArcGIS, Bing Maps and Google Maps. A random train/test split puts all three platforms in both sets. How much does the score drop when the test images come from a platform the model never saw during training?

## Dataset

- **MLC-BD**: Rafi, M. A. H., Johora, M. F., Sultana, T., Hossain, M. D., Hossen, M. J. and Islam, M. M. (2026), *MLC-BD: Multi-Platform Land Cover Imagery Dataset of Bangladesh*, Mendeley Data, V1, DOI 10.17632/gr549zm73h.1. Licence: CC BY 4.0.
- 10,683 PNG images: ArcGIS 3,556, Bing 3,632, Google 3,495. Seven classes (Agricultural Land, Beaches/Coastline, Built-up Areas, Grasslands/Open Land, Hills/Mountains, Trees/Forests, Water Bodies), 1,509 to 1,547 images each. Images are 88 to 506 pixels wide.
- The dataset is not included here. Download it from Mendeley Data.
- The images are basemap imagery from the three providers. This repository contains no dataset images and no Grad-CAM overlays. Image outputs that showed them were removed from the notebooks. See the dataset page for the terms that apply to the images.

## What the notebooks do

1. Build a table of image paths with platform and class, taken from the folder names.
2. Audit the data. SHA-256 hashes found 16 exact-duplicate images in 8 groups, all from Google; one copy of each was kept (10,675 images remain). A perceptual-hash check found no identical hashes. Near-duplicates at a distance threshold were not checked.
3. Make the splits: a stratified 70/15/15 mixed split, and three leave-one-platform-out splits (train on two platforms with a 15% validation slice, test on the third). No image from the held-out platform is used to choose a model.
4. Train ResNet-50 and MobileNetV3-Large (torchvision, ImageNet weights, new 7-class head). Settings: 224 x 224 input, batch size 64, AdamW with learning rate 1e-4 and weight decay 1e-4, up to 15 epochs, early stopping on validation macro-F1 with patience 5, seed 42, mixed precision on one T4 GPU.
5. Run the leave-one-platform-out tests with ResNet-50 (the better model on the mixed split), once with standard augmentation (random resized crop, flips, rotation) and once with strong augmentation (the same plus colour jitter and a light blur).
6. Report accuracy, macro-F1 and MCC, and draw Grad-CAM maps (Captum) for a few cases.

## Results (macro-F1)

The two notebooks are two runs of the same pipeline, one trained once per setting. Run 2 is the later one.

| Setting | Run 1 | Run 2 |
|---|---|---|
| Mixed split, ResNet-50 | 0.991 | 0.992 |
| Mixed split, MobileNetV3-Large | 0.989 | 0.987 |
| Held out ArcGIS, standard augmentation | 0.716 | 0.716 |
| Held out Bing, standard | 0.767 | 0.849 |
| Held out Google, standard | 0.301 | 0.287 |
| **Mean held out, standard** | **0.595** | **0.617** |
| Held out ArcGIS, strong augmentation | 0.777 | 0.781 |
| Held out Bing, strong | 0.766 | 0.827 |
| Held out Google, strong | 0.577 | 0.498 |
| **Mean held out, strong** | **0.706** | **0.702** |
| Gap (mixed ResNet-50 minus mean standard) | 0.397 | 0.375 |
| Gain from strong augmentation (mean) | 0.112 | 0.085 |

All numbers are in `results/summary.csv`. The bar charts are in `figures/`.

What they show:

- A random split gives about 0.99. On a platform the model never saw, macro-F1 runs from 0.29 to 0.85.
- Google is the hardest platform in both runs. With standard augmentation its macro-F1 is about 0.29 to 0.30 and its accuracy about 0.35, against a chance level of 0.14 for seven classes.
- Strong augmentation raised the held-out average to about 0.70 in both runs. It helped ArcGIS and Google and did not help Bing.
- On held-out Google, the wrong predictions are confident. In the Grad-CAM cases I inspected, confidence ran from 0.53 to 1.00, and many images are predicted as Built-up Areas (see the confusion matrix in the run 2 notebook).

## Limitations

- Each setting was trained once per run. The two runs differ by up to 0.08 on a single platform, so differences smaller than that are noise. Several seeds with a mean and standard deviation are the next step.
- The platforms are not sensors. The providers may use the same upstream imagery.
- The dataset gives no coordinates, so the same place appearing on two platforms cannot be excluded. That could inflate the mixed-split score.
- Labels are image-level only, and Grad-CAM is qualitative evidence.
- Images of different sizes are all resized to 224 x 224.

## Planned

Several seeds per setting; MobileNetV3-Large in the leave-one-platform-out runs; calibration under platform shift (expected calibration error, temperature scaling); tests of why Google fails (grayscale training, colour normalisation, re-estimated batch-norm statistics).

## Reproduce

The notebooks were run on Kaggle. Cell 3 searches `/kaggle/input` for the images, and the paths in the outputs point to my own copy of the dataset. To rerun, download MLC-BD, change the search root in cell 3, and install the packages in `requirements.txt`. A GPU is needed for reasonable run times (about 10 minutes per ResNet-50 run on a T4).

## Credits and AI assistance

Code and experiments: MD Alif Ziad Sarkar. I wrote the code with help from AI coding assistants, ran every experiment myself, and checked the numbers in this README against the notebook outputs.

Dataset: the MLC-BD authors above. Methods: ResNet (He et al., 2016), MobileNetV3 (Howard et al., 2019), Grad-CAM (Selvaraju et al., 2017), and the Captum library.

## Licence

The code is under the MIT licence (see `LICENSE`). The dataset keeps its own licence.
