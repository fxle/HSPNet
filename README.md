
# Localization, Consistency, and Interaction: A Holistic Spatio-Temporal Perception Network for Video Camouflaged Object Detection

## Datasets
Please search for and download the MoCA-Mask, CAD and COD10K datasets, and modify the paths accordingly in the `dataset.yaml` file.

## Install Requirements
To set up the experimental environment, please run the following command:

```bash
conda env create -f environment.yml
```

## Training

1. Pretrain on COD10K-TR: 

```bash
python main_for_image.py --config configs/icod_pretrain.py --info pretrain --model-name PvtV2B5_HSPNet --pretrained
```

2. Finetune on MoCA-Mask-TR:

```bash
python main_for_video.py --config configs/vcod_finetune.py --info finetune --model-name videoPvtV2B5_HSPNet --load-from <PRETRAINED_WEIGHT>
```
To use the multi-scale method, replace `PvtV2B5_HSPNet` with `PvtV2B5_HSPNet_MS`, and `videoPvtV2B5_HSPNet` with `videoPvtV2B5_HSPNet_MS` for the `--model-name` option.

## Evaluation
Due to file size limitations (the model weights exceed 50MB), we are unable to include the pretrained model weights in this supplementary material. To evaluate the model's performance, please first retrain the model by following the instructions in the Training section, and then run the following command:

```bash
python main_for_video.py --config configs/vcod_finetune.py --model-name videoPvtV2B5_HSPNet --evaluate --load-from <RETRAINED_WEIGHTS>
```

Please replace `<RETRAINED_WEIGHTS>` with the path to the weights obtained after retraining the model.

## Acknowledgements
We would like to express our gratitude to the authors of SLT-Net and ZoomNeXt, as we have referred to their code in our work.

## Note
This code is provided anonymously for review purposes. It contains no author information or external hyperlinks.

