# Fine-grained Vision-language Pre-training for Enhanced CT Image Understanding 
[Paper](https://openreview.net/pdf?id=nYpPAT4L3D) (ICLR 2025 Spotlight)


## Data processing

- Download the [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE) dataset into the data folder.

- Download ImageNet pre-trained ViT weights from [link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), and BiomedVLP-CXR-BERT-specialized text encoder from [link](https://huggingface.co/microsoft/BiomedVLP-CXR-BERT-specialized), as used by CT-CLIP.

- Download the decomposed anatomy-wise descriptions from our provided supplementary materials [link](https://drive.google.com/drive/folders/10bz2UFxqxDPzl2P9NohESSNyBuld_Iek?usp=drive_link), and process the CT volume with the following commands.

  ```bash
  cd data
  python fix_data.py --split [train/valid]
  python generate_mask.py --split [train/valid]
  python resize.py --split [train/valid]
  python preprocess.py --split [train/valid]
  ```

  The processed results.

  ```bash
  |-- BiomedVLP-CXR-BERT
  |-- data
  |   |-- train
  |   |-- valid
  |   |-- train_fixed
  |   |-- valid_fixed
  |   |-- train_mask
  |   |-- valid_mask
  |   |-- resized_train_images
  |   |-- resized_train_masks
  |   |-- resized_valid_images
  |   |-- resized_valid_masks
  |   |-- processed_train_images
  |   |-- processed_train_masks
  |   |-- processed_valid_images
  |   |-- processed_valid_masks
  |   |-- multi_abnormality_labels
  |   |-- desc_info.json
  |   |-- conc_info.json
  |-- mae_pretrain_vit_base.pth
  ```



## Training

```shell
torchrun --nproc_per_node=4 train.py
```

[Pre-trained weights](https://drive.google.com/drive/folders/15BnMo1lIAlOH_8KLdB2NugiHnmj9AWSD?usp=drive_link) of CT-RATE are released. 


## Evaluation

```bash
torchrun --nproc_per_node=4 eval.py
```

Then, you can calculate the metrics using the generated CSV file.

```bash
python calc_metrics.py --csv_file res/xxx.csv
```

## Citation
If you find this repository useful, please cite:
```
@inproceedings{fvlm_iclr25,
  title={Large-scale and fine-grained vision-language pre-training for enhanced CT image understanding},
  author={Zhongyi Shui, Jianpeng Zhang, Weiwei Cao, Sinuo Wang, Ruizhe Guo, Le Lu, Lin Yang, Xianghua Ye, Tingbo Liang, Qi Zhang, Ling Zhang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  pages={},
  year={2025}
}
```
