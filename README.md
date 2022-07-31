# CIL-road-segmentation
Given a sattelite image we return the estimated road segmentation.<br><br>

## Setup
1. Download datasets and put both folders into the /data folder
    - [Cil-road-segmentation-2022](https://drive.google.com/file/d/1QGM-OkmZZX5SNyVctfb49yjWqATp8bR_/view?usp=sharing)
    - [Google](https://drive.google.com/file/d/1PqwfdOSPaAug51c93ndg5-A3Yfx7fico/view?usp=sharing)
2. If you want to use the ConvNext model or Drn-D you have to download their pretrained weights and place them into models/pretrained:
    - [ConvNext](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)
    - [Drn-D-105](https://drive.google.com/drive/folders/1fIsCB877l37cFAJomoQzNrEYPqwMwZ4Q)
3. Set Up the conda environment
    - conda env create -f environment.yml -n cil
    - conda activate cil

## Train, Finetune and Test
You should first write/choose a config file that determines the different design choices. Note that we have places example config files into the config/ folder. What parameter determines what is descibed below.<br>
- To Train:
    ```
    python3 train.py --config_path <path_to_your_config_file>
    ```
- To Test:
    ```
    python3 test.py --config_path <path_to_your_config_file>
    ```

## Google dataset creation
1. Make file: `api_key.txt` and paste your google maps API key (static maps)
2. Run the following commands: 
    ```
    python3 google_maps_data_generation.py
    bash dataset_subfolder_split.sh
    python3 roadmap_to_mask.py
    python3 delete_empty_masks.py
    python3 normalize_color.py
    ```

## Reproduce our Submission
1. Train all 5 classical models on the data that we provide
2. Create a submission using all 5 trained models using the ensemble.py script


## Config Guide
Here we describe what the different parameters represent