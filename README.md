# CIL-road-segmentation
Road segmentation is the task of labeling all pixels of an input satellite image as road or no-road. This is a popular problem in computer vision with applications in GPS navigation systems or autonomous driving. In recent years a plethora of different network architectures have been presented, achieving amazing results. These methods all look at road segmentation as a fully supervised task and present discriminative models. In this paper we interpret road segmentation as an image to image translation problem that can be solved with generative models like conditional generative adversarial networks. We compare the classic and generative methods over a variety of different network architectures and find the generative approach achieves segmentation results on par with the traditional classic approach.

## Setup
1. Download datasets and put both folders into the /data folder
    - [Cil-road-segmentation-2022](https://drive.google.com/file/d/1QGM-OkmZZX5SNyVctfb49yjWqATp8bR_/view?usp=sharing)
    - [Google](https://drive.google.com/file/d/1PqwfdOSPaAug51c93ndg5-A3Yfx7fico/view?usp=sharing)
2. If you want to use the ConvNext model or Drn-D you have to download their pretrained weights and place them into models/pretrained:
    - [ConvNext](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)
    - [Drn-D-105](https://drive.google.com/drive/folders/1fIsCB877l37cFAJomoQzNrEYPqwMwZ4Q)
3. Set Up the conda environment
    ```
    conda env create -f environment.yml -n cil
    conda activate cil
    ```

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
Here we describe what the different parameters represent:
 - **batch_size**: Batch size. Hence the number of data samples per patch. In all experiments we used batch size equal to 1.
 - **beta1**: Hyperparameter for the momentum in the Adam optimizer. In the classic mode we choose 0.9 and for GAN's we choose 0.5.
 - **beta2**: Hyperparameter for the second momentum in the Adam optimizer. We choose 0.999 for all experiments.
 - **checkpoint_load_pth**: Path to your file location where you placed your model checkpoint.
 - **checkpoint_root**: Path to your file location where we will save your model checkpoints. We save a checkpoint after every epoch.
 - **dataset**: We present different choices for datasets, which you will find in the /dataset folder. We choose PairedDatasetLabel for the classic and GAN-Mask approach. For GAN-Map we choose PairedDataset.
 - **debug**: If debug mode is activated additional flags get set for more logging and deterministic executions for easier debugging.
 - **device**: Chose 'cuda' to run on gpu or 'cpu' to run on your cpu.
 - **disc_lambda**: Hyperparameter to weight the GAN-loss. Default is set to 100.
 - **epoch_count**: If you want to load a checkpoint for testing or continue a run you set the epoch from which you want to load the checkpoint.
 - **epochs**: Until which epoch you want to train.
 - **generation_mode**: Choose 'classic' or 'gan'. The classic-mode just uses the supervised training, while the gan-mode will also train/load/save the discriminator.
 - **image_height**: To what height you would like to resize your image before feeding it into the network. Default is 384. For the pix2pix model you should however choose 256 or 512.
 - **image_width**: To what width you would like to resize your image before feeding it into the network. Default is 384. For the pix2pix model you should however choose 256 or 512.
 - **learning_rate**: Learning rate paramter for the Adam optimizer. Default is set to 0.0002.
 - **load_from_checkpoint**: This flag needs to be set to True if we want to load from a checkpoint. Otherwise it should be set to False.
 - **loss_function**: We implemented different loss functions in utils/loss_functions.py to choose from. We usually use BCELoss during pretraining and DiceLoss during finetuning. In GAN-Map we choose the L1Loss.
 - **model**: We implemented a variety of differnt network architectures that are used to generate the mask/roadmap. Your choices are: unet, drnA, drnD, pix2pix, pix2pixhd, unet3plus, convnext_unet. More details are provided in the paper and in the code.
 - **num_workers**: The number of threads you can use. Default is set to 8.
 - **pin_memory**: For faster training it is useful to set this to True.
 - **result_dir**: Path to the directory where you want your results to be stored. Default ist "results"
 - **root_A**: Path to the training data directroy that holds the images of domain A.
 - **root_B**: Path to the training data directroy that holds the images of domain B.
 - **save_checkpoint**: This needs to be set to True if we want to save model and optimizer checkpoints.
 - **seed**: Initial seed for reproducability.
 - **train_loss_track**: Initial value for the training loss we track. We employ an average Meter.
 - **transformation**: If the domain B is a mask you should choose 'label'. If the domain B is an rgb image like a roadmap you should choose 'rgb'.
 - **val_loss_track**: Initial value for the validation loss we track. We employ an average Meter.
 - **wandb_logging**: We support logging via weights and biases. Set this flag to True to log your results. Otherwise turn if off by setting the flag to false.