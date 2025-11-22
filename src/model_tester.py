#!/usr/bin/env python3
"""
Spectral Ratio Model Testing Script
===================================

This script loads a trained spectral ratio model and evaluates it on a specified
test dataset. It supports UNet and ViT-based map-prediction architectures and
can optionally test an augmented test split.

Usage
-----
Run from the command line:

    python test_map_model.py \
        --run RUN_NAME \
        --method {modulate,rotate} \
        [--aug_test] [--linear] [--last_checkpoint]

Required Arguments
------------------
--run RUN_NAME
    Name of the experiment folder inside:
    /projects/SuperResolutionData/spectralRatio/SR_prediction/training/results/map/
    Determines which checkpoint to load and where results are saved.

--method METHOD
    Illumination augmentation type applied to the test set samples.
    Options:
        modulate  - cosine modulation augmentation
        rotate    - rotation-based augmentation

Optional Flags
--------------
--aug_test
    Use the augmented test dataset:
    /projects/SuperResolutionData/spectralRatio/SR_prediction/training/data/map/
    map_modulated_dataset_20251105_test_x10.csv

    If this flag is omitted, the script uses:
    /projects/SuperResolutionData/spectralRatio/SR_prediction/training/data/map/
    map_dataset_20250524_test_x0.csv

--linear
    Set this flag if the input images are already in **linear RGB**.
    When set, the transform pipeline will **skip** the log→linear conversion
    (i.e., it will NOT apply `ToLogRGB()`).

--last_checkpoint
    Set this flag if you would like to use the last checkpoint from the model. Default is to us eth best model checkpoint based on validation set performance

Example Commands
----------------
1) Standard test split, method=modulate, input in log space:
    python test_map_model.py --run ViT_run_01 --method modulate

2) Augmented test split, method=rotate, input already linear:
    python test_map_model.py --run UNET_ViT_run_12 --method rotate --aug_test --linear

Outputs
-------
- Results directory:
      SR_prediction/testing/map/results/<RUN_NAME>_complete_DEBUG
- Log file:
      SR_prediction/testing/map/results/<RUN_NAME>_complete_DEBUG/<RUN_NAME>.log
- Saves evaluation metrics, histograms, and a summary report.

Notes
-----
- The architecture is inferred from RUN_NAME (e.g., contains 'UNET_run', 'UNET_ViT_run', or 'ViT_run') and
  selects the corresponding model class.
- GPU is used if available; otherwise falls back to CPU.
- If `--linear` is NOT provided, the pipeline assumes log-RGB inputs in [0,1] and applies `ToLogRGB()`.
"""

############################################################################################# 
# PACKAGES 
import os
import sys
import logging
import argparse
import torch
import torch.nn                 as nn
import torchvision.transforms   as transforms

############################################################################################# 
# MODULES 
module_path = "/projects/SuperResolutionData/spectralRatio/SR_prediction/training"
sys.path.append(module_path)

from model_test_class                     import ModelTester
from utils.dataset_and_transform_classes  import AugmentedSpectralDataset_MAP, ToTensor, RandomCropGPU, ToLogRGB, LargestDivisibleCrop, CenterCropToSize
from models.unet_models                   import ResNet50UNet, ResNet50UNet_ViT, ResNet18UNet
from models.old_unet_models               import ResNet50UNet_old
from models.map_ViT_models                import ViT_Patch2Patch_ver2



#############################################################################################
# FUNCTIONS

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Test a trained model on the spectral ratio test set.")
    parser.add_argument("--run", type=str, required=True, help="Name of the training results dir to test (name also used for saving results).")
    parser.add_argument("--aug_test", action="store_true", help="Use augmented dataset in testing.")
    parser.add_argument("--method", type=str, required=True, help="Type of augmentation applied to test set, if any: 'modulate', 'rotate'.")
    parser.add_argument("--linear", action="store_true", help="Set this flag if the input images are already linear RGB (skip log->linear conversion in transforms).")
    parser.add_argument("--last_checkpoint", action="store_true", help="Set this flag to use the last model checkpoint, default is best.")    
    return parser.parse_args()

def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,                                                 
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
    )
    return logging.getLogger(__name__)

############################################################################################# 
# MAIN 

def main():
    
    args = parse_args()
    test_name = args.run
    model_checkpoint = "last" if args.last_checkpoint else "best"
    checkpoint_path = f"/projects/SuperResolutionData/spectralRatio/SR_prediction/training/results/map/{test_name}/model_states/{model_checkpoint}_model.pth"
    if args.aug_test:
        test_dataset_path = "/projects/SuperResolutionData/spectralRatio/SR_prediction/training/data/map/map_modulated_dataset_20251105_test_x10.csv"
        save_path = f"SR_prediction/testing/map/results/{test_name}_aug_test_{model_checkpoint}_checkpoint"
    else:
        test_dataset_path = "/projects/SuperResolutionData/spectralRatio/SR_prediction/training/data/map/map_dataset_20250524_test_x0.csv"
        save_path = f"SR_prediction/testing/map/results/{test_name}_{model_checkpoint}_checkpoint"
    image_dir = "/projects/SuperResolutionData/spectralRatio/sr_map_testing_data/images/"
    map_dir = "/projects/SuperResolutionData/spectralRatio/sr_map_testing_data/map_annotations"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    os.makedirs(save_path, exist_ok=True)
    log_file = f"{save_path}/results.log"
    augmentation_method = args.method
    is_linear_input = args.linear

    # Set up logging
    logger = setup_logger(os.path.join(save_path, f"{test_name}.log"))
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Test: {test_name}")
    logger.info(f"Dataset path: {test_dataset_path}")
    logger.info(f"Checkpoint path: {checkpoint_path}")

    size=512
    try:
        # Set transforms
        transform_list = [ToTensor(), CenterCropToSize(target_size=(size, size))]
        if not is_linear_input:
            transform_list.append(ToLogRGB())
        composed_transforms = transforms.Compose(transform_list)

        composed_transforms = transforms.Compose(transform_list)
        logger.info("Transforms:")
        for t in composed_transforms.transforms:
            logger.info(f"       {t.__class__.__name__}")

        #Init Model
        if 'UNET_run' in test_name:
            model = ResNet50UNet(
                in_channels=3,
                out_channels=3,
                pretrained=False,
                se_block=True,
                dropout=0.0
            ).to(device)

        elif 'UNET_ViT_run' in test_name:
            model = ResNet50UNet_ViT(
                in_channels=3,
                out_channels=3,
                pretrained=False,
                checkpoint=None,
                use_vit_bottleneck=True, 
                vit_embed_dim=512, 
                vit_depth=4, 
                vit_heads=8,
                se_block=True,
                freeze_encoder=False,
                freeze_decoder=False,
            ).to(device)    

        elif "ViT_run" in test_name:
            model = ViT_Patch2Patch_ver2(
                img_size=512, 
                patch_size=16, 
                in_ch=3, 
                out_ch=3, 
                embed_dim=768, 
                depth=4, 
                heads=8,
                dropout=0.1
            )
        logger.info(f"Model initialized: {model.__class__.__name__}")


        # Init testing set
        test_ds = AugmentedSpectralDataset_MAP(
                                                csv_path=test_dataset_path, 
                                                image_dir=image_dir,
                                                map_dir=map_dir,
                                                augmentation_method=augmentation_method,
                                                augment=True,
                                                ds_device=device, 
                                                image_transforms=composed_transforms,
                                                include_img_name=True
                                                )  
        logger.info(f"Test Dataset initialized with {len(test_ds)} samples")

        # Init tester class
        model_tester = ModelTester(
                                model=model, 
                                checkpoint_path=checkpoint_path, 
                                test_dataset=test_ds,
                                is_linear_input=is_linear_input, 
                                save_path=save_path, 
                                device=device
                                )
        logger.info(f"Tester initialized: {model_tester.__class__.__name__}")
        logger.info(f"Linear Input: {is_linear_input}")


        # Test
        logger.info("Testing...")
        metrics = model_tester.test(outliers=True, plot_hist=True)
        model_tester.save_model_test_summary(metrics=metrics, test_name=test_name)
        logger.info("Test complete.")
    except Exception as e:
        logger.error(f"An error occurred during testing: {e}", exc_info=True)

if __name__ == "__main__":
    main()