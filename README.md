# Emphysema Subtyping on Chest CT using Deep Neural Networks

## News

- The source code (v1.0) is now available.


### Table Of Contents
- [Introduction](#introduction)
- [Usage](#usage)
- [Main Results](#main-results)

## Introduction

![Screenshot](https://github.com/DIAGNijmegen/bodyct-dram-emph-subtype/blob/main/figure1.png)

The proposed model can automatically identify severity-based emphysema subtypes according to Fleischner visual scoring system by analyzing a given CT scan. The proposed model outperformed the existing method on the presented dataset with improved interpretability.

## Usage
 - Use `train.py` for training. The training, testing and prediction scripts were all implemented using pytorch, and pytorch-lightning library.
 - Use `processor.py` or `run.sh` for inference. both `processor.py` and `run.sh` require you define the input data (ct image and its lobe segmentation) using `--scan_path` and `--lobe_path`, and the output path using `--output_path` arguments.
 - The code supports inference and training using multiple GPUs. Please use `--ngpus` and `--workers` to specify the number of GPUs and the number of workers for the executation. Check line `60` and `70` for details of possible input arguments.
 - Please check `\install_files\requirements.in` for 3rd-party libraries to be installed to run the scripts. Run `pip install -r install_files/requirements.in` to install dependencies. The code has been tested with python 3.8 version. If you want to install torch with GPU support, please use `--extra-index-url=https://download.pytorch.org/whl/cu113` (chose the cuda version you have already installed, e.g., `11.3` in this example).
 - We provide the classification and regression training strategies. Please switch to `med3d` in `--model_arch` cli argument.
 - The class and regression activation maps were generated during training or testing.
 - For the Grand-challenge [algorithm](https://grand-challenge.org/algorithms/weakly-supervised-emphysema-subtyping/), we use the prediction mode in pytorch-lightning for outputs. 

# Main Results
Tab 1. Centrilobular and ParaseptalEmphysema Severity Scores Classification Accuracy (ACC(%)) and F-measurement, in comparison with the Fleischner algorithm.


|Method     |    Subtype     |   ACC (\%)    | F1-score | Linear Weighted Kappa(95\% CI) |
|:---------:|:--------------:|:-------------:|:--------:|:---------------------:|
|The Fleischner algorithm |      CLE       |      45       |    -     |          60           |
|Ours (classification)  |      CLE       |     52.23     |  51.00   |  64.29 (63.16-65.42)     |
|Ours (classification)  |      PSE       |     59.12     |  57.12   |    42.03 (40.21-43.85)     |
|Ours (regression)|      CLE       |     51.32     |  49.61   |      64.24 (63.14-65.35)       |
|Ours (regression)| PSE| 64.62 |  60.74   |       52.06 (50.40-53.73)         |

## Qualitative Results
The first row showcases the dense regression activation maps (dRAM) for centrilobular emphysema, and the second row illustrates the dRAM for paraseptal emphysema.
![Screenshot](https://github.com/DIAGNijmegen/bodyct-dram-emph-subtype/blob/main/showcase.png)

[MIT](https://choosealicense.com/licenses/mit/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
