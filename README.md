# CNN-LSTM-Transformer (CLT) for EEG motor imagery decoding

## Abstract

To improve the classification accuracy of motor imagery EEG signals, In this research I propose the CLT model, which combines three architectures: Convolutional Neural Network (CNN), Long-short term memory (LSTM), and Transformer.

- CNN is used to extract local spatial and temporal features
- Leveraging the high temporal resolution of EEG signals, optimally capture temporal information using Long Short-Term Memory (LSTM)
- Global features are integrated through the Transformer's self-attention mechanism

![CLT Architecture](images/CLT_Structure_white.png)

## Datasets

### [BCI Competition IV 2a](https://www.bbci.de/competition/iv/desc_2a.pdf)  

- 9 subjects
- Tasks: Left Hand, Right Hand, Both Feet, Tongue
- Data file: gdf file, 288x22x1000 (trials x channels x sample) 
- True Labels: Mat file, 288x1 (trials x 1)
- [Download Dataset](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip) and [True Labels](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip)

### [BCI Competition IV 2b](https://www.bbci.de/competition/iv/desc_2b.pdf)

- 9 subjects
- Tasks: Left Hand, Right Hand
- Data file: gdf file, 120x3x1000 (trials x channels x sample) 
- True Labels: Mat file, 120x1 (trials x 1)
- [Download Dataset](https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip) and [True Labels](https://www.bbci.de/competition/iv/results/ds2b/true_labels.zip)

### [Physionet EEG Motor Movement/Imagery Dataset](https://www.physionet.org/content/eegmmidb/1.0.0/)

- 100 subjects
- Tasks: Left Fist, Right Fist, Both Fist, Both Feet
- Data file: edf file, 84x64x640 (trials x channels x sample) 
- True Labels: edf file, 84x1 (trials x 1)

## Analysis

- The 3 models EEGNet  [[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)], EEGConformer  [[paper](https://ieeexplore.ieee.org/document/9991178), [original code](https://github.com/eeyhsong/EEG-Conformer)], and CLT (the proposed model) are defined in files in the Model folder
- The files [Within_Subj_Main.py](programs/Within_Subj_Main.py), [LOSO_Main.py](programs/LOSO_Main.py), and [Physionet_Main.py](programs/Physionet_Main.py) perform training and testing according to three evaluation approaches: Within Subject, Leave one subject out (LOSO), and Leave multiple subjects out (LMSO)
- After downloading BCI Competition IV 2a, BCI Competition IV 2b, and Physionet EEG Motor Movement/Imagery Datasets, the data path should be set in the 'data_path' variable, and model [CLT, EEGNet, Conformer] should be put in the 'Model_name' variable in 3 xx_Main.py files

## Requirement

Models were trained and tested by a single Nvidia RTX 4070 Ti 12GB GPU with CUDA version 11.8

The list of required packages is as follows:

- Python 3.10.14
- Pytorch 2.3.0
- MNE 1.8.0
- numpy 1.26.3
- scipy 1.13.1
- sklearn 1.5.0
