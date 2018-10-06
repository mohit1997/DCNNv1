## TL;DR

`python stacked_pymodel_middle.py`

Set the path for training/test data in main function  

**NB1**: Tested ONLY on python 3.5, pytorch 0.4 and GTX960M / GTX1080 TI GPU.

## Overview

### Training the model

The main model code can be found in [`pymodel_multispeakers.py`](pymodel_multispeakers.py) which contains the network **MyNet**, as well as the train and test code. Resumes are also supported from previously saved checkpoints.

The main aspects of training are outlined below:

- Data loading: The _main_ function reads both the train (variable `train_data`) and test data (`test_data`), via calls the data loading routine `create_dataloader`. The path to the data is provided is provided as the first (path to the file containing the speech signal) and second (path to the file containing gci locations) argument to this function. The number of files chosen for training can be set via the **select** argument.
- Checkpointing: The _main_ function also initializes the `Saver` object (`save_model`) that checkpoints at a preset interval (5 **epochs** by default). The path to the **checkpoint directory** needs to be set at initialization.
- Hyperparameter Tuning: Basic hyperparameter tuning for number of epochs, learning rate (and its schedule) etc can be directly tweaked from the main function. Adjustment of loss weights can be done by specifying optional arguments for the `train` or `test` functions. The **window and subwindow** size is set in the calls to the create_dataloader function.

### Data Description

The data provided with this model is hierarchically organized first on the basis of speaker, then noise type and finally noise added.

An example would be:

```
CMU  
|- bdl  
    |- bdl_peaks: contains the gci locations  
    |- bdl_speech: contains preprocessed, cleaned, **low pass filtered** speech
    |- noise_bdl: contains subfolders for various SNR speech signals
        |- 0: contains additive 0 dB noise speech signals
            |- babble: speech signals having additive 0 dB babble noise
            |- white: speech signals having additive 0 dB white noise
        |- 5:  contains additive 5 dB noise speech signals
            |- babble: speech signals having additive 5 dB babble noise
            |- white: speech signals having additive 5 dB white noise
        |- 10:  contains additive 10 dB noise speech signals
            |- babble: speech signals having additive 10 dB babble noise
            |- white: speech signals having additive 10 dB white noise
        |- 15:  contains additive 15 dB noise speech signals
            |- babble: speech signals having additive 15 dB babble noise
            |- white: speech signals having additive 15 dB white noise
        |- 20:  contains additive 20 dB noise speech signals
            |- babble: speech signals having additive 20 dB babble noise
            |- white: speech signals having additive 20 dB white noise
        |- 25:  contains additive 25 dB noise speech signals
            |- babble: speech signals having additive 25 dB babble noise
            |- white: speech signals having additive 25 dB white noise
|- bdl_speech_raw: contains preprocessed RAW (non low pass filtered speech)
```