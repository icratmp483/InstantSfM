# InstantSfM
InstantSfM is a novel global SfM system completed in CUDA/Triton and PyTorch.  

## 1. Installation  
**Note: The project requires an NVIDIA GPU with CUDA support. The code is tested on Ubuntu 20.04 with CUDA 12.1 and PyTorch 2.3.1.** 
**Windows system is strongly unrecommended as the pyro_slam package lacks support for Windows.**  
Create a conda environment:  
```bash
conda create -n instantsfm python=3.12
conda activate instantsfm
```
Install PyTorch dependencies, use the following command to install the recommended version of PyTorch, or choose your own version according to your CUDA version:  
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```
If scikit-sparse installation fails due to suitesparse, this dependency shall be installed manually. For example, 
```bash
conda install -c conda-forge suitesparse
# Linux
export SUITESPARSE_INCLUDE_DIR=$CONDA_PREFIX/include/suitesparse
export SUITESPARSE_LIBRARY_DIR=$CONDA_PREFIX/lib
# Windows
export SUITESPARSE_INCLUDE_DIR=$CONDA_PREFIX\Library\include\suitesparse
export SUITESPARSE_LIBRARY_DIR=$CONDA_PREFIX\Library\lib
```
Then you can install instantsfm locally by running:  
```bash
pip install -e .
```
Install pyro_slam by running:
```bash
pip install -e pyro_slam/
```
If you find error like
```bash
fatal error: cudss.h: No such file or directory
   10 | #include <cudss.h>
      |          ^~~~~~~~~
compilation terminated
```
then you need to download and install cuDSS package from [here](https://developer.nvidia.com/cudss-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local).

If opencv-python fail to load xcb, you can install opencv-python-headless
```bash
pip install opencv-python-headless
```

By default feature extraction is done via COLMAP, which requires you to install COLMAP first. You can follow the instructions [here](https://colmap.github.io/install.html) to install COLMAP. Make sure the `colmap` command is available in your terminal.  

## 2. Demo  
To run the demo, simply try the command `python demo.py`. In the demo, you can choose to reconstruct either from user-provided images or from a image directory. A valid input image directory should follow the structure shown below:  
```
- demo_input_folder/
    - images/
    - database.db (optional, will be used if provided)
```
In both cases, the output will be saved in the corresponding folder(`demo_output/` or your specified folder), and the results will be displayed directly in the web viewer.  

## 3. Command Line Usage
The whole pipeline consists of three main steps: feature extraction and matching, global structure from motion (SfM), and 3DGS training.  
Before performing these steps, prepair your dataset (a collection of images) in a folder structure like mentioned in the demo section, that is, a folder containing a subfolder `images/` with all the images inside.  
To extract features and perform matching, use the following command:
```bash
ins-feat --data_path /path/to/folder
```
To run the global SfM and bundle adjustment, use:  
```bash
ins-sfm --data_path /path/to/folder
```
We also provided 3DGS training support, and you can run it with the following command:
```bash
ins-gs --data_path /path/to/folder
```
For a more detailed usage, you can run the command with `--help` to see all available options.  

## 4. Manual configuration   
While the default configuration should work for most cases, you can also try to modify the configuration in the `config/` folder to improve the performance on your own dataset.  
Want to apply several modifications to config files while keeping the original ones? Add the `--manual_config_name` argument when invoking `glomap.py` and specify the name of your own config file. For example, if you created a new config file `config/my_config.py`, add `--manual_config_name my_config` to the command line. Please make sure the config file is a valid one, the recommended way is to copy an original config file and modify it.  