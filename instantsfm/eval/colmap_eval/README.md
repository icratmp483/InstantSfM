# Usage for colmap_eval module  
This module is used to evaluate the results of several SfM pipelines compared to given ground truth. Only AUC is tested for now.  
## 1. Install dependencies  
```bash
conda create -n eval python=3.12
conda activate eval
pip install -r instantsfm/eval/colmap_eval/requirements.txt
```
## 2. Prepare the results  
Place reconstruction result of several methods to specified place (by default `dataset/`). The results should be organized in a specific way as described below. Currently you need to manually place the results in the `dataset` directory.  
The `dataset` directory should be organized as follows:  

- Each dataset is placed in the `dataset` directory and named after the dataset's name.  
- Inside each dataset folder, there are fixed categories, and each category is named accordingly.  
- Within each category folder, there are multiple scene folders. Each scene folder follows the COLMAP storage format, which includes:  
  - An `images` folder containing the images for the scene.  
  - One or more `sparse` folders from different reconstruction methods used for computing metrics.  

Ensure that the directory structure adheres to this format for proper evaluation.  
There are different names for supported datasets, and we will show their default folder structure one by one. Note that the name of categories and scenes are not specified and you can add any number of them, while the dataset names are currently hardcoded. Extra files in scene folders are also allowed (usually files like database.db or other files included in the original dataset), so there's no need to delete them.  
- **ETH3D**:  
```
- eth3d
  - dslr
    - botanical_garden
      - dslr_calibration_undistorted
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - mvs
    - ...(similarly)
```
- **Tanks and Temples**:  
```
- tt
  - Advanced
    - Auditorium
      - cams_1
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - Intermediate
    - ...(similarly)
```
- **DTU**:  
```
- dtu
  - dtu_testing
    - scan1
      - cams
      - images
      - sparse_colmap
      - sparse_glomap
      - sparse
    - ...(similarly)
  - dtu_training
    - ...(similarly)
```
