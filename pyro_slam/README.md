# torch_slam

## Dev notes
- solver for block sparse matrix [1](https://github.com/ROCm/rocm-examples/tree/develop/Libraries/rocSPARSE/level_2/bsrsv/) [2](https://pytorch.org/docs/stable/generated/torch.triangular_solve.html)

## Quick setup instruction
The GPU version of this repo is to compile localy with.
- Step 1: Install cudss. Recommend to install through package manager. 
- Step 2: Clone this repository and open as vscode workspace.
- Step 3: Install dependency `pip install -r requirements.txt`
- Step 4: `pip install -e .`
- Step 5: Run `ba_example.py`

## 1dsfm
```bash
ram_cache=/dev/shm/zitongz3
core_count=0
for scene in Alamo         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    core_count_end=$((core_count+4))
    mkdir -p $ram_cache/$scene
    mkdir -p $ram_cache/$scene/sparse
    rm -rf $ram_cache/$scene/sparse/*
    taskset -c $core_count-$core_count_end nohup ~/colmap/build/src/colmap/exe/colmap mapper --image_path /scratch/zitongz3/1dsfm_data/$scene --database_path /scratch/zitongz3/1dsfm_cache/$scene/database.db --output_path $ram_cache/$scene/sparse  --Mapper.ba_global_max_num_iterations 0 --Mapper.ba_local_max_num_iterations 0 --Mapper.ba_global_max_refinements 0 --Mapper.ba_local_max_refinements 0 --Mapper.ba_local_max_refinement_change 0.0 --Mapper.ba_global_max_refinement_change 0.0 --Mapper.multiple_models 0 --Mapper.max_num_models 1 --Mapper.filter_max_reproj_error 0.8 --Mapper.filter_min_tri_angle 15.0 &
    core_count=$((core_count_end+1))
done
```

```bash
for scene in Alamo         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    echo $scene
    python /home/csgrad/zzhan4/colmap/scripts/python/read_write_model.py --input_model $ram_cache/$scene/sparse/0/
done
```



--SiftExtraction.peak_threshold 0.02 
Feature extraction and matching
```bash
rm -r $ram_cache/$scene
mkdir $ram_cache/$scene
colmap feature_extractor  --SiftExtraction.use_gpu 0 --database_path $ram_cache/$scene/database.db --image_path /scratch/zitongz3/1dsfm_data/$scene --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true 

colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path $ram_cache/$scene/database.db --SiftMatching.guided_matching=true

# 
mkdir -p $ram_cache/$scene/sparse

~/colmap/build/src/colmap/exe/colmap mapper --image_path /scratch/zitongz3/1dsfm_data/$scene --database_path $ram_cache/$scene/database.db --output_path $ram_cache/$scene/sparse  --Mapper.ba_global_max_num_iterations 0 --Mapper.ba_local_max_num_iterations 0 --Mapper.ba_global_max_refinements 0 --Mapper.ba_local_max_refinements 0 --Mapper.ba_local_max_refinement_change 0.0 --Mapper.ba_global_max_refinement_change 0.0 --Mapper.multiple_models 0 --Mapper.max_num_models 1

python /home/csgrad/zzhan4/colmap/scripts/python/read_write_model.py --input_model $ram_cache/$scene/sparse/0/
```


## control num features
```bash
export CUDA_VISIBLE_DEVICES=0,2,7
ram_cache=/dev/shm/zitongz3
core_count=0
nfeat=500
for scene in         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    core_count_end=$((core_count+4))
    taskset -c $core_count-$core_count_end nohup \
    python colmap_helpers/incremental_helper.py \
    --image_path /scratch/zitongz3/1dsfm_data/$scene \
    --result_folder $ram_cache/$nfeat/$scene \
    --max_num_features $nfeat &> nohup$nfeat$scene.out &
    core_count=$((core_count_end+1))
done
```


```bash
nfeat=500
for scene in         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    echo $nfeat
    ls -al $ram_cache/$nfeat/$scene/sparse/0/cameras.bin
    python /home/csgrad/zzhan4/colmap/scripts/python/read_write_model.py --input_model $ram_cache/$nfeat/$scene/sparse/0/
done
```

```bash
core_count=0
nfeat=500
for scene in         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    core_count_end=$((core_count+4))
    echo $nfeat
    ls -al $ram_cache/$nfeat/$scene/sparse/0/cameras.bin
    taskset -c $core_count-$core_count_end nohup \
    python colmap_helpers/colmap2bal.py --input_path $ram_cache/$nfeat/$scene/sparse/0/ --output_path ./1dsfm_bal/$scene.txt &> nohup$nfeat$scene.out &
    core_count=$((core_count_end+1))
done
```

Run gtsam baseline
```bash
for scene in         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    echo $scene
    python ./python/gtsam/examples/SFMExample_bal.py -i ./1dsfm_bal/$scene.txt
done
```

Run DeepLM baseline

dataset_root=/home/zitongzhan/Documents/pyro_slam/1dsfm_bal
```bash
for scene in         Gendarmenmarkt     Montreal_Notre_Dame  Piazza_del_Popolo  Roman_Forum      Trafalgar     Vienna_Cathedral Ellis_Island  Madrid_Metropolis  NYC_Library          Piccadilly         Tower_of_London  Union_Square  Yorkminster
do
    echo $scene
    TORCH_USE_RTLD_GLOBAL=YES python3 examples/BundleAdjuster/bundle_adjuster.py --balFile /1dsfm/$scene.txt
done
```