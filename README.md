# Re<sup>3</sup>Sim: Generating High-Fidelity Simulation Data via 3D-Photorealistic Real-to-Sim for Robotic Manipulation

<a href="" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-RE3SIM-red?logo=arxiv" height="25" />
</a>
<a href="http://xshenhan.github.io/Re3Sim/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-RE3SIM-blue.svg" height="25" />
</a>
<a href="" target="_blank">
    <img alt="HF Model: RoboVLMs" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-RE3SIM-ffc107?color=ffc107&logoColor=white" height="25" />
</a>
<a href="https://www.python.org/" target="_blank">
    <img alt="Python 3.10" src="https://img.shields.io/badge/Python-%3E=3.10-blue" height="25" />
</a>
<div align="center">
    <br>
<div style="text-align: center;">
    <a href="https://scholar.google.com/citations?hl=en&user=Mo8I5WMAAAAJ"  target="_blank">Xiaoshen Han</a> &emsp;
    <a href="https://minghuanliu.com/"  target="_blank">Minghuan Liu</a><sup>^</sup> &emsp;
    <a href="https://yilunchen.com/about/"  target="_blank">Yilun Chen</a><sup>^&dagger;</sup> &emsp;
    <a href="" target="_blank">Junqiu Yu</a> &emsp;
    <a href="https://shawlyu.github.io" target="_blank">Xiaoyang Lyu</a> &emsp;
    <a href="https://github.com/Nimolty?tab=overview&from=2024-01-01&to=2024-01-31"  target="_blank">Yang Tian</a> &emsp;
    <br>
    <a href=""  target="_blank">Bolun Wang</a> &emsp;
    <a href="https://wnzhang.net" target="_blank">Weinan Zhang</a> &emsp;
    <a href="https://oceanpang.github.io" target="_blank">Jiangmiao Pang</a><sup>&dagger;</sup> &emsp;
    <br>
    <p style="text-align: center; margin-bottom: 0;">
        <span class="author-note"><sup>^</sup>Project lead</span>&emsp;
        <span class="author-note"><sup>&dagger;</sup>Corresponding author</span>
    </p>
<br>
<p style="text-align: center;">
    Shanghai Jiao Tong University &emsp; Shanghai AI Lab &emsp;The University of Hong Kong</p>
</div>
</div>

<hr>



## 📋 Contents

- [Real-to-Sim](#-real-to-sim)
    - [Installation](#installation)
    - [Getting Started](#getting-started)
- [Policy Training](#-policy-training)
    - [Env Setup](#env-setup)
    - [Tutorial](#tutorial)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

## 🏠 Real-to-Sim

### Installation

#### Install the Simulator Environment

We provide a Dockerfile to install the simulator environment. Here is the installation guide:

```shell
docker build -t re3sim:1.0.0 .
```

After the installation, you can run the container:

```shell
docker run --name re3sim --entrypoint bash -itd --runtime=nvidia --gpus='"device=0"' -e "ACCEPT_EULA=Y" --rm --network=bridge --shm-size="32g" -e "PRIVACY_CONSENT=Y" \
    -v /path/to/resources:/root/resources:rw \
    re3sim:1.0.0
```

Install [CUDA 11.8](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run) in the Docker container. Then, install `diff-gaussian-rasterization` and `simple-knn`:

```shell
./cuda_11.8.0_520.61.05_linux.run --silent --toolkit
pip install src/gaussian_splatting/submodules/diff-gaussian-rasterization/
pip install src/gaussian_splatting/submodules/simple-knn/
```

#### install openmvs 

To reconstruct the geometry of the scene, you need to install OpenMVS by following the instructions in the [OpenMVS Wiki](https://github.com/cdcseacave/openMVS/wiki/Building) within the Docker container and add the binary files to the PATH.

After that, we recommend saving the Docker image with `docker commit`.

### Getting Started

#### Real-to-Sim in Predefined Scene

We offer the necessary resources [here](https://huggingface.co/datasets/RE3SIM/sim-resources). You can download them and place them in the following path:

```
<project root>
    - existing code
    - data/
        - align/
        - gs-data/
        - items/
        - urdfs/
        - usd/
```

- **Collect data in simulator**
- collect data for pick and place tasks
```shell
python src/standalone/clean_example/pick_into_basket_lmdb.py
```

- **Visualize the Data**

You can use `utils/checker/check_lmdb_data_by_vis.ipynb` to visualize the data.

#### Real-to-Sim in Customized Scene

1. Prepare the data:

- Place the images in the folder `/path/to/input/images`.
- Place the image for alignment in `/path/to/input/align_image.png`.

2. Reconstruct the scene:

```shell
# in docker
python reconstrct.py -i /path/to/input
```

The scene will be reconstructed automatically.

3. Calibrate and align the scene:

- Run `real-deployment/calibration/hand_in_eye_shooting.ipynb`.

```shell
python real-deployment/calibration/hand_in_eye_calib.py --data_root /path/to/calibrate_folder
python real-deployment/utils/get_marker2base_aruco.py --data_root /path/to/calibrate_folder
```

4. The file `configs/pick_and_place/example.yaml` provides an example of how to configure the required paths in the configuration file.

5. Replace the config path in `src/standalone/clean_example/pick_into_basket_lmdb.py` to begin collecting data in the simulator:

```shell
python src/standalone/clean_example/pick_into_basket_lmdb.py
```

## 🤖 Policy Training
The GitHub repository [act-plus-plus](https://github.com/xshenhan/act-plus-plus) contains our modified ACT code. 

### Env Setup
1. create conda env in `conda_env.yaml`
2. install torch
3. install other modules in `requirements.txt`
4. install detr
```shell
cd detr
pip install -e .
```
### Tutorial
1. Put the data inside `data/5_items_lmdb` and uncompress them. The file structure should look like this:
```
    ├── data
        └── 5_items_lmdb 
            ├── random_position_1021_1_eggplant_low_res_continuous_better_controller2_multi_item_filtered_30_lmdb
                └── ...
            ├── random_position_1021_1_eggplant_low_res_continuous_better_controller2_multi_item_filtered_45_lmdb
                └── ... 
            └── ...
```
2. check data and remove broken files (optional)
```shell
python <path to the data> # use --fast flag to enable a partly check.
```
3. process data to get act dataset:
```shell
# run the command in the act project root dir **/act-plus-plus/
python process_data.py --source-path /path/to/source --output-path /path/to/act_dataset 
```

4.  start training
```python
# Single machine, 8 GPUs
torchrun --nproc_per_node=8 --master_port=12314 imitate_episodes_cosine.py --config-path=conf --config-name=<config name> hydra.job.chdir=True params.num_epochs=24 params.seed=100
# Multi-machine, multi-GPU
# First machine
torchrun --nproc_per_node=8 --node_rank=0 --nnodes=2 --master_addr=<master ip> --master_port=12314 imitate_episodes_cosine.py --config-path=conf --config-name=<config name> hydra.job.chdir=True params.num_epochs=24 params.seed=100

# Second machine
torchrun --nproc_per_node=8 --node_rank=1 --nnodes=2 --master_addr=<master ip> --master_port=12314 imitate_episodes_cosine.py --config-path=conf --config-name=<config name> hydra.job.chdir=True params.num_epochs=24 params.seed=100
```

#### Example
We provide the data of `pick a bottle` task [here](https://huggingface.co/datasets/RE3SIM/act-dataset). And we show how to train the policy with our data. Please download the dataset and place them in `clean/policies/act_plus_plus/lmdb_data` 
```shell
python process_data.py --source-path /path/to/act_plus_plus/lmdb_data/pick_one --output-path /path/to/act_plus_plus/data/pick_one
# The configs are in `constants.py` and `conf/pick_one_into_basket.yaml`
torchrun --nproc_per_node=8 --master_port=12314 imitate_episodes_cosine.py --config-path=conf --config-name=pick_one_into_basket hydra.job.chdir=True params.num_epochs=24 params.seed=100
```

## 📝 TODO List

- \[ \] Polish related codes.
- \[ \] More tasks.
- \[ \] A user-friendly GUI.
- \[ \] Unified rendering implementation and articulation reconstrction.

## 🔗 Citation

If you find our work helpful, please cite:

```latex
TODO
```

## 📄 License

The work is licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">Creative Commons Attribution-NonCommercial 4.0 International</a>

## 👏 Acknowledgements

- [Gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting): We use 3D Gaussian splatting as the rendering engine.
- [Act-plus-plus](https://github.com/MarkFzp/act-plus-plus): We modify the ACT model based on the code.
- [Franka_grasp_baseline](https://github.com/jimazeyu/franka_grasp_baseline). We borrowed the code of the hand-eye calibration implementation from this codebase.
- [IsaacLab](https://github.com/isaac-sim/IsaacLab): We used the script from this library to convert OBJ to USD.