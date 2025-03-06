## Install the Real-to-Sim environment mannally

> We suppose the Re3Sim folder is in `<project root>`.

1. Install isaac-sim:4.0.0 following offical instructions in `<project root>/isaac-sim`

2. Install [isaac-lab:v1.1.0](https://github.com/isaac-sim/IsaacLab/archive/refs/tags/v1.1.0.zip) following offical instructions in `<project root>/IsaacLab`

3. Install OpenMVS in `<project root>/OpenMVS` following offical instructions in `<project root>/OpenMVS`. (If you want to use OpenMVS to get the geometry of the scene, you need to install it.)

4. In `<project root>` run the following command:
    ```bash
    mkdir dev
    cd dev
    ln -s ../Re3Sim/re3sim real2sim2real
    cp ../Re3Sim/re3sim/setup.py .
    pip install -e .
    ```

5. Install cuda-11.8.

6. Install other dependencies:
    ```bash
    # in <project root>/isaac-sim/
    ln -s ../Re3Sim/re3sim src
    pip install src/gaussian_splatting/submodules/diff-gaussian-rasterization/
    pip install src/gaussian_splatting/submodules/simple-knn/
    pip install -r src/requirements.txt
    ```

> Note: Then you should modify the `data_log_root_path` in the config file to the path of the data you want to use. You can also change the headless to False in the config file if you want to visualize the simulation.