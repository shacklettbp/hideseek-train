# Installation
* Install [GPU rearrange](https://github.com/shacklettbp/gpu_hideseek).
* Install Habitat-Sim
    * `git clone https://github.com/facebookresearch/habitat-sim.git`
    * `cd habitat-sim`
    * `pip install -r requirements.txt`
    * `python setup.py install --headless --bullet`
* Install Habitat Lab
    * `git clone -b madrona https://github.com/ASzot/habitat-lab.git`
    * `cd habitat-lab`
    * `pip install -e habitat-lab`
    * `pip install -e habitat-baselines`

From `habitat-lab/habitat-baselines/` run 
```
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONPATH=/home/aszot/gpu_hideseek/build python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/madrona.yaml --run-type train habitat_baselines.num_environments=192 habitat_baselines.debug_env=True
```

