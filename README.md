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

# Running the Benchmark
From the `/home/aszot/habitat-lab/habitat-baselines/` directory
```unix
MADRONA_MWGPU_KERNEL_CACHE=/home/aszot/madrona_gpu_tmp CUDA_VISIBLE_DEVICES=0 /home/aszot/miniconda3/envs/madrona2/bin/python ../scripts/time_runs.py
```
Data is then saved to `data/speeds.csv`

# Repro Physics Bugs
From `habitat-lab/habitat-baselines/` run 
```
MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONPATH=/home/aszot/gpu_hideseek/build python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/madrona.yaml --run-type train habitat_baselines.num_environments=192 habitat_baselines.debug_env=True
```
# Training
* Install experiment launcher: `pip install rl-exp-utils`

* Add the following to `~/hideseek.yaml`.
```yaml
base_data_dir: "data/"
ckpt_cfg_key: "CHECKPOINT_FOLDER"
ckpt_append_name: False
add_env_vars:
  - "MAGNUM_LOG=quiet"
  - "HABITAT_SIM_LOG=quiet"
  - "PYTHONPATH=/home/aszot/gpu_hideseek/build"
add_all: "habitat_baselines.wb.entity=$WB_ENTITY habitat_baselines.wb.run_name=$SLURM_ID habitat_baselines.wb.project_name=$PROJECT_NAME habitat_baselines.checkpoint_folder=$DATA_DIR/checkpoints/$SLURM_ID/ habitat_baselines.video_dir=$DATA_DIR/vids/$SLURM_ID/ habitat_baselines.log_file=$DATA_DIR/logs/$SLURM_ID.log habitat_baselines.tensorboard_dir=$DATA_DIR/tb/$SLURM_ID/ habitat_baselines.writer_type=wb"
eval_sys:
  ckpt_load_k: "habitat_baselines.eval_ckpt_path_dir"
  ckpt_search_dir: "checkpoints"
  run_id_k: "habitat_baselines.wb.run_name"
  sep: "="
  eval_run_cmd: "python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/madrona.yaml --run-type train"
  add_eval_to_vals:
    - "habitat_baselines.checkpoint_folder"
    - "habitat_baselines.log_file"
    - "habitat_baselines.video_dir"
    - "habitat_baselines.wb.run_name"
  change_vals:
    "--run-type": "eval"
proj_data:
  eval: "habitat_baselines.num_environments 6"
  test: "habitat_baselines.dry_run=True"
  nowb: "habitat_baselines.writer_type=tb"
```
- Change `PYTHONPATH` to be where your madrona build directory is.
- Add following to your bashrc
```
alias rlt='python -m rl_utils.launcher --cfg ~/hideseek.yaml'
alias rlteval='python -m rl_utils.launcher.eval_sys --cfg ~/hideseek.yaml'
```
From `habitat-lab/habitat-baselines/`:
* For training: `rlt --proj-dat nowb python habitat_baselines/run.py --exp-config habitat_baselines/config/rearrange/madrona.yaml --run-type train`
* For evaluation: `rlteval --runs RUNID --proj-dat eval,nowb`
