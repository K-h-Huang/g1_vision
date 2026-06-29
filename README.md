# G1 29DoF Depth MuJoCo Deployment Test

This folder is a standalone Python MuJoCo test harness for the G1 29DoF parkour
policy. The robot XML/meshes and policy files are copied into this folder, so it
does not depend on `FeapVision_Mujoco_deployment` or `hiking-in-the-wild-sim2sim`
at runtime.

It follows the deployment flow used by `FeapVision_Mujoco_deployment`:

1. run MuJoCo at a small simulation step,
2. update a PD target every policy step,
3. build proprioceptive observations with history,
4. render `depth_cam`,
5. encode the 8-frame depth stack with `0-depth_encoder.onnx`,
6. replace the raw depth tail of the observation with the encoder latent,
7. run `actor.onnx` and apply the joint-position action.

## Conda Environment

Create the environment from the repository root:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate g1-vision-mujoco
```

If you prefer to create the environment manually:

```bash
conda create -n g1-vision-mujoco python=3.10 pip -y
conda activate g1-vision-mujoco
pip install -r requirements.txt
```

## Quick Verification

Run this first on a new machine. It loads the MuJoCo model, depth encoder, actor
ONNX model, and runs a short headless simulation:

```bash
python deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --headless --duration 5
```

Expected startup output includes ONNX input/output signatures, the MuJoCo joint
order, and:

```text
Loaded depth encoder: ...
  inputs: [('input', [1, 8, 18, 32])]
Loaded actor: ...
  inputs: [('input', [1, 896])]
Policy observation dim before encoder: 5376
Depth tail dim: 4608
```

## Run With Viewer

After the headless check passes:

```bash
python deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml
```

Useful options:

```bash
python deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --headless --duration 5
python deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --cmd 0.5 0.0 0.0
python deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --no-depth-window
```

## Repository Layout

The default configuration is self-contained and uses:

- `g1/g1_29dof_with_camera.xml`
- `g1/meshes/`
- `parkour_policy/params/deploy.yaml`
- `parkour_policy/exported/0-depth_encoder.onnx`
- `parkour_policy/exported/actor.onnx`
- `parkour_policy/data/depth_obs.csv`

## Notes

- Run commands from this repository directory, not from the parent workspace.
- On Linux servers without a display, use `--headless`.
- On Linux, if OpenGL/EGL libraries are missing, install the system MuJoCo
  runtime dependencies for your distribution before running the viewer.
- The policy uses a fixed velocity command from `configs/g1_29dof_depth.yaml`
  unless `--cmd VX VY WZ` is provided.
