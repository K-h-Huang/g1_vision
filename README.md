# G1 29DoF depth MuJoCo deployment test

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

## Requirements

Install the runtime packages in the Python environment you use for MuJoCo:

```bash
pip install mujoco onnxruntime opencv-python pyyaml numpy
```

## Run

From the workspace root:

```bash
python g1_29dof_depth_mujoco_deployment/deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml
```

Useful options:

```bash
python g1_29dof_depth_mujoco_deployment/deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --headless --duration 5
python g1_29dof_depth_mujoco_deployment/deploy_g1_29dof_depth.py configs/g1_29dof_depth.yaml --cmd 0.5 0.0 0.0
```

Use `--headless --duration 5` first to verify ONNX signatures and observation
dimensions before opening the MuJoCo viewer.

The default configuration uses:

- `g1/g1_29dof_with_camera.xml`
- `g1/meshes/`
- `parkour_policy/params/deploy.yaml`
- `parkour_policy/exported/0-depth_encoder.onnx`
- `parkour_policy/exported/actor.onnx`
- `parkour_policy/data/depth_obs.csv`
