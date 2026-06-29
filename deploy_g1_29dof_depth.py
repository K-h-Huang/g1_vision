from __future__ import annotations

import argparse
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import onnxruntime as ort
import yaml


DEPLOY_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = DEPLOY_DIR.parent


class DeploymentYamlLoader(yaml.FullLoader):
    pass


def _construct_python_tuple(loader: DeploymentYamlLoader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node))


def _construct_python_slice(loader: DeploymentYamlLoader, node: yaml.Node) -> slice:
    return slice(*loader.construct_sequence(node))


DeploymentYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)
DeploymentYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    _construct_python_slice,
)


def resolve_path(path_text: str) -> Path:
    text = path_text.replace("{DEPLOY_DIR}", str(DEPLOY_DIR))
    text = text.replace("{WORKSPACE_ROOT}", str(WORKSPACE_ROOT))
    return Path(text).expanduser().resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f, Loader=DeploymentYamlLoader)


def load_config(config_arg: str) -> dict[str, Any]:
    path = Path(config_arg)
    if not path.is_file():
        path = DEPLOY_DIR / config_arg
    if not path.is_file():
        path = DEPLOY_DIR / "configs" / config_arg
    cfg = load_yaml(path.resolve())
    for key in (
        "robot_xml_path",
        "deploy_yaml_path",
        "actor_onnx_path",
        "depth_encoder_onnx_path",
    ):
        cfg[key] = str(resolve_path(cfg[key]))
    return cfg


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    q_vec = np.array([qx, qy, qz], dtype=np.float32)
    a = v * (2.0 * qw * qw - 1.0)
    b = np.cross(q_vec, v) * (-2.0 * qw)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return a + b + c


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse(quat_wxyz.astype(np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32))


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    return (target_q - q) * kp - dq * kd


G1_CAMERA_JOINT_SIGNS = np.asarray(
    [
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
    dtype=np.float32,
)


G1_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def xml_vector(values: list[float]) -> str:
    return " ".join(f"{float(v):.8g}" for v in values)


def feapvision_terrain_geoms() -> list[dict[str, Any]]:
    geoms: list[dict[str, Any]] = [
        {
            "name": "floor",
            "type": "plane",
            "pos": [0.0, 0.0, 0.0],
            "size": [0.0, 0.0, 0.05],
            "rgba": [0.45, 0.45, 0.45, 1.0],
        }
    ]

    def box(name: str, x: float, y: float, z: float, sx: float, sy: float, sz: float) -> None:
        geoms.append(
            {
                "name": name,
                "type": "box",
                "pos": [x, y, z],
                "size": [sx, sy, sz],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "rgba": [0.58, 0.48, 0.34, 1.0],
            }
        )

    # Same obstacle layout as FeapVision_Mujoco_deployment/resources/.../scene_terrain.xml:
    # wide stairs up/down, a long low ramp, and rear platforms.
    for i in range(8):
        h = 0.08 * (i + 1)
        box(f"feap_stair_up_{i:02d}", 1.165 + 0.33 * i, 0.0, h, 0.165, 1.75, h)

    box("feap_stair_top", 4.14, 0.0, 0.64, 0.5, 1.75, 0.64)

    for i in range(7):
        h = 0.56 - 0.08 * i
        box(f"feap_stair_down_{i:02d}", 4.805 + 0.33 * i, 0.0, h, 0.165, 1.75, h)

    ramp_heights = [0.008333333333333333 * (i + 1) for i in range(30)]
    for i, h in enumerate(ramp_heights):
        box(f"feap_ramp_up_{i:02d}", 8.05 + 0.1 * i, 0.0, h, 0.05, 2.0, h)

    box("feap_ramp_top", 11.5, 0.0, 0.25, 0.5, 2.0, 0.25)

    for i, h in enumerate(reversed(ramp_heights[:-1])):
        box(f"feap_ramp_down_{i:02d}", 12.05 + 0.1 * i, 0.0, h, 0.05, 2.0, h)

    box("feap_rear_platform_0", -2.0, 0.0, 0.15, 1.0, 1.75, 0.15)
    box("feap_rear_platform_1", -4.0, 0.0, 0.15, 0.5, 1.75, 0.15)
    return geoms


def make_scene_xml(robot_xml_path: Path, extra_geoms: list[dict[str, Any]]) -> Path:
    robot_tree = ET.parse(robot_xml_path)
    root = robot_tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        meshdir = Path(compiler.get("meshdir", ""))
        if not meshdir.is_absolute():
            compiler.set("meshdir", str((robot_xml_path.parent / meshdir).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    for geom in extra_geoms:
        if worldbody.find(f"./geom[@name='{geom['name']}']") is not None:
            continue
        elem = ET.SubElement(worldbody, "geom")
        elem.set("name", geom["name"])
        elem.set("type", geom["type"])
        elem.set("pos", xml_vector(geom.get("pos", [0, 0, 0])))
        elem.set("size", xml_vector(geom["size"]))
        if "quat" in geom:
            elem.set("quat", xml_vector(geom["quat"]))
        if "rgba" in geom:
            elem.set("rgba", xml_vector(geom["rgba"]))

    scene_path = Path(tempfile.gettempdir()) / "g1_29dof_depth_scene.xml"
    robot_tree.write(scene_path, encoding="utf-8", xml_declaration=False)
    return scene_path


def mujoco_indices_for_policy_order(
    model: mujoco.MjModel, policy_joint_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    policy_joint_names = [G1_SDK_JOINT_NAMES[int(motor_id)] for motor_id in policy_joint_ids]
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    ctrl_indices: list[int] = []

    for joint_name in policy_joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Joint not found in MuJoCo model: {joint_name}")
        qpos_indices.append(int(model.jnt_qposadr[joint_id]))
        qvel_indices.append(int(model.jnt_dofadr[joint_id]))

        actuator_id = -1
        for aid in range(model.nu):
            if int(model.actuator_trnid[aid, 0]) == joint_id:
                actuator_id = aid
                break
        if actuator_id < 0:
            raise RuntimeError(f"Actuator not found for joint: {joint_name}")
        ctrl_indices.append(actuator_id)

    return (
        np.asarray(qpos_indices, dtype=np.int32),
        np.asarray(qvel_indices, dtype=np.int32),
        np.asarray(ctrl_indices, dtype=np.int32),
        policy_joint_names,
    )


class OrtModule:
    def __init__(self, model_path: str, label: str):
        self.label = label
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        print(f"Loaded {label}: {model_path}")
        print(f"  inputs: {[(i.name, i.shape) for i in self.inputs]}")
        print(f"  outputs: {[(o.name, o.shape) for o in self.outputs]}")

    def _reshape_input(self, data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        shape = self.inputs[0].shape
        if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
            return arr

        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(
                f"{self.label} expects input shape {shape} ({expected} values), "
                f"got {arr.shape} ({arr.size} values)"
            )
        if tuple(arr.shape) != tuple(shape):
            arr = arr.reshape(shape)
        return arr

    def run_single(self, data: np.ndarray, preferred_name: str = "obs") -> np.ndarray:
        name = self.inputs[0].name
        feed = {name: self._reshape_input(data)}
        out = self.session.run(None, feed)[0]
        return np.asarray(out, dtype=np.float32)


class G1DepthPolicy:
    def __init__(self, actor_path: str, encoder_path: str, depth_dim: int):
        self.encoder = OrtModule(encoder_path, "depth encoder")
        self.actor = OrtModule(actor_path, "actor")
        self.depth_dim = depth_dim

    def infer(self, full_obs: np.ndarray) -> np.ndarray:
        if full_obs.ndim != 2 or full_obs.shape[0] != 1:
            raise ValueError(f"full_obs must be [1, D], got {full_obs.shape}")
        if full_obs.shape[1] <= self.depth_dim:
            raise ValueError(
                f"full_obs length {full_obs.shape[1]} is not larger than depth dim {self.depth_dim}"
            )

        offset = full_obs.shape[1] - self.depth_dim
        depth_obs = full_obs[:, offset:]
        depth_latent = self.encoder.run_single(depth_obs)
        actor_obs = np.concatenate([full_obs[:, :offset], depth_latent.reshape(1, -1)], axis=1)
        action = self.actor.run_single(actor_obs)
        return action.reshape(-1)


class HistoryTerm:
    def __init__(self, history_length: int, scale: list[float] | None = None):
        self.history_length = int(history_length)
        self.scale = None if scale is None else np.asarray(scale, dtype=np.float32)
        self.buffer: deque[np.ndarray] = deque(maxlen=self.history_length)

    def reset(self, value: np.ndarray) -> None:
        self.buffer.clear()
        for _ in range(self.history_length):
            self.add(value)

    def add(self, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=np.float32).copy()
        if self.scale is not None:
            arr *= self.scale
        self.buffer.append(arr)

    def concat(self) -> np.ndarray:
        return np.concatenate(list(self.buffer), axis=0)


class G1ObservationBuilder:
    def __init__(self, deploy_cfg: dict[str, Any], depth_dim: int):
        obs_cfg = deploy_cfg["observations"]
        self.terms = {
            "base_ang_vel": HistoryTerm(
                obs_cfg["base_ang_vel"]["history_length"],
                obs_cfg["base_ang_vel"]["scale"],
            ),
            "projected_gravity": HistoryTerm(
                obs_cfg["projected_gravity"]["history_length"],
                obs_cfg["projected_gravity"]["scale"],
            ),
            "velocity_commands": HistoryTerm(
                obs_cfg["velocity_commands"]["history_length"],
                obs_cfg["velocity_commands"]["scale"],
            ),
            "joint_pos_rel": HistoryTerm(
                obs_cfg["joint_pos_rel"]["history_length"],
                obs_cfg["joint_pos_rel"]["scale"],
            ),
            "joint_vel_rel": HistoryTerm(
                obs_cfg["joint_vel_rel"]["history_length"],
                obs_cfg["joint_vel_rel"]["scale"],
            ),
            "last_action": HistoryTerm(
                obs_cfg["last_action"]["history_length"],
                obs_cfg["last_action"]["scale"],
            ),
        }
        self.depth_dim = depth_dim
        self.depth = np.zeros(depth_dim, dtype=np.float32)

    def reset(
        self,
        omega: np.ndarray,
        gravity: np.ndarray,
        cmd: np.ndarray,
        joint_pos_rel: np.ndarray,
        joint_vel: np.ndarray,
        action: np.ndarray,
        depth: np.ndarray,
    ) -> None:
        values = {
            "base_ang_vel": omega,
            "projected_gravity": gravity,
            "velocity_commands": cmd,
            "joint_pos_rel": joint_pos_rel,
            "joint_vel_rel": joint_vel,
            "last_action": action,
        }
        for name, value in values.items():
            self.terms[name].reset(value)
        self.depth = depth.astype(np.float32).copy()

    def add(
        self,
        omega: np.ndarray,
        gravity: np.ndarray,
        cmd: np.ndarray,
        joint_pos_rel: np.ndarray,
        joint_vel: np.ndarray,
        action: np.ndarray,
        depth: np.ndarray,
    ) -> np.ndarray:
        self.terms["base_ang_vel"].add(omega)
        self.terms["projected_gravity"].add(gravity)
        self.terms["velocity_commands"].add(cmd)
        self.terms["joint_pos_rel"].add(joint_pos_rel)
        self.terms["joint_vel_rel"].add(joint_vel)
        self.terms["last_action"].add(action)
        self.depth = depth.astype(np.float32).copy()
        return self.full_obs()

    def full_obs(self) -> np.ndarray:
        ordered = [
            self.terms["base_ang_vel"].concat(),
            self.terms["projected_gravity"].concat(),
            self.terms["velocity_commands"].concat(),
            self.terms["joint_pos_rel"].concat(),
            self.terms["joint_vel_rel"].concat(),
            self.terms["last_action"].concat(),
            self.depth,
        ]
        return np.concatenate(ordered, axis=0).reshape(1, -1).astype(np.float32)


def joint_name_order(model: mujoco.MjModel) -> list[str]:
    names = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return names


def depth_image(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_id: int,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera_id)
    raw = renderer.render()
    if int(cfg.get("depth_vis_rotate_k", 0)):
        raw_vis = np.rot90(raw, k=int(cfg["depth_vis_rotate_k"]))
    else:
        raw_vis = raw.copy()
    small = cv2.resize(
        raw,
        (int(cfg["depth_width"]), int(cfg["depth_height"])),
        interpolation=cv2.INTER_AREA,
    )
    if float(cfg.get("depth_noise_amp", 0.0)) > 0:
        small += float(cfg["depth_noise_amp"]) * np.random.randn(*small.shape)
    clipped = np.clip(small, float(cfg["depth_min_range"]), float(cfg["depth_max_range"]))
    normalized = (clipped - float(cfg["depth_min_range"])) / (
        float(cfg["depth_max_range"]) - float(cfg["depth_min_range"])
    )
    return normalized.astype(np.float32).reshape(-1), raw_vis


def make_depth_history_buffer(first_frame: np.ndarray, depth_history: int) -> deque[np.ndarray]:
    buf: deque[np.ndarray] = deque(maxlen=depth_history)
    for _ in range(depth_history):
        buf.append(first_frame.copy())
    return buf


def depth_history_vector(buffer: deque[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(buffer), axis=0).astype(np.float32)


def show_depth(raw_depth: np.ndarray, cfg: dict[str, Any]) -> None:
    disp = np.clip(raw_depth, float(cfg["depth_min_range"]), float(cfg["depth_max_range"]))
    disp = (disp - float(cfg["depth_min_range"])) / (
        float(cfg["depth_max_range"]) - float(cfg["depth_min_range"])
    )
    disp_u8 = (disp * 255.0).astype(np.uint8)
    scale = int(cfg.get("depth_window_scale", 10))
    disp_u8 = cv2.resize(
        disp_u8,
        (disp_u8.shape[1] * scale, disp_u8.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    color = cv2.applyColorMap(disp_u8, cv2.COLORMAP_TURBO)
    cv2.namedWindow("g1 depth_cam", cv2.WINDOW_NORMAL)
    cv2.imshow("g1 depth_cam", color)
    cv2.waitKey(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1 29DoF MuJoCo depth policy deployment test")
    parser.add_argument("config", nargs="?", default="configs/g1_29dof_depth.yaml")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer")
    parser.add_argument("--no-depth-window", action="store_true", help="Disable OpenCV depth preview")
    parser.add_argument("--duration", type=float, default=None, help="Override simulation duration in seconds")
    parser.add_argument("--cmd", nargs=3, type=float, metavar=("VX", "VY", "WZ"), help="Fixed velocity command")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.duration is not None:
        cfg["simulation_duration"] = args.duration
    if args.cmd is not None:
        cfg["cmd_init"] = args.cmd
    if args.no_depth_window:
        cfg["show_depth_window"] = False

    deploy_cfg = load_yaml(Path(cfg["deploy_yaml_path"]))
    terrain_mode = str(cfg.get("terrain_mode", "feapvision"))
    extra_geoms = feapvision_terrain_geoms() if terrain_mode == "feapvision" else cfg.get("extra_geoms", [])
    scene_xml_path = make_scene_xml(Path(cfg["robot_xml_path"]), extra_geoms)

    model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = float(cfg["simulation_dt"])

    depth_camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, str(cfg.get("depth_camera_name", "depth_cam"))
    )
    if depth_camera_id < 0:
        raise RuntimeError("depth_cam was not found in the G1 XML")

    policy_joint_ids = np.asarray(deploy_cfg["joint_ids_map"], dtype=np.int32)
    default_joint_pos = np.asarray(deploy_cfg["default_joint_pos"], dtype=np.float32)
    stiffness = np.asarray(deploy_cfg["stiffness"], dtype=np.float32)
    damping = np.asarray(deploy_cfg["damping"], dtype=np.float32)
    action_scale = np.asarray(deploy_cfg["actions"]["JointPositionAction"]["scale"], dtype=np.float32)
    action_offset = np.asarray(deploy_cfg["actions"]["JointPositionAction"]["offset"], dtype=np.float32)

    actuated_joint_count = sum(
        1 for jid in range(model.njnt) if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE
    )
    if actuated_joint_count != len(policy_joint_ids):
        raise RuntimeError(f"Expected 29 actuated joints, got {actuated_joint_count}")
    if len(G1_CAMERA_JOINT_SIGNS) != len(policy_joint_ids):
        raise RuntimeError("G1 joint sign map length does not match the policy joint count")
    qpos_policy_idx, qvel_policy_idx, ctrl_policy_idx, policy_joint_names = mujoco_indices_for_policy_order(
        model, policy_joint_ids
    )

    data.qpos[0:3] = np.asarray(cfg["initial_base_pos"], dtype=np.float32)
    data.qpos[3:7] = np.asarray(cfg["initial_base_quat"], dtype=np.float32)
    data.qpos[qpos_policy_idx] = default_joint_pos * G1_CAMERA_JOINT_SIGNS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    depth_dim = int(cfg["depth_width"]) * int(cfg["depth_height"]) * int(cfg["depth_history"])
    policy = G1DepthPolicy(cfg["actor_onnx_path"], cfg["depth_encoder_onnx_path"], depth_dim)
    obs_builder = G1ObservationBuilder(deploy_cfg, depth_dim)

    renderer = mujoco.Renderer(
        model,
        width=int(cfg["depth_render_width"]),
        height=int(cfg["depth_render_height"]),
    )
    renderer.enable_depth_rendering()

    depth_frame, raw_depth = depth_image(renderer, data, depth_camera_id, cfg)
    depth_buffer = make_depth_history_buffer(depth_frame, int(cfg["depth_history"]))

    cmd_target = np.asarray(cfg["cmd_init"], dtype=np.float32)
    cmd = np.zeros_like(cmd_target)
    action = np.zeros(len(policy_joint_ids), dtype=np.float32)
    target_policy_order = action * action_scale + action_offset

    q_policy = data.qpos[qpos_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
    dq_policy = data.qvel[qvel_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
    obs_builder.reset(
        data.qvel[3:6].astype(np.float32),
        projected_gravity(data.qpos[3:7]),
        cmd,
        q_policy - default_joint_pos,
        dq_policy,
        action,
        depth_history_vector(depth_buffer),
    )

    print("MuJoCo joint order:")
    print(joint_name_order(model))
    print("Policy joint order:")
    print(policy_joint_names)
    print(f"Policy observation dim before encoder: {obs_builder.full_obs().shape[1]}")
    print(f"Depth tail dim: {depth_dim}")

    control_decimation = int(cfg["control_decimation"])
    depth_skip = int(cfg["depth_history_skip_frames"])
    simulation_duration = float(cfg["simulation_duration"])
    warmup_duration = float(cfg.get("warmup_duration", 1.0))
    command_ramp_duration = float(cfg.get("command_ramp_duration", 2.0))
    action_smoothing = float(cfg.get("action_smoothing", 0.2))
    action_clip = float(cfg.get("action_clip", 1.0))
    show_depth_preview = bool(cfg.get("show_depth_window", True)) and not args.headless
    render_mode = "headless" if args.headless else "viewer"
    print(
        f"Running {render_mode} simulation for {simulation_duration:.2f}s "
        f"(depth window: {'on' if show_depth_preview else 'off'})"
    )
    if show_depth_preview:
        show_depth(raw_depth, cfg)

    def step_loop(viewer: Any | None = None) -> None:
        counter = 0
        control_counter = 0
        t0 = time.time()
        nonlocal action, target_policy_order, raw_depth
        while time.time() - t0 < simulation_duration:
            if viewer is not None and not viewer.is_running():
                break
            step_start = time.time()
            elapsed = time.time() - t0

            q_policy = data.qpos[qpos_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
            dq_policy = data.qvel[qvel_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
            tau_policy = pd_control(target_policy_order, q_policy, stiffness, dq_policy, damping)
            tau_mujoco = np.zeros(model.nu, dtype=np.float32)
            tau_mujoco[ctrl_policy_idx] = tau_policy * G1_CAMERA_JOINT_SIGNS
            data.ctrl[:] = tau_mujoco
            mujoco.mj_step(model, data)
            counter += 1

            if counter % control_decimation == 0:
                if control_counter % depth_skip == 0:
                    frame, raw_depth = depth_image(renderer, data, depth_camera_id, cfg)
                    depth_buffer.append(frame)
                    if show_depth_preview:
                        show_depth(raw_depth, cfg)

                if elapsed < warmup_duration:
                    cmd = np.zeros_like(cmd_target)
                    action = np.zeros_like(action)
                    target_policy_order = action_offset.copy()
                    control_counter += 1
                    continue

                if command_ramp_duration > 0:
                    ramp = min((elapsed - warmup_duration) / command_ramp_duration, 1.0)
                else:
                    ramp = 1.0
                cmd = cmd_target * np.clip(ramp, 0.0, 1.0)

                q_policy = data.qpos[qpos_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
                dq_policy = data.qvel[qvel_policy_idx].astype(np.float32) * G1_CAMERA_JOINT_SIGNS
                full_obs = obs_builder.add(
                    data.qvel[3:6].astype(np.float32),
                    projected_gravity(data.qpos[3:7]),
                    cmd,
                    q_policy - default_joint_pos,
                    dq_policy,
                    action,
                    depth_history_vector(depth_buffer),
                )
                raw_action = np.clip(policy.infer(full_obs), -action_clip, action_clip)
                action = (1.0 - action_smoothing) * action + action_smoothing * raw_action
                target_policy_order = action * action_scale + action_offset
                control_counter += 1

            if viewer is not None:
                viewer.sync()
            dt_rem = model.opt.timestep - (time.time() - step_start)
            if dt_rem > 0:
                time.sleep(dt_rem)
        print(f"Done. MuJoCo steps: {counter}, policy steps: {control_counter}")

    if args.headless:
        step_loop(None)
    else:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            step_loop(viewer)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
