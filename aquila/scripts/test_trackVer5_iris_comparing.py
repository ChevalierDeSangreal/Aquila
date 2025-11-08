#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import asyncio
import pickle
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter

import pytz
from datetime import datetime
import sys

# MAVLink imports
from mavsdk import System
from mavsdk.offboard import (AttitudeRate, OffboardError, PositionNedYaw)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.modules.mlp import MLP
# from aquila.envs.target_trackVer5 import TrackEnvVer5  # Disabled for performance
# from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper  # Disabled for performance

"""
Test TrackVer5 policy with Gazebo SITL using MAVLink communication.
Similar to test_trackVer5.py but adapted for onboard/SITL testing.
Uses JAX for inference.
"""


# ==================== Utility Functions ====================
def load_trained_policy(checkpoint_path):
    """Load trained policy parameters"""
    print(f"Loading policy from: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        params = data['params']
        env_config = data.get('env_config', {})
        final_loss = data.get('final_loss', 'Unknown')
        training_epochs = data.get('training_epochs', 'Unknown')
        action_repeat = data.get('action_repeat', 10)
        buffer_size = data.get('action_obs_buffer_size', 10)
        input_dim = data.get('input_dimension', None)
    else:
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
        action_repeat = 10
        buffer_size = 10
        input_dim = None
    
    print("✅ Policy parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    print(f"   Action repeat: {action_repeat}")
    print(f"   Buffer size: {buffer_size}")
    if input_dim:
        print(f"   Input dimension: {input_dim}")
    
    return params, env_config, action_repeat, buffer_size, input_dim


def normalize_observation(obs, obs_min, obs_max):
    """Normalize observation to [-1, 1]"""
    return 2.0 * (obs - obs_min) / (obs_max - obs_min) - 1.0


def denormalize_action(action_normalized, action_low, action_high):
    """Denormalize action from [-1, 1] to actual range"""
    return (action_normalized + 1.0) / 2.0 * (action_high - action_low) + action_low


def normalize_action_for_gazebo(action_network):
    """Convert network action from [-1, 1] to Gazebo's [0, 1] range"""
    return (action_network + 1.0) / 2.0


def compute_hovering_thrust(mass=1.0, gravity=9.81, num_rotors=4):
    """Calculate hovering thrust per rotor"""
    return mass * gravity / num_rotors


def get_time():
    """Get current timestamp in Shanghai timezone"""
    timestamp = time.time()
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    return formatted_time_local


def safe_norm(x, eps=1e-8):
    """Safe norm computation"""
    return jnp.sqrt(jnp.sum(x * x) + eps)


def euler_to_rotation_matrix(euler):
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix (NED frame)"""
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    # Compute rotation matrices
    cy = jnp.cos(yaw)
    sy = jnp.sin(yaw)
    cp = jnp.cos(pitch)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    sr = jnp.sin(roll)
    
    # NED frame rotation matrix
    R = jnp.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ])
    
    return R


def compute_angle_between_body_x_and_target(R, quad_pos, target_pos):
    """Compute angle between body x-axis and direction to target (degrees)"""
    # Body x-axis in world frame (NED: body x points forward/机头方向)
    body_x_world = R @ jnp.array([1.0, 0.0, 0.0])
    
    # Direction to target
    direction_to_target = target_pos - quad_pos
    direction_to_target_normalized = direction_to_target / safe_norm(direction_to_target)
    
    # Compute angle
    cos_angle = jnp.dot(body_x_world, direction_to_target_normalized)
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    angle_rad = jnp.arccos(cos_angle)
    angle_deg = jnp.degrees(angle_rad)
    
    return angle_deg


# ==================== Telemetry Subscribers ====================
latest_state = {
    "odometry": None,
    "attitude": None
}

async def subscribe_telemetry(drone):
    """Subscribe to odometry updates"""
    async for odometry in drone.telemetry.odometry():
        latest_state["odometry"] = odometry

async def subscribe_attitude(drone):
    """Subscribe to attitude updates"""
    async for attitude in drone.telemetry.attitude_euler():
        latest_state["attitude"] = attitude


# ==================== Main Test Function ====================
async def run():
    # ==================== Setup ====================
    print(f"\n{'='*60}")
    print(f"TrackVer5 Gazebo SITL Test")
    print(f"{'='*60}\n")
    
    # JAX setup
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # Random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # ==================== Load Policy ====================
    # policy_file = 'aquila/param_saved/trackVer7_policy_with_Kp_rand.pkl'
    policy_file = 'aquila/param_saved/trackVer6_policy.pkl'
    
    if not os.path.exists(policy_file):
        print(f"❌ Error: Policy file not found: {policy_file}")
        print(f"   Please run train_trackVer5.py first")
        return
    
    params, env_config, action_repeat, buffer_size, input_dim = load_trained_policy(policy_file)
    
    # ==================== Environment Configuration ====================
    # Test configuration (independent of training env_config)
    dt = 0.01  # Time step in seconds
    max_steps = 1000  # Maximum number of steps per episode (doubled for extended testing)
    target_height = 2.0  # Target height in meters
    
    # Infer observation dimension from input_dim and buffer_size
    action_dim = 4  # [thrust, omega_x, omega_y, omega_z]
    if input_dim:
        obs_dim = (input_dim // buffer_size) - action_dim
        print(f"   Computed obs_dim from input_dim: {obs_dim}")
    else:
        # Fallback: use TrackEnvVer5 default observation space (9 dims)
        obs_dim = 9
    
    # Observation space bounds from TrackEnvVer5 (9 dimensions)
    # TrackEnvVer5 obs: [v_body(3), g_body(3), target_pos_body(3)]
    obs_min = jnp.array([
        -20.0, -20.0, -20.0,  # v_body (body-frame velocity)
        -1.0, -1.0, -1.0,     # g_body (body-frame gravity direction)
        -100.0, -100.0, -100.0,  # target_pos_body (body-frame target position, relative)
    ])
    
    obs_max = jnp.array([
        20.0, 20.0, 20.0,     # v_body
        1.0, 1.0, 1.0,        # g_body
        100.0, 100.0, 100.0,  # target_pos_body
    ])
    
    # Action space bounds
    mass = 1.0  # kg (Iris quadrotor)
    gravity = 9.81
    thrust_min = 0.0
    thrust_max = 1.5 * mass * gravity / 4.0  # 3x hovering thrust per rotor
    omega_max = 0.5  # rad/s
    
    action_low = jnp.array([thrust_min * 4, -omega_max, -omega_max, -omega_max])
    action_high = jnp.array([thrust_max * 4, omega_max, omega_max, omega_max])
    
    # Hovering action (normalized)
    # In Gazebo's [0, 1] range, hovering thrust is 0.71
    # Convert to network's [-1, 1] range: (0.71 * 2) - 1 = 0.42
    hovering_thrust_gazebo = 0.71  # Gazebo range [0, 1]
    hovering_thrust_network = 2.0 * hovering_thrust_gazebo - 1.0  # Network range [-1, 1] = 0.42
    # For angular rates, 0 (hovering) in [0, 1] corresponds to 0 in [-1, 1] (since 0 = (0 + 1) / 2 = 0.5 in [0,1] maps to -1 in [-1,1])
    # Actually, 0 angular rate in [0, 1] would be 0.5, which maps to 0 in [-1, 1]
    hovering_omega_network = 0.0  # Zero angular rates in [-1, 1] range
    hovering_action_normalized = jnp.array([hovering_thrust_network, hovering_omega_network, hovering_omega_network, hovering_omega_network])
    
    # ==================== Model Setup ====================
    input_dim = buffer_size * (obs_dim + action_dim)
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    policy_apply = jax.jit(lambda p, x: policy.apply(p, x))
    
    # ==================== Environment Simulation Disabled for Performance ====================
    # Environment simulation has been disabled to achieve 100Hz control rate
    # It was taking ~280ms per step, causing control delays
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Action repeat: {action_repeat} (每个action持续{action_repeat * dt:.3f}s)")
    print(f"  Input dim: {input_dim}")
    print(f"  Time step: {dt}s (100Hz)")
    print(f"  Max steps: {max_steps}")
    print(f"  Target height: {target_height}m")
    print(f"  Hovering thrust (Gazebo [0,1]): {hovering_thrust_gazebo:.3f}")
    print(f"  Hovering thrust (Network [-1,1]): {hovering_thrust_network:.3f}")
    print(f"  Parallel env simulation: Disabled (for performance)")
    print(f"\n  📊 ulog对齐:")
    print(f"     - TensorBoard step 0 对应 ulog命令索引 25")
    print(f"     - 前25条是预热命令（20条offboard + 5条JAX编译）")
    print(f"     - 查看TensorBoard的 'Debug/Global_Command_Index' 曲线")
    print(f"{'='*60}\n")
    
    # ==================== TensorBoard Setup ====================
    run_name = f"trackVer5_sitl__{seed}__{get_time()}"
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    log_path = os.path.join(project_root, "aquila", "test_runs", run_name)
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_path)
    tb_scalars = defaultdict(list)
    timing_records = defaultdict(list)
    step_log_queue = []
    writer.add_text(
        "hyperparameters",
        f"|param|value|\n|-|-|\n"
        f"|obs_dim|{obs_dim}|\n"
        f"|action_dim|{action_dim}|\n"
        f"|buffer_size|{buffer_size}|\n"
        f"|action_repeat|{action_repeat}|\n"
        f"|dt|{dt}|\n"
        f"|max_steps|{max_steps}|\n"
        f"|target_height|{target_height}|\n"
    )
    
    writer.add_text(
        "ulog_alignment",
        "**📊 如何对齐TensorBoard和ulog：**\n\n"
        "- TensorBoard step 0 = ulog中第25条attitude_rate命令\n"
        "- 前25条命令是预热命令：\n"
        "  - 前20条：offboard模式预热（set_attitude_rate，每次0.02s）\n"
        "  - 后5条：JAX编译预热（悬停命令，每次0.01s）\n"
        "- 使用 `Debug/Global_Command_Index` 曲线查看全局命令索引\n"
        "- 每个action持续 action_repeat=10 个steps（0.1秒）\n"
        "- TensorBoard记录频率：100Hz (dt=0.01s)\n"
        "- ulog记录频率：通常250Hz\n\n"
        "**对齐公式：**\n"
        "- `ulog_command_number = Global_Command_Index`\n"
        "- `tensorboard_step = ulog_command_number - 25`\n"
    )
    
    print(f"TensorBoard logs: {log_path}\n")
    
    # ==================== MAVLink Connection ====================
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to drone!")
            break
    
    print("-- Arming")
    await drone.action.arm()
    
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    
    # Get initial position
    async for odometry in drone.telemetry.odometry():
        desired_z = odometry.position_body.z_m
        print(f"Initial position: N={odometry.position_body.x_m:.3f}, "
              f"E={odometry.position_body.y_m:.3f}, D={desired_z:.3f}")
        break
    
    # Start offboard mode
    target_yaw_deg = 0.0
    await drone.offboard.set_position_ned(
        PositionNedYaw(north_m=0, east_m=0, down_m=-target_height, yaw_deg=target_yaw_deg)
    )
    await drone.offboard.start()
    
    print("-- Initializing position")
    for _ in range(100):
        await drone.offboard.set_position_ned(
            PositionNedYaw(north_m=0, east_m=0, down_m=-target_height, yaw_deg=target_yaw_deg)
        )
        await asyncio.sleep(0.05)
    
    print("-- Preheating offboard setpoints...")
    for _ in range(20):
        await drone.offboard.set_attitude_rate(AttitudeRate(0, 0.0, 0.0, 0.71))
        await asyncio.sleep(0.02)
    
    # ==================== Start Telemetry Subscribers ====================
    telemetry_task = asyncio.create_task(subscribe_telemetry(drone))
    attitude_task = asyncio.create_task(subscribe_attitude(drone))
    
    # Wait for telemetry data
    while (latest_state["odometry"] is None) or (latest_state["attitude"] is None):
        await asyncio.sleep(0.1)
    
    print("✅ Telemetry data received\n")
    
    # ==================== Initialize State ====================
    odometry = latest_state["odometry"]
    attitude = latest_state["attitude"]
    
    # Quadrotor state in NED frame
    quad_pos = jnp.array([
        odometry.position_body.x_m,
        odometry.position_body.y_m,
        odometry.position_body.z_m
    ])
    
    quad_vel = jnp.array([
        odometry.velocity_body.x_m_s,
        odometry.velocity_body.y_m_s,
        odometry.velocity_body.z_m_s
    ])
    
    quad_euler = jnp.array([
        attitude.roll_deg * np.pi / 180.0,
        attitude.pitch_deg * np.pi / 180.0,
        attitude.yaw_deg * np.pi / 180.0
    ])
    
    quad_omega = jnp.array([
        odometry.angular_velocity_body.roll_rad_s,
        odometry.angular_velocity_body.pitch_rad_s,
        odometry.angular_velocity_body.yaw_rad_s
    ])
    
    quad_R = euler_to_rotation_matrix(quad_euler)  # 3x3 matrix
    
    # Initialize target position (0.5m in front, at target height)
    # In NED frame: z is negative for positions above ground
    target_pos = jnp.array([
        quad_pos[0] + 1,  # 1m north
        quad_pos[1],        # same east
        -target_height      # at target height (negative z is up)
    ])
    
    target_vel = jnp.array([1, 0.0, 0.0])
    
    initial_distance = float(jnp.linalg.norm(target_pos - quad_pos))
    print(f"Initial quadrotor position: {np.array(quad_pos)}")
    print(f"Initial target position: {np.array(target_pos)}")
    print(f"Initial distance: {initial_distance:.3f}m")
    print(f"Target height (above ground): {target_height}m\n")
    
    # ==================== Initialize Action-Observation Buffer ====================
    # Create initial observation following TrackEnvVer5 format
    # TrackEnvVer5 obs: [v_body(3), g_body(3), target_pos_body(3)]
    
    # Get body-frame velocity
    v_body = quad_vel  # already in body frame from MAVLink
    
    # Get body-frame gravity direction (R_transpose @ g_world)
    g_world = jnp.array([0.0, 0.0, 1.0])  # In NED: gravity points down (positive z)
    R_transpose = jnp.transpose(quad_R)
    g_body = R_transpose @ g_world
    
    # Get body-frame target position
    rel_pos = target_pos - quad_pos
    target_pos_body = R_transpose @ rel_pos
    
    obs = jnp.concatenate([
        v_body,           # body-frame velocity (3)
        g_body,           # body-frame gravity direction (3)
        target_pos_body,  # body-frame target position (3)
    ])

    print("Initial target position in body frame: ", target_pos_body)
    
    # Normalize observation
    obs_normalized = normalize_observation(obs, obs_min, obs_max)
    
    # Initialize buffer with hovering action and current observation
    # Buffer shape should be (1, buffer_size, obs_dim + action_dim) to match training format
    # Hovering action: thrust=0.42 (maps to 0.71 in Gazebo's [0,1]), angular rates=0.0 (zero in [-1,1])
    action_obs_combined = jnp.concatenate([hovering_action_normalized, obs_normalized])
    # Fill entire buffer with the same hovering action + observation (all timesteps start with hover)
    action_obs_buffer = jnp.tile(action_obs_combined[None, None, :], (1, buffer_size, 1))
    
    # Get initial action
    # Reshape to (1, -1) to match training: (num_envs, buffer_size * (obs_dim + action_dim))
    action_obs_buffer_flat = action_obs_buffer.reshape(1, -1)
    initial_action = policy_apply(params, action_obs_buffer_flat)[0]
    jax.block_until_ready(initial_action)
    
    print(f"\n{'='*60}")
    print(f"🎯 INITIAL ACTION (will be used for first few steps):")
    print(f"   thrust={float(initial_action[0]):+.3f}, roll={float(initial_action[1]):+.3f}, "
          f"pitch={float(initial_action[2]):+.3f}, yaw={float(initial_action[3]):+.3f}")
    print(f"{'='*60}\n")
    
    # 立即更新buffer（确保下次推理时输入不同）
    action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=1)
    action_obs_combined_new = jnp.concatenate([initial_action, obs_normalized])
    action_obs_buffer = action_obs_buffer.at[0, -1, :].set(action_obs_combined_new)
    
    current_action = initial_action
    # 初始化为1，表示initial_action已获取，将持续到action_repeat后才获取新action
    action_counter = 1
    
    # Variables to store network input and output for TensorBoard logging
    current_network_input = np.array(action_obs_buffer_flat[0])  # Store initial network input
    current_network_output = np.array(initial_action)  # Store initial network output
    
    warm_angle = compute_angle_between_body_x_and_target(quad_R, quad_pos, target_pos)
    jax.block_until_ready(warm_angle)

    # ==================== JAX Compilation Warmup ====================
    # 执行warmup推理来预编译JAX函数，同时预热MAVLink命令发送和状态计算
    print(f"{'='*60}")
    print(f"Warming up JAX compilation and MAVLink...")
    print(f"{'='*60}\n")
    
    warmup_steps = 10
    warmup_action_counter = action_counter
    warmup_buffer = action_obs_buffer  # JAX数组是immutable的，可以直接赋值
    
    print(f"Starting warmup with action_counter={action_counter}")
    
    for warmup_iter in range(warmup_steps):
        warmup_start = time.perf_counter()
        
        # 模拟主循环的网络推理逻辑（用于JAX编译）
        if warmup_action_counter % action_repeat == 0:
            print(f"  Warmup iter {warmup_iter}: counter={warmup_action_counter}, getting NEW warmup action")
            warmup_buffer_for_input = jnp.roll(warmup_buffer, shift=-1, axis=1)
            empty_action = jnp.zeros(action_dim)
            action_obs_combined_empty = jnp.concatenate([empty_action, obs_normalized])
            warmup_buffer_for_input = warmup_buffer_for_input.at[0, -1, :].set(action_obs_combined_empty)
            
            warmup_buffer_flat = warmup_buffer_for_input.reshape(1, -1)
            warmup_action = policy_apply(params, warmup_buffer_flat)[0]
            jax.block_until_ready(warmup_action)
            print(f"       Warmup action: thrust={float(warmup_action[0]):+.3f}, roll={float(warmup_action[1]):+.3f}")
            
            warmup_buffer = jnp.roll(warmup_buffer, shift=-1, axis=1)
            action_obs_combined_new = jnp.concatenate([warmup_action, obs_normalized])
            warmup_buffer = warmup_buffer.at[0, -1, :].set(action_obs_combined_new)
            
            warmup_action_counter = 1
        else:
            warmup_action_counter += 1
        
        # 预热MAVLink命令发送（使用实际的action转换逻辑）
        warmup_thrust_network = float(hovering_action_normalized[0])
        warmup_thrust_gazebo = normalize_action_for_gazebo(jnp.array([warmup_thrust_network]))[0]
        warmup_thrust_normalized = float(np.clip(warmup_thrust_gazebo, 0.0, 1.0))
        
        warmup_omega = np.array([0.0, 0.0, 0.0])
        warmup_roll_rate_deg = warmup_omega[0] * 180.0 / np.pi
        warmup_pitch_rate_deg = warmup_omega[1] * 180.0 / np.pi
        warmup_yaw_rate_deg = warmup_omega[2] * 180.0 / np.pi
        
        await drone.offboard.set_attitude_rate(
            AttitudeRate(
                roll_deg_s=float(warmup_roll_rate_deg),
                pitch_deg_s=float(warmup_pitch_rate_deg),
                yaw_deg_s=float(warmup_yaw_rate_deg),
                thrust_value=float(warmup_thrust_normalized)
            )
        )
        
        # 预热状态更新和几何计算（模拟主循环的完整state update流程）
        warmup_odometry = latest_state["odometry"]
        warmup_attitude = latest_state["attitude"]
        
        warmup_quad_pos = jnp.array([
            warmup_odometry.position_body.x_m,
            warmup_odometry.position_body.y_m,
            warmup_odometry.position_body.z_m
        ])
        
        warmup_quad_vel = jnp.array([
            warmup_odometry.velocity_body.x_m_s,
            warmup_odometry.velocity_body.y_m_s,
            warmup_odometry.velocity_body.z_m_s
        ])
        
        warmup_quad_euler = jnp.array([
            warmup_attitude.roll_deg * np.pi / 180.0,
            warmup_attitude.pitch_deg * np.pi / 180.0,
            warmup_attitude.yaw_deg * np.pi / 180.0
        ])
        
        warmup_quad_R = euler_to_rotation_matrix(warmup_quad_euler)
        warmup_R_transpose = jnp.transpose(warmup_quad_R)
        
        # 预热几何计算
        warmup_g_world = jnp.array([0.0, 0.0, 1.0])
        warmup_g_body = warmup_R_transpose @ warmup_g_world
        
        warmup_rel_pos = target_pos - warmup_quad_pos
        warmup_target_pos_body = warmup_R_transpose @ warmup_rel_pos
        
        # 预热observation计算
        warmup_obs = jnp.concatenate([
            warmup_quad_vel,
            warmup_g_body,
            warmup_target_pos_body,
        ])
        warmup_obs_normalized = normalize_observation(warmup_obs, obs_min, obs_max)
        
        # 预热metrics计算（包括float转换）
        warmup_distance = float(safe_norm(warmup_rel_pos))
        warmup_height = float(-warmup_quad_pos[2])
        warmup_speed = float(safe_norm(warmup_quad_vel))
        warmup_angle = float(compute_angle_between_body_x_and_target(
            warmup_quad_R, warmup_quad_pos, target_pos
        ))
        
        # 确保所有JAX计算完成
        jax.block_until_ready(warmup_obs_normalized)
        
        warmup_elapsed = time.perf_counter() - warmup_start
        if warmup_iter < 3:
            print(f"  Warmup iter {warmup_iter}: counter={warmup_action_counter}, elapsed={warmup_elapsed*1000:.2f}ms")
        
        await asyncio.sleep(dt)
    
    # ✅ 同步warmup后的状态到主循环（关键！）
    action_counter = warmup_action_counter
    action_obs_buffer = warmup_buffer
    
    print(f"✅ JAX warmup completed ({warmup_steps} iterations)")
    print(f"   Warmup commands sent: {warmup_steps} hovering commands (not control commands)")
    print(f"   Action counter after warmup: {action_counter} (initial_action will be used for {action_repeat - action_counter + 1} more steps)\n")
    
    # 记录预热命令数量，用于对齐ulog
    # 20次 attitude_rate 预热命令 + warmup_steps 个悬停命令
    preheating_commands_sent = 20 + warmup_steps
    
    total_iterations = max_steps
    logged_steps = 0

    print(f"{'='*60}")
    print(f"Starting test episode...")
    print(f"Preheating commands sent before main loop: {preheating_commands_sent}")
    print(f"  - 20 attitude_rate preheating commands")
    print(f"  - {warmup_steps} JAX warmup hovering commands")
    print(f"Main loop will record ALL steps from step 0 (aligned with ulog)")
    print(f"{'='*60}\n")
    
    # ==================== Main Test Loop ====================
    for iteration in range(total_iterations):
        # 所有iteration都记录到TensorBoard
        log_enabled = True
        step_idx = iteration
        # 全局命令计数器（用于对齐ulog）
        # global_command_idx = 预热命令数 + 当前iteration
        global_command_idx = preheating_commands_sent + iteration

        step_start_time = time.perf_counter()
        
        # ==================== Get New Action (if needed) ====================
        # 打印前30个steps的counter状态（详细调试）
        if step_idx < 15:
            will_get_new = (action_counter % action_repeat == 0)
            # 显示当前使用的action (简化)
            current_action_roll_cmd = float(current_action[1])
            # 获取当前的姿态角
            current_attitude = latest_state["attitude"]
            current_roll_deg = float(current_attitude.roll_deg) if current_attitude else 0.0
            print(f"  [Step {step_idx:2d}, ulog#{global_command_idx:3d}] counter={action_counter:2d}, will_get_new={will_get_new}, "
                  f"roll_cmd={current_action_roll_cmd:+.3f}, roll_attitude={current_roll_deg:+.2f}°")
        
        section_start = time.perf_counter()
        timing_records['00_debug_print_ms'].append((iteration, (section_start - step_start_time) * 1000.0))
        
        if action_counter % action_repeat == 0:
            # Create temporary buffer for getting new action
            # Buffer shape is (1, buffer_size, obs_dim + action_dim), roll on axis=1
            action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=1)
            # Use zero action (0.0 in [-1, 1] range) as placeholder for current timestep
            # This will be replaced by the network's output action
            empty_action = jnp.zeros(action_dim)  # [0, 0, 0, 0] in [-1, 1] range
            action_obs_combined_empty = jnp.concatenate([empty_action, obs_normalized])
            # Set the last element in the buffer (along axis=1)
            action_obs_buffer_for_input = action_obs_buffer_for_input.at[0, -1, :].set(action_obs_combined_empty)
            
            # 打印用于推理的observation（前15步，调试）
            if step_idx < 15:
                print(f"       Using obs_normalized[0:3] (v_body): [{float(obs_normalized[0]):+.3f}, {float(obs_normalized[1]):+.3f}, {float(obs_normalized[2]):+.3f}]")
                print(f"       Using obs_normalized[6:9] (target_pos_body): [{float(obs_normalized[6]):+.3f}, {float(obs_normalized[7]):+.3f}, {float(obs_normalized[8]):+.3f}]")
            
            # Get new action from network
            # Reshape to (1, -1) to match training format
            action_obs_buffer_flat = action_obs_buffer_for_input.reshape(1, -1)
            new_action = policy_apply(params, action_obs_buffer_flat)[0]
            
            # 打印action变化（前15个steps，帮助调试）
            if step_idx < 15:
                print(f"       >>> NEW ACTION: thrust={float(new_action[0]):+.3f}, roll={float(new_action[1]):+.3f}, "
                      f"pitch={float(new_action[2]):+.3f}, yaw={float(new_action[3]):+.3f}")
            
            current_action = new_action
            
            # Store network input and output for TensorBoard logging
            current_network_input = np.array(action_obs_buffer_flat[0])  # Convert to numpy for logging
            current_network_output = np.array(current_action)  # Convert to numpy for logging
            
            # Update buffer with new action for next iteration
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=1)
            action_obs_combined_new = jnp.concatenate([current_action, obs_normalized])
            action_obs_buffer = action_obs_buffer.at[0, -1, :].set(action_obs_combined_new)
            
            action_counter = 1
        else:
            action_counter += 1

        section_after_policy = time.perf_counter()
        timing_records['01_policy_selection_ms'].append((iteration, (section_after_policy - section_start) * 1000.0))
            
        # ==================== Convert Action to MAVLink Command ====================
        # Network outputs actions in [-1, 1] range
        # Convert thrust from [-1, 1] to Gazebo's [0, 1] range
        thrust_network = float(current_action[0])
        thrust_gazebo = normalize_action_for_gazebo(jnp.array([thrust_network]))[0]
        thrust_normalized = float(np.clip(thrust_gazebo, 0.0, 1.0))
        
        # Angular rates: denormalize from [-1, 1] to actual rad/s range
        omega_network = np.array([current_action[1], current_action[2], current_action[3]])
        omega_cmd = denormalize_action(omega_network, -omega_max, omega_max)
        
        # Angular rates in rad/s (for logging)
        roll_rate_rad_s = float(omega_cmd[0])
        pitch_rate_rad_s = float(omega_cmd[1])
        yaw_rate_rad_s = float(omega_cmd[2])
        
        # Angular rates in deg/s (convert to deg/s for MAVLink)
        roll_rate_deg = omega_cmd[0] * 180.0 / np.pi
        pitch_rate_deg = omega_cmd[1] * 180.0 / np.pi
        yaw_rate_deg = omega_cmd[2] * 180.0 / np.pi
        
        # Send command
        await drone.offboard.set_attitude_rate(
            AttitudeRate(
                roll_deg_s=float(roll_rate_deg),
                pitch_deg_s=float(pitch_rate_deg),
                yaw_deg_s=float(yaw_rate_deg),
                thrust_value=float(thrust_normalized)
            )
        )

        section_after_mavlink = time.perf_counter()
        timing_records['02_mavlink_send_ms'].append((iteration, (section_after_mavlink - section_after_policy) * 1000.0))
        
        # ==================== Update State from Telemetry ====================
        odometry = latest_state["odometry"]
        attitude = latest_state["attitude"]
        
        quad_pos = jnp.array([
            odometry.position_body.x_m,
            odometry.position_body.y_m,
            odometry.position_body.z_m
        ])
        
        quad_vel = jnp.array([
            odometry.velocity_body.x_m_s,
            odometry.velocity_body.y_m_s,
            odometry.velocity_body.z_m_s
        ])
        
        quad_euler = jnp.array([
            attitude.roll_deg * np.pi / 180.0,
            attitude.pitch_deg * np.pi / 180.0,
            attitude.yaw_deg * np.pi / 180.0
        ])
        
        quad_omega = jnp.array([
            odometry.angular_velocity_body.roll_rad_s,
            odometry.angular_velocity_body.pitch_rad_s,
            odometry.angular_velocity_body.yaw_rad_s
        ])
        
        quad_R = euler_to_rotation_matrix(quad_euler)
        
        # Update target position based on its velocity
        target_pos = target_pos + target_vel * dt
        
        # ==================== Compute Observation ====================
        # Create observation following TrackEnvVer5 format
        rel_pos = target_pos - quad_pos
        
        # Get body-frame velocity
        v_body = quad_vel  # already in body frame from MAVLink
        
        # Get body-frame gravity direction
        g_world = jnp.array([0.0, 0.0, 1.0])
        R_transpose = jnp.transpose(quad_R)
        g_body = R_transpose @ g_world
        
        # Get body-frame target position
        target_pos_body = R_transpose @ rel_pos
        
        obs = jnp.concatenate([
            v_body,           # body-frame velocity (3)
            g_body,           # body-frame gravity direction (3)
            target_pos_body,  # body-frame target position (3)
        ])
        obs_normalized = normalize_observation(obs, obs_min, obs_max)
        
        # ==================== Compute Metrics ====================
        distance = float(safe_norm(rel_pos))
        height = float(-quad_pos[2])  # NED: negative z is up
        speed = float(safe_norm(quad_vel))
        
        angle_to_target = float(compute_angle_between_body_x_and_target(
            quad_R, quad_pos, target_pos
        ))
            
        # Simple reward (similar to TrackEnvVer5)
        distance_penalty = -distance
        height_error = abs(height - target_height)
        height_penalty = -height_error
        reward = distance_penalty + height_penalty

        section_after_state = time.perf_counter()
        timing_records['03_state_update_ms'].append((iteration, (section_after_state - section_after_mavlink) * 1000.0))
        
        # ==================== Print Progress ====================
        if step_idx % 50 == 0:
            print(f"Step {step_idx:4d} (ulog#{global_command_idx:3d}) | "
                  f"Dist: {distance:6.3f}m | "
                  f"Height: {height:5.2f}m | "
                  f"Speed: {speed:5.2f}m/s | "
                  f"Angle: {angle_to_target:5.1f}° | "
                  f"Reward: {reward:7.3f}")
        
        # ==================== Timing ====================
        step_end_time = time.perf_counter()
        elapsed_time = step_end_time - step_start_time
        sleep_time = dt - elapsed_time
        
        section_after_logging = time.perf_counter()
        timing_records['04_logging_prepare_ms'].append((iteration, (section_after_logging - section_after_state) * 1000.0))

        if step_idx < 15:
            print(
                f"[TIMING] step={step_idx} "
                f"debug={timing_records['00_debug_print_ms'][-1][1]:6.2f} ms | "
                f"policy={timing_records['01_policy_selection_ms'][-1][1]:6.2f} ms | "
                f"mavlink={timing_records['02_mavlink_send_ms'][-1][1]:6.2f} ms | "
                f"state={timing_records['03_state_update_ms'][-1][1]:6.2f} ms | "
                f"log_prep={timing_records['04_logging_prepare_ms'][-1][1]:6.2f} ms | "
                f"total={elapsed_time * 1000.0:6.2f} ms"
            )

        # 记录所有step到TensorBoard
        step_log_queue.append({
            'step_idx': step_idx,
            'global_command_idx': global_command_idx,
            'distance': distance,
            'height': height,
            'speed': speed,
            'angle_to_target': angle_to_target,
            'reward': reward,
            'quad_pos': np.asarray(quad_pos),
            'quad_vel': np.asarray(quad_vel),
            'attitude_deg': (
                float(attitude.roll_deg),
                float(attitude.pitch_deg),
                float(attitude.yaw_deg),
            ),
            'quad_omega': np.asarray(quad_omega),
            'thrust_normalized': thrust_normalized,
            'roll_rate_rad_s': roll_rate_rad_s,
            'pitch_rate_rad_s': pitch_rate_rad_s,
            'yaw_rate_rad_s': yaw_rate_rad_s,
            'network_output': np.asarray(current_network_output)
            if current_network_output is not None
            else None,
            'obs_normalized': np.asarray(obs_normalized),
            'target_pos': np.asarray(target_pos),
            'loop_duration_ms': elapsed_time * 1000.0,
            'loop_sleep_ms': max(sleep_time, 0.0) * 1000.0,
        })

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # 跳过step 0的警告（JAX warmup后可能还有少量开销）
        if step_idx > 0 and elapsed_time > dt * 1.2:
            print(f"[WARN] Control loop lag at step {step_idx}: {elapsed_time * 1000.0:.2f} ms (target {dt * 1000.0:.2f} ms)")
        
        # ==================== Check Termination ====================
        if distance > 100.0:
            print(f"\n❌ Episode terminated: distance too large ({distance:.1f}m)")
            break
        if height < 0.1 or height > 20.0:
            print(f"\n❌ Episode terminated: height out of bounds ({height:.1f}m)")
            break
    
    logged_steps = len(step_log_queue)
    last_log = step_log_queue[-1] if logged_steps > 0 else None

    for log in step_log_queue:
        step_idx = log['step_idx']
        global_cmd_idx = log['global_command_idx']
        
        # 记录全局命令索引以对齐ulog
        tb_scalars['Debug/Global_Command_Index'].append((step_idx, global_cmd_idx))

        tb_scalars['Metrics/Distance'].append((step_idx, log['distance']))
        tb_scalars['Metrics/Height'].append((step_idx, log['height']))
        tb_scalars['Metrics/Speed'].append((step_idx, log['speed']))
        tb_scalars['Metrics/Angle_to_Target'].append((step_idx, log['angle_to_target']))
        tb_scalars['Metrics/Reward'].append((step_idx, log['reward']))

        quad_pos = log['quad_pos']
        quad_vel = log['quad_vel']
        quad_omega = log['quad_omega']

        tb_scalars['Position/North'].append((step_idx, float(quad_pos[0])))
        tb_scalars['Position/East'].append((step_idx, float(quad_pos[1])))
        tb_scalars['Position/Down'].append((step_idx, float(quad_pos[2])))

        tb_scalars['Velocity/North'].append((step_idx, float(quad_vel[0])))
        tb_scalars['Velocity/East'].append((step_idx, float(quad_vel[1])))
        tb_scalars['Velocity/Down'].append((step_idx, float(quad_vel[2])))

        roll_deg, pitch_deg, yaw_deg = log['attitude_deg']
        tb_scalars['Attitude/Roll_deg'].append((step_idx, roll_deg))
        tb_scalars['Attitude/Pitch_deg'].append((step_idx, pitch_deg))
        tb_scalars['Attitude/Yaw_deg'].append((step_idx, yaw_deg))

        tb_scalars['Angular_Velocity/Yaw_rad_s'].append((step_idx, float(quad_omega[2])))

        tb_scalars['Action/Thrust_normalized'].append((step_idx, log['thrust_normalized']))
        tb_scalars['Action/Roll_rate_rad_s'].append((step_idx, log['roll_rate_rad_s']))
        tb_scalars['Action/Pitch_rate_rad_s'].append((step_idx, log['pitch_rate_rad_s']))
        tb_scalars['Action/Yaw_rate_rad_s'].append((step_idx, log['yaw_rate_rad_s']))

        network_output = log['network_output']
        if network_output is not None:
            tb_scalars['Network_Output/Thrust_raw'].append((step_idx, float(network_output[0])))
            tb_scalars['Network_Output/Roll_rate_raw'].append((step_idx, float(network_output[1])))
            tb_scalars['Network_Output/Pitch_rate_raw'].append((step_idx, float(network_output[2])))
            tb_scalars['Network_Output/Yaw_rate_raw'].append((step_idx, float(network_output[3])))

        obs_host = log['obs_normalized']
        if obs_host.shape[0] >= 3:
            tb_scalars['Network_Input/Current_Obs_Vx'].append((step_idx, float(obs_host[0])))
            tb_scalars['Network_Input/Current_Obs_Vy'].append((step_idx, float(obs_host[1])))
            tb_scalars['Network_Input/Current_Obs_Vz'].append((step_idx, float(obs_host[2])))
        if obs_host.shape[0] >= 6:
            tb_scalars['Network_Input/Current_Obs_Gx'].append((step_idx, float(obs_host[3])))
            tb_scalars['Network_Input/Current_Obs_Gy'].append((step_idx, float(obs_host[4])))
            tb_scalars['Network_Input/Current_Obs_Gz'].append((step_idx, float(obs_host[5])))
        if obs_host.shape[0] >= 9:
            tb_scalars['Network_Input/Current_Obs_TargetX'].append((step_idx, float(obs_host[6])))
            tb_scalars['Network_Input/Current_Obs_TargetY'].append((step_idx, float(obs_host[7])))
            tb_scalars['Network_Input/Current_Obs_TargetZ'].append((step_idx, float(obs_host[8])))
        for i in range(9, obs_host.shape[0]):
            tb_scalars[f'Network_Input/Current_Obs_Dim{i}'].append((step_idx, float(obs_host[i])))

        target_pos = log['target_pos']
        tb_scalars['Target/North'].append((step_idx, float(target_pos[0])))
        tb_scalars['Target/East'].append((step_idx, float(target_pos[1])))
        tb_scalars['Target/Down'].append((step_idx, float(target_pos[2])))

        tb_scalars['Loop/Step_Duration_ms'].append((step_idx, log['loop_duration_ms']))
        tb_scalars['Loop/Sleep_Time_ms'].append((step_idx, log['loop_sleep_ms']))
        
        # Angular velocity for all axes
        tb_scalars['Angular_Velocity/Roll_rad_s'].append((step_idx, float(quad_omega[0])))
        tb_scalars['Angular_Velocity/Pitch_rad_s'].append((step_idx, float(quad_omega[1])))
    # ==================== Transition to Position Hold ====================
    latest_attitude = latest_state.get("attitude")
    hold_yaw_deg = float(latest_attitude.yaw_deg) if latest_attitude is not None else 0.0

    latest_odometry = latest_state.get("odometry")
    if latest_odometry is not None:
        hold_north = float(latest_odometry.position_body.x_m)
        hold_east = float(latest_odometry.position_body.y_m)
        hold_down = float(latest_odometry.position_body.z_m)
    else:
        hold_north = 0.0
        hold_east = 0.0
        hold_down = -target_height

    print("-- Holding current position")
    for _ in range(50):
        await drone.offboard.set_position_ned(
            PositionNedYaw(
                north_m=hold_north,
                east_m=hold_east,
                down_m=hold_down,
                yaw_deg=hold_yaw_deg,
            )
        )
        await asyncio.sleep(0.05)

    # ==================== Cleanup ====================
    print(f"\n{'='*60}")
    print(f"Test episode completed!")
    print(f"Total logged steps: {logged_steps}")
    if logged_steps > 0:
        print(f"Final distance: {last_log['distance']:.3f}m")
        print(f"Final height: {last_log['height']:.3f}m")
    print(f"{'='*60}\n")
    
    telemetry_task.cancel()
    attitude_task.cancel()
    
    for tag, entries in tb_scalars.items():
        for step_idx, value in entries:
            writer.add_scalar(tag, value, step_idx)

    writer.close()
    
    print(f"✅ TensorBoard logs saved to: {log_path}")
    print(f"   View with: tensorboard --logdir={log_path}\n")


if __name__ == "__main__":
    asyncio.run(run())
