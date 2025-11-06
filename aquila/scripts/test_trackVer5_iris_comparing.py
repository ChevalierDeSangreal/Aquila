#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import asyncio
import pickle

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
from aquila.envs.target_trackVer5 import TrackEnvVer5
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper

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
    policy_file = 'aquila/param_saved/trackVer5_policy_iris.pkl'
    
    if not os.path.exists(policy_file):
        print(f"❌ Error: Policy file not found: {policy_file}")
        print(f"   Please run train_trackVer5.py first")
        return
    
    params, env_config, action_repeat, buffer_size, input_dim = load_trained_policy(policy_file)
    
    # ==================== Environment Configuration ====================
    # Test configuration (independent of training env_config)
    dt = 0.01  # Time step in seconds
    max_steps = 1000  # Maximum number of steps per episode
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
    
    # ==================== Parallel Environment Simulation Setup ====================
    # Setup TrackEnvVer5 environment for comparison (same as test_trackVer5.py)
    env_sim = TrackEnvVer5(
        max_steps_in_episode=1000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        action_penalty_weight=0.5,
        # Observation dynamics time constants
        obs_tau_pos=0.3,
        obs_tau_vel=0.2,
        obs_tau_R=0.02,
        # Tracking specific parameters
        target_height=2.0,
        target_init_distance_min=1,
        target_init_distance_max=1,
        target_speed_max=1.0,
        reset_distance=100.0,
        max_speed=20.0,
        # Parameter randomization (quadrotor)
        thrust_to_weight_min=1.4,
        thrust_to_weight_max=1.5,
    )
    
    # Apply same wrappers as training
    env_sim = MinMaxObservationWrapper(env_sim)
    env_sim = NormalizeActionWrapper(env_sim)
    
    # Initialize environment simulation with same random seed
    # Note: Initial state may differ from SITL due to different reset mechanisms,
    # but we use the same action sequence to compare dynamics differences
    env_key = jax.random.key(seed)
    env_key, subkey = jax.random.split(env_key)
    env_state, env_obs = env_sim.reset(subkey)
    
    print(f"\n{'='*60}")
    print(f"Environment Simulation Initial State:")
    print(f"  Quad position: {np.array(env_state.quadrotor_state.p)}")
    print(f"  Target position: {np.array(env_state.target_pos)}")
    print(f"  Initial distance: {float(safe_norm(env_state.quadrotor_state.p - env_state.target_pos)):.3f}m")
    print(f"{'='*60}\n")
    
    # Initialize environment simulation buffer (same as SITL buffer)
    env_action_obs_combined = jnp.concatenate([hovering_action_normalized, env_obs])
    env_action_obs_buffer = jnp.tile(env_action_obs_combined[None, None, :], (1, buffer_size, 1))
    
    # Get initial action from environment simulation
    env_action_obs_buffer_flat = env_action_obs_buffer.reshape(1, -1)
    env_initial_action = policy.apply(params, env_action_obs_buffer_flat)[0]
    env_current_action = env_initial_action
    env_action_counter = 0
    
    # Data storage for environment simulation comparison
    env_sim_data = {
        'time': [],
        'quad_pos': [],
        'quad_vel': [],
        'quad_R': [],
        'quad_omega': [],
        'target_pos': [],
        'target_vel': [],
        'action': [],
        'reward': [],
        'distance': [],
        'height': [],
        'angle_body_x_target': [],
    }
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Action repeat: {action_repeat}")
    print(f"  Input dim: {input_dim}")
    print(f"  Time step: {dt}s")
    print(f"  Max steps: {max_steps}")
    print(f"  Target height: {target_height}m")
    print(f"  Hovering thrust (Gazebo [0,1]): {hovering_thrust_gazebo:.3f}")
    print(f"  Hovering thrust (Network [-1,1]): {hovering_thrust_network:.3f}")
    print(f"  Parallel env simulation: Enabled")
    print(f"{'='*60}\n")
    
    # ==================== TensorBoard Setup ====================
    run_name = f"trackVer5_sitl__{seed}__{get_time()}"
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    log_path = os.path.join(project_root, "aquila", "test_runs", run_name)
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_path)
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
    
    target_vel = jnp.zeros(3)  # Target is stationary
    
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
    initial_action = policy.apply(params, action_obs_buffer_flat)[0]
    
    current_action = initial_action
    action_counter = 0
    
    # Variables to store network input and output for TensorBoard logging
    current_network_input = np.array(action_obs_buffer_flat[0])  # Store initial network input
    current_network_output = np.array(initial_action)  # Store initial network output
    
    print(f"{'='*60}")
    print(f"Starting test episode...")
    print(f"{'='*60}\n")
    
    # ==================== Main Test Loop ====================
    for step in range(max_steps):
        step_start_time = time.perf_counter()
        
        # ==================== Get New Action (if needed) ====================
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
            
            # Get new action from network
            # Reshape to (1, -1) to match training format
            action_obs_buffer_flat = action_obs_buffer_for_input.reshape(1, -1)
            current_action = policy.apply(params, action_obs_buffer_flat)[0]
            
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
        
        # ==================== Parallel Environment Simulation Step ====================
        # Use the SAME action as SITL for environment simulation (for fair comparison)
        # This ensures we're comparing the same policy actions applied to different dynamics
        env_action_to_apply = current_action  # Use the same action as SITL
        
        # Convert environment simulation action to rad/s for logging
        env_omega_network = np.array([env_action_to_apply[1], env_action_to_apply[2], env_action_to_apply[3]])
        env_omega_cmd = denormalize_action(env_omega_network, -omega_max, omega_max)
        env_roll_rate_rad_s = float(env_omega_cmd[0])
        env_pitch_rate_rad_s = float(env_omega_cmd[1])
        env_yaw_rate_rad_s = float(env_omega_cmd[2])
        
        # Step environment simulation with the same action used for SITL
        env_key, subkey = jax.random.split(env_key)
        env_transition = env_sim.step(env_state, env_action_to_apply, subkey)
        env_state, env_obs, env_reward, env_terminated, env_truncated, env_info = env_transition
        
        # Update environment simulation buffer (same logic as SITL)
        # Buffer should be updated with the action that was applied and the new observation
        if env_action_counter % action_repeat == 0:
            # Update buffer with new action and observation
            env_action_obs_buffer = jnp.roll(env_action_obs_buffer, shift=-1, axis=1)
            env_action_obs_combined_new = jnp.concatenate([env_action_to_apply, env_obs])
            env_action_obs_buffer = env_action_obs_buffer.at[0, -1, :].set(env_action_obs_combined_new)
            
            env_action_counter = 1
        else:
            # Update only observation part (action remains the same)
            env_action_obs_buffer = jnp.roll(env_action_obs_buffer, shift=-1, axis=1)
            env_action_obs_combined_new = jnp.concatenate([env_action_to_apply, env_obs])
            env_action_obs_buffer = env_action_obs_buffer.at[0, -1, :].set(env_action_obs_combined_new)
            env_action_counter += 1
        
        # Record environment simulation data
        env_sim_data['time'].append(float(env_state.time))
        env_sim_data['quad_pos'].append(np.array(env_state.quadrotor_state.p))
        env_sim_data['quad_vel'].append(np.array(env_state.quadrotor_state.v))
        env_sim_data['quad_R'].append(np.array(env_state.quadrotor_state.R))
        env_sim_data['quad_omega'].append(np.array(env_state.quadrotor_state.omega))
        env_sim_data['target_pos'].append(np.array(env_state.target_pos))
        env_sim_data['target_vel'].append(np.array(env_state.target_vel))
        env_sim_data['action'].append(np.array(env_action_to_apply))
        env_sim_data['reward'].append(float(env_reward))
        
        # Compute metrics for environment simulation
        env_distance = float(safe_norm(env_state.quadrotor_state.p - env_state.target_pos))
        env_sim_data['distance'].append(env_distance)
        env_height = float(-env_state.quadrotor_state.p[2])  # NED: negative z is up
        env_sim_data['height'].append(env_height)
        env_angle = float(compute_angle_between_body_x_and_target(
            env_state.quadrotor_state.R,
            env_state.quadrotor_state.p,
            env_state.target_pos
        ))
        env_sim_data['angle_body_x_target'].append(env_angle)
            
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
        
        # Update target position (stationary for now)
        # You can add target motion here if needed
        
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
        
        # ==================== Log to TensorBoard ====================
        writer.add_scalar('Metrics/Distance', distance, step)
        writer.add_scalar('Metrics/Height', height, step)
        writer.add_scalar('Metrics/Speed', speed, step)
        writer.add_scalar('Metrics/Angle_to_Target', angle_to_target, step)
        writer.add_scalar('Metrics/Reward', reward, step)
        
        writer.add_scalar('Position/North', float(quad_pos[0]), step)
        writer.add_scalar('Position/East', float(quad_pos[1]), step)
        writer.add_scalar('Position/Down', float(quad_pos[2]), step)
        
        writer.add_scalar('Velocity/North', float(quad_vel[0]), step)
        writer.add_scalar('Velocity/East', float(quad_vel[1]), step)
        writer.add_scalar('Velocity/Down', float(quad_vel[2]), step)
        
        writer.add_scalar('Attitude/Roll_deg', attitude.roll_deg, step)
        writer.add_scalar('Attitude/Pitch_deg', attitude.pitch_deg, step)
        writer.add_scalar('Attitude/Yaw_deg', attitude.yaw_deg, step)
        
        # Roll and pitch angular velocities will be logged together with EnvSim for comparison
        # Yaw angular velocity remains separate
        writer.add_scalar('Angular_Velocity/Yaw_rad_s', float(quad_omega[2]), step)
        
        writer.add_scalar('Action/Thrust_normalized', thrust_normalized, step)
        writer.add_scalar('Action/Roll_rate_rad_s', roll_rate_rad_s, step)
        writer.add_scalar('Action/Pitch_rate_rad_s', pitch_rate_rad_s, step)
        writer.add_scalar('Action/Yaw_rate_rad_s', yaw_rate_rad_s, step)
        
        # Log network raw output (in [-1, 1] range)
        if current_network_output is not None:
            writer.add_scalar('Network_Output/Thrust_raw', float(current_network_output[0]), step)
            writer.add_scalar('Network_Output/Roll_rate_raw', float(current_network_output[1]), step)
            writer.add_scalar('Network_Output/Pitch_rate_raw', float(current_network_output[2]), step)
            writer.add_scalar('Network_Output/Yaw_rate_raw', float(current_network_output[3]), step)
        
        # Log network input (only current frame observation)
        # Record only the current frame's observation (the latest observation in the buffer)
        # The current frame observation is obs_normalized, which is computed from current state
        if obs_normalized is not None:
            # Log current frame observation components
            # For TrackEnvVer5: obs_normalized contains [v_body(3), g_body(3), target_pos_body(3)] = 9 dims
            # But we'll log all available dimensions dynamically
            if obs_dim >= 3:
                writer.add_scalar('Network_Input/Current_Obs_Vx', float(obs_normalized[0]), step)
                writer.add_scalar('Network_Input/Current_Obs_Vy', float(obs_normalized[1]), step)
                writer.add_scalar('Network_Input/Current_Obs_Vz', float(obs_normalized[2]), step)
            if obs_dim >= 6:
                writer.add_scalar('Network_Input/Current_Obs_Gx', float(obs_normalized[3]), step)
                writer.add_scalar('Network_Input/Current_Obs_Gy', float(obs_normalized[4]), step)
                writer.add_scalar('Network_Input/Current_Obs_Gz', float(obs_normalized[5]), step)
            if obs_dim >= 9:
                writer.add_scalar('Network_Input/Current_Obs_TargetX', float(obs_normalized[6]), step)
                writer.add_scalar('Network_Input/Current_Obs_TargetY', float(obs_normalized[7]), step)
                writer.add_scalar('Network_Input/Current_Obs_TargetZ', float(obs_normalized[8]), step)
            # Log any additional observation dimensions if obs_dim > 9
            for i in range(9, obs_dim):
                writer.add_scalar(f'Network_Input/Current_Obs_Dim{i}', float(obs_normalized[i]), step)
        
        writer.add_scalar('Target/North', float(target_pos[0]), step)
        writer.add_scalar('Target/East', float(target_pos[1]), step)
        writer.add_scalar('Target/Down', float(target_pos[2]), step)
        
        # ==================== Log Environment Simulation Comparison ====================
        # Log environment simulation metrics for comparison
        writer.add_scalar('EnvSim_Metrics/Distance', env_distance, step)
        writer.add_scalar('EnvSim_Metrics/Height', env_height, step)
        writer.add_scalar('EnvSim_Metrics/Angle_to_Target', env_angle, step)
        writer.add_scalar('EnvSim_Metrics/Reward', float(env_reward), step)
        
        writer.add_scalar('EnvSim_Position/North', float(env_state.quadrotor_state.p[0]), step)
        writer.add_scalar('EnvSim_Position/East', float(env_state.quadrotor_state.p[1]), step)
        writer.add_scalar('EnvSim_Position/Down', float(env_state.quadrotor_state.p[2]), step)
        
        writer.add_scalar('EnvSim_Velocity/North', float(env_state.quadrotor_state.v[0]), step)
        writer.add_scalar('EnvSim_Velocity/East', float(env_state.quadrotor_state.v[1]), step)
        writer.add_scalar('EnvSim_Velocity/Down', float(env_state.quadrotor_state.v[2]), step)
        
        # Compute Euler angles from rotation matrix for environment simulation
        env_R = env_state.quadrotor_state.R
        env_roll = float(jnp.arctan2(env_R[2, 1], env_R[2, 2])) * 180.0 / np.pi
        env_pitch = float(jnp.arcsin(-env_R[2, 0])) * 180.0 / np.pi
        env_yaw = float(jnp.arctan2(env_R[1, 0], env_R[0, 0])) * 180.0 / np.pi
        
        writer.add_scalar('EnvSim_Attitude/Roll_deg', env_roll, step)
        writer.add_scalar('EnvSim_Attitude/Pitch_deg', env_pitch, step)
        writer.add_scalar('EnvSim_Attitude/Yaw_deg', env_yaw, step)
        
        # Log roll and pitch angular velocities together with SITL for comparison
        writer.add_scalars('Angular_Velocity/Roll_rad_s', {
            'SITL': float(quad_omega[0]),
            'EnvSim': float(env_state.quadrotor_state.omega[0])
        }, step)
        writer.add_scalars('Angular_Velocity/Pitch_rad_s', {
            'SITL': float(quad_omega[1]),
            'EnvSim': float(env_state.quadrotor_state.omega[1])
        }, step)
        writer.add_scalar('EnvSim_Angular_Velocity/Yaw_rad_s', float(env_state.quadrotor_state.omega[2]), step)
        
        # Log environment simulation actions (in rad/s)
        # Convert thrust from normalized [-1, 1] to [0, 1] for comparison
        env_thrust_normalized = normalize_action_for_gazebo(jnp.array([env_action_to_apply[0]]))[0]
        writer.add_scalar('EnvSim_Action/Thrust_normalized', float(env_thrust_normalized), step)
        writer.add_scalar('EnvSim_Action/Roll_rate_rad_s', env_roll_rate_rad_s, step)
        writer.add_scalar('EnvSim_Action/Pitch_rate_rad_s', env_pitch_rate_rad_s, step)
        writer.add_scalar('EnvSim_Action/Yaw_rate_rad_s', env_yaw_rate_rad_s, step)
        
        # Log comparison differences
        writer.add_scalar('Comparison/Distance_Diff', distance - env_distance, step)
        writer.add_scalar('Comparison/Height_Diff', height - env_height, step)
        writer.add_scalar('Comparison/Angle_Diff', angle_to_target - env_angle, step)
        writer.add_scalar('Comparison/Roll_rate_Diff_rad_s', roll_rate_rad_s - env_roll_rate_rad_s, step)
        writer.add_scalar('Comparison/Pitch_rate_Diff_rad_s', pitch_rate_rad_s - env_pitch_rate_rad_s, step)
        writer.add_scalar('Comparison/Yaw_rate_Diff_rad_s', yaw_rate_rad_s - env_yaw_rate_rad_s, step)
        
        # ==================== Print Progress ====================
        if step % 50 == 0:
            print(f"Step {step:4d} | "
                  f"Dist: {distance:6.3f}m (EnvSim: {env_distance:6.3f}m) | "
                  f"Height: {height:5.2f}m (EnvSim: {env_height:5.2f}m) | "
                  f"Speed: {speed:5.2f}m/s | "
                  f"Angle: {angle_to_target:5.1f}° (EnvSim: {env_angle:5.1f}°) | "
                  f"Reward: {reward:7.3f} (EnvSim: {float(env_reward):7.3f})")
        
        # ==================== Timing ====================
        step_end_time = time.perf_counter()
        elapsed_time = step_end_time - step_start_time
        sleep_time = dt - elapsed_time
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        # ==================== Check Termination ====================
        if distance > 100.0:
            print(f"\n❌ Episode terminated: distance too large ({distance:.1f}m)")
            break
        if height < 0.1 or height > 20.0:
            print(f"\n❌ Episode terminated: height out of bounds ({height:.1f}m)")
            break
    
    # ==================== Cleanup ====================
    print(f"\n{'='*60}")
    print(f"Test episode completed!")
    print(f"Total steps: {step + 1}")
    print(f"Final distance (SITL): {distance:.3f}m")
    print(f"Final height (SITL): {height:.3f}m")
    print(f"Final distance (EnvSim): {env_distance:.3f}m")
    print(f"Final height (EnvSim): {env_height:.3f}m")
    print(f"{'='*60}\n")
    
    telemetry_task.cancel()
    attitude_task.cancel()
    
    writer.close()
    
    # Save environment simulation data for comparison
    output_dir = 'aquila/output'
    os.makedirs(output_dir, exist_ok=True)
    env_sim_data_file = os.path.join(output_dir, 'test_data_comparing.pkl')
    with open(env_sim_data_file, 'wb') as f:
        pickle.dump(env_sim_data, f)
    print(f"✅ Environment simulation data saved to: {env_sim_data_file}")
    
    print(f"✅ TensorBoard logs saved to: {log_path}")
    print(f"   View with: tensorboard --logdir={log_path}\n")


if __name__ == "__main__":
    asyncio.run(run())
