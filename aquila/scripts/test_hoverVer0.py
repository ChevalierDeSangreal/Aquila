#!/usr/bin/env python
# coding: utf-8

import os
import sys

# ==================== GPU Configuration ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.hoverVer0 import HoverEnvVer0, ExtendedQuadrotorParams
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP


def load_trained_policy(checkpoint_path):
    """加载训练好的策略参数"""
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
    else:
        # 兼容旧格式
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
        action_repeat = 10
        buffer_size = 10
    
    print("✅ Policy parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    print(f"   Action repeat: {action_repeat}")
    print(f"   Buffer size: {buffer_size}")
    
    return params, env_config, action_repeat, buffer_size


def create_initial_state_at_target(env, key):
    """创建一个初始状态：无人机静止在目标悬停点"""
    # 获取默认的quad_params（带参数随机化）
    keys = jax.random.split(key, 2)
    key_reset, key_randomize = keys
    
    # 使用默认参数随机化
    base_params = env.quadrotor.default_params()
    randomized_params = env.quadrotor.randomize_params(
        base_params,
        env.quadrotor._mass,
        key_randomize,
        thrust_to_weight_min=env.thrust_to_weight_min,
        thrust_to_weight_max=env.thrust_to_weight_max
    )
    
    # 转换为扩展的参数类
    quad_params = ExtendedQuadrotorParams(
        thrust_max=randomized_params.thrust_max,
        omega_max=randomized_params.omega_max,
        motor_tau=randomized_params.motor_tau,
        Kp=randomized_params.Kp,
        mass=env.quadrotor._mass,
        gravity=9.81
    )
    
    # 设置位置为目标悬停点
    p = env.hover_origin
    
    # 设置速度为0
    v = jnp.array([0.0, 0.0, 0.0])
    
    # 设置姿态为水平（单位旋转矩阵）
    R = jnp.eye(3)
    
    # 设置角速度为0
    omega = jnp.array([0.0, 0.0, 0.0])
    
    # 创建quadrotor状态
    quadrotor_state = env.quadrotor.create_state(p, R, v, omega=omega)
    
    # 计算悬停动作
    thrust_hover = env.quadrotor._mass * 9.81
    hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
    
    # 初始化动作历史
    last_actions = jnp.tile(hovering_action, (env.num_last_actions, 1))
    action_raw = jnp.zeros(4)
    filtered_acc = jnp.array([0.0, 0.0, 9.81])
    filtered_thrust = thrust_hover
    
    # 创建state
    from aquila.envs.hoverVer0 import HoverStateVer0
    state = HoverStateVer0(
        time=0.0,
        step_idx=0,
        quadrotor_state=quadrotor_state,
        last_actions=last_actions,
        quad_params=quad_params,
        action_raw=action_raw,
        filtered_acc=filtered_acc,
        filtered_thrust=filtered_thrust,
        hover_origin=env.hover_origin,
    )
    
    return state, quad_params


def normalize_hovering_action(env, state):
    """归一化悬停动作（和训练时保持一致）"""
    original_env = env.unwrapped
    hovering_action_raw = original_env.hovering_action
    
    # 获取当前环境的实际参数
    actual_thrust_max = state.quad_params.thrust_max
    actual_omega_max = state.quad_params.omega_max
    
    # 确保维度正确
    if actual_omega_max.ndim == 1:
        actual_omega_max_scalar = actual_omega_max[0]
    else:
        actual_omega_max_scalar = actual_omega_max
    
    # 归一化推力
    action_low_thrust = original_env.thrust_min * 4
    action_high_thrust = actual_thrust_max * 4
    hovering_thrust_raw = hovering_action_raw[0]
    hovering_thrust_normalized = 2.0 * (hovering_thrust_raw - action_low_thrust) / (action_high_thrust - action_low_thrust) - 1.0
    
    # 归一化角速度（都是0）
    hovering_omega_normalized = jnp.zeros(3)
    
    # 组合归一化的悬停动作
    hovering_action_normalized = jnp.concatenate([
        jnp.array([hovering_thrust_normalized]),
        hovering_omega_normalized
    ])
    
    return hovering_action_normalized


def run_episode(env, policy, params, state, max_steps=1000, action_repeat=10, buffer_size=10):
    """运行一个episode，记录所有数据"""
    # 数据记录
    positions = []
    velocities = []
    actions = []
    rewards = []
    distances = []
    heights = []
    times = []
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 通过reset获取正确处理（归一化）的观测
    key = jax.random.key(42)
    state, obs = env.reset(key, state)
    
    # 初始化动作-状态缓冲区（和训练时保持一致）
    hovering_action_normalized = normalize_hovering_action(env, state)
    action_obs_combined = jnp.concatenate([hovering_action_normalized, obs])
    action_obs_buffer = jnp.tile(action_obs_combined[None, :], (buffer_size, 1))
    
    # 初始化动作计数器
    action_counter = 0
    current_action = hovering_action_normalized
    
    for step in range(max_steps):
        # 记录当前状态
        positions.append(np.array(state.quadrotor_state.p))
        velocities.append(np.array(state.quadrotor_state.v))
        times.append(float(state.time))
        
        # 计算距离和高度
        distance_to_origin = float(jnp.linalg.norm(state.quadrotor_state.p - state.hover_origin))
        height = float(-state.quadrotor_state.p[2])  # NED坐标系，z是向下的
        distances.append(distance_to_origin)
        heights.append(height)
        
        # 每action_repeat步获取新动作（和训练时保持一致）
        if action_counter % action_repeat == 0:
            # 步骤1：用空动作+当前观测更新缓冲区
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            empty_action = jnp.zeros(action_dim)
            action_obs_combined_empty = jnp.concatenate([empty_action, obs])
            action_obs_buffer = action_obs_buffer.at[-1, :].set(action_obs_combined_empty)
            
            # 步骤2：使用缓冲区获取新动作
            action_obs_buffer_flat = action_obs_buffer.reshape(-1)
            current_action = policy.apply(params, action_obs_buffer_flat)
            
            # 步骤3：用新动作更新缓冲区
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            action_obs_combined_new = jnp.concatenate([current_action, obs])
            action_obs_buffer = action_obs_buffer.at[-1, :].set(action_obs_combined_new)
            
            action_counter = 1
        else:
            action_counter += 1
        
        # 记录动作（原始归一化动作）
        actions.append(np.array(current_action))
        
        # 执行动作
        key, subkey = jax.random.split(key)
        transition = env.step(state, current_action, subkey)
        state, obs, reward, terminated, truncated, info = transition
        
        # 记录奖励
        rewards.append(float(reward))
        
        # 检查终止条件
        if terminated or truncated:
            print(f"Episode terminated at step {step}")
            break
    
    # 转换为numpy数组
    positions = np.array(positions)
    velocities = np.array(velocities)
    actions = np.array(actions)
    rewards = np.array(rewards)
    distances = np.array(distances)
    heights = np.array(heights)
    times = np.array(times)
    
    return {
        'positions': positions,
        'velocities': velocities,
        'actions': actions,
        'rewards': rewards,
        'distances': distances,
        'heights': heights,
        'times': times,
        'target_pos': np.array(state.hover_origin)
    }


def plot_trajectory_3d(data, save_path):
    """绘制3D轨迹图"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = data['positions']
    target_pos = data['target_pos']
    
    # 绘制轨迹
    ax.plot(positions[:, 0], positions[:, 1], -positions[:, 2], 
            'b-', linewidth=2, label='Drone Trajectory', alpha=0.7)
    
    # 绘制起点
    ax.scatter(positions[0, 0], positions[0, 1], -positions[0, 2], 
               c='green', s=100, marker='o', label='Start', zorder=5)
    
    # 绘制终点
    ax.scatter(positions[-1, 0], positions[-1, 1], -positions[-1, 2], 
               c='red', s=100, marker='o', label='End', zorder=5)
    
    # 绘制目标点
    ax.scatter(target_pos[0], target_pos[1], -target_pos[2], 
               c='orange', s=200, marker='*', label='Target (Hover Point)', zorder=5)
    
    # 设置标签和标题
    ax.set_xlabel('X (North) [m]', fontsize=12)
    ax.set_ylabel('Y (East) [m]', fontsize=12)
    ax.set_zlabel('Z (Up) [m]', fontsize=12)
    ax.set_title('Drone Hovering Trajectory (3D)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置相同的坐标轴范围
    max_range = np.max([
        np.max(np.abs(positions[:, 0])),
        np.max(np.abs(positions[:, 1])),
        np.max(np.abs(-positions[:, 2]))
    ])
    max_range = max(max_range, 0.5)  # 至少0.5m的范围
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range * 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D trajectory plot saved to: {save_path}")
    plt.close()


def plot_data_analysis(data, save_path):
    """绘制数据分析图"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    times = data['times']
    actions = data['actions']
    rewards = data['rewards']
    distances = data['distances']
    heights = data['heights']
    velocities = data['velocities']
    
    # 1. 动作随时间变化
    ax = axes[0, 0]
    ax.plot(times, actions[:, 0], label='Thrust (normalized)', linewidth=1.5)
    ax.plot(times, actions[:, 1], label='ωx (normalized)', linewidth=1.5, alpha=0.7)
    ax.plot(times, actions[:, 2], label='ωy (normalized)', linewidth=1.5, alpha=0.7)
    ax.plot(times, actions[:, 3], label='ωz (normalized)', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Normalized Action', fontsize=11)
    ax.set_title('Actions vs Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Reward随时间变化
    ax = axes[0, 1]
    ax.plot(times, rewards, 'r-', linewidth=1.5)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Reward', fontsize=11)
    ax.set_title('Reward vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 3. 距离随时间变化
    ax = axes[1, 0]
    ax.plot(times, distances, 'g-', linewidth=1.5)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Distance to Target [m]', fontsize=11)
    ax.set_title('Distance to Hover Point vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.1m)')
    ax.legend(fontsize=9)
    
    # 4. 高度随时间变化
    ax = axes[1, 1]
    target_height = -data['target_pos'][2]  # NED坐标系转换
    ax.plot(times, heights, 'b-', linewidth=1.5, label='Actual Height')
    ax.axhline(y=target_height, color='orange', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Target Height ({target_height:.1f}m)')
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Height [m]', fontsize=11)
    ax.set_title('Height vs Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. 速度模长随时间变化
    ax = axes[2, 0]
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    ax.plot(times, velocity_magnitude, 'm-', linewidth=1.5)
    ax.set_xlabel('Time [s]', fontsize=11)
    ax.set_ylabel('Velocity Magnitude [m/s]', fontsize=11)
    ax.set_title('Velocity Magnitude vs Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.1m/s)')
    ax.legend(fontsize=9)
    
    # 6. 统计信息
    ax = axes[2, 1]
    ax.axis('off')
    
    # 计算统计信息
    stats_text = f"""
    Episode Statistics:
    
    Duration: {times[-1]:.2f} s
    Total Steps: {len(times)}
    
    Distance to Target:
      Mean: {np.mean(distances):.4f} m
      Min: {np.min(distances):.4f} m
      Max: {np.max(distances):.4f} m
      Final: {distances[-1]:.4f} m
    
    Height Error:
      Mean: {np.mean(np.abs(heights - target_height)):.4f} m
      Final: {np.abs(heights[-1] - target_height):.4f} m
    
    Velocity:
      Mean: {np.mean(velocity_magnitude):.4f} m/s
      Max: {np.max(velocity_magnitude):.4f} m/s
      Final: {velocity_magnitude[-1]:.4f} m/s
    
    Reward:
      Mean: {np.mean(rewards):.4f}
      Total: {np.sum(rewards):.4f}
      Final: {rewards[-1]:.4f}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Data analysis plot saved to: {save_path}")
    plt.close()


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # ==================== Load Policy ====================
    policy_file = 'aquila/param/hoverVer0_policy.pkl'
    params, env_config, action_repeat, buffer_size = load_trained_policy(policy_file)
    
    # ==================== Environment Setup ====================
    env = HoverEnvVer0(
        max_steps_in_episode=2000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        action_penalty_weight=0.1,
        hover_height=2.0,
        init_pos_range=0.5,
        max_distance=10.0,
        max_speed=20.0,
        thrust_to_weight_min=1.2,
        thrust_to_weight_max=3.0,
    )
    
    # 应用和训练时相同的wrapper
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # ==================== Model Setup ====================
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)
    
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    print(f"\n{'='*60}")
    print(f"Testing HoverVer0 Policy")
    print(f"{'='*60}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Buffer size: {buffer_size}")
    print(f"Action repeat: {action_repeat}")
    print(f"Input dimension: {input_dim}")
    print(f"{'='*60}\n")
    
    # ==================== Create Initial State ====================
    key = jax.random.key(0)
    state, quad_params = create_initial_state_at_target(env, key)
    
    print(f"Initial state created:")
    print(f"  Position: {state.quadrotor_state.p}")
    print(f"  Velocity: {state.quadrotor_state.v}")
    print(f"  Target: {state.hover_origin}")
    print(f"  Distance to target: {jnp.linalg.norm(state.quadrotor_state.p - state.hover_origin):.6f} m")
    print(f"  Thrust max: {quad_params.thrust_max:.2f} N")
    print(f"  Omega max: {quad_params.omega_max}")
    print()
    
    # ==================== Run Episode ====================
    print("Running test episode...")
    data = run_episode(
        env, policy, params, state, 
        max_steps=2000, 
        action_repeat=action_repeat,
        buffer_size=buffer_size
    )
    
    print(f"\n✅ Episode completed!")
    print(f"   Total steps: {len(data['times'])}")
    print(f"   Duration: {data['times'][-1]:.2f} s")
    print(f"   Final distance to target: {data['distances'][-1]:.4f} m")
    print(f"   Final height: {data['heights'][-1]:.4f} m (target: {-data['target_pos'][2]:.4f} m)")
    print(f"   Mean reward: {np.mean(data['rewards']):.4f}")
    
    # ==================== Save Results ====================
    # 确保输出目录存在
    output_dir = 'aquila/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    data_path = os.path.join(output_dir, 'test_hoverVer0_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"\n✅ Test data saved to: {data_path}")
    
    # ==================== Visualization ====================
    print("\nGenerating visualizations...")
    
    # 3D轨迹图
    trajectory_path = os.path.join(output_dir, 'hoverVer0_trajectory_3d.png')
    plot_trajectory_3d(data, trajectory_path)
    
    # 数据分析图
    analysis_path = os.path.join(output_dir, 'hoverVer0_data_analysis.png')
    plot_data_analysis(data, analysis_path)
    
    print(f"\n{'='*60}")
    print(f"Test completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

