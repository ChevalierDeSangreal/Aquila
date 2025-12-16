#!/usr/bin/env python
# coding: utf-8

import os
import sys

# ==================== GPU Configuration ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer10 import TrackEnvVer10, ExtendedQuadrotorParams
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
        action_repeat = data.get('action_repeat', 10)
        buffer_size = data.get('action_obs_buffer_size', 10)
        final_loss = data.get('final_loss', 'Unknown')
        training_epochs = data.get('num_epochs', 'Unknown')
    else:
        # 兼容旧格式
        params = data
        env_config = {}
        action_repeat = 10
        buffer_size = 10
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
    
    print("✅ Policy parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    print(f"   Action repeat: {action_repeat}")
    print(f"   Action-obs buffer size: {buffer_size}")
    
    return params, env_config, action_repeat, buffer_size


def get_action_from_policy(policy, params, action_obs_buffer):
    """从策略网络获取动作
    
    Args:
        policy: MLP策略网络
        params: 网络参数
        action_obs_buffer: 动作-状态缓冲区，形状 (buffer_size, obs_dim + action_dim)
    
    Returns:
        action: 归一化后的动作 (4,)，范围 [-1, 1]
    """
    # 展平缓冲区作为网络输入
    buffer_flat = action_obs_buffer.reshape(-1)
    # 获取动作
    action = policy.apply(params, buffer_flat)
    return action


def normalize_hovering_action(hovering_action_raw, thrust_min, thrust_max, omega_max):
    """将原始悬停动作归一化到 [-1, 1] 范围
    
    Args:
        hovering_action_raw: 原始悬停动作 [thrust, wx, wy, wz]
        thrust_min: 推力最小值
        thrust_max: 推力最大值
        omega_max: 角速度最大值
    
    Returns:
        归一化后的悬停动作
    """
    thrust_raw = hovering_action_raw[0]
    omega_raw = hovering_action_raw[1:]
    
    # 推力归一化：[thrust_min*4, thrust_max*4] -> [-1, 1]
    action_low_thrust = thrust_min * 4
    action_high_thrust = thrust_max * 4
    thrust_normalized = 2.0 * (thrust_raw - action_low_thrust) / (action_high_thrust - action_low_thrust) - 1.0
    
    # 角速度归一化：[-omega_max, omega_max] -> [-1, 1]
    omega_normalized = omega_raw / omega_max
    
    return jnp.concatenate([jnp.array([thrust_normalized]), omega_normalized])


def test_policy():
    """测试训练好的策略"""
    
    # ==================== Load Policy ====================
    # policy_file = 'aquila/param/trackVer8_policy_stabler.pkl'
    policy_file = 'aquila/param/trackVer10_policy.pkl'
    params, env_config, action_repeat, buffer_size = load_trained_policy(policy_file)
    
    # ==================== Environment Setup ====================
    # 创建环境（与训练时相同的配置）
    env = TrackEnvVer10(
        max_steps_in_episode=1000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        action_penalty_weight=0.5,
        target_height=2.0,
        target_init_distance_min=0.5,
        target_init_distance_max=1.5,
        target_speed_max=1.0,
        reset_distance=100.0,
        max_speed=20.0,
        thrust_to_weight_min=3.0,
        thrust_to_weight_max=3.1,
    )
    
    # 应用与训练时相同的wrapper
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # ==================== Model Setup ====================
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)
    
    # 创建与训练时相同的网络
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    print(f"\n{'='*60}")
    print(f"Testing TrackVer10 Policy")
    print(f"{'='*60}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Action repeat: {action_repeat}")
    print(f"{'='*60}\n")
    
    # ==================== Custom Initialization ====================
    key = jax.random.key(42)
    
    # 先调用一次reset获取环境中随机化好的参数
    temp_key, key = jax.random.split(key)
    temp_state, _ = env.reset(temp_key)
    # 从返回的state中提取随机化后的quad_params
    quad_params = temp_state.quad_params
    
    print(f"Using randomized quadrotor parameters:")
    print(f"  thrust_max: {quad_params.thrust_max:.3f} N")
    print(f"  omega_max: {quad_params.omega_max}")
    print(f"  motor_tau: {quad_params.motor_tau:.4f} s")
    print(f"  mass: {quad_params.mass:.3f} kg")
    print(f"  gravity: {quad_params.gravity:.2f} m/s²")
    
    # 创建自定义初始状态
    # 无人机：原点，静止，高度2m（NED坐标系中z=-2）
    quad_p = jnp.array([0.0, 0.0, -2.0])
    quad_v = jnp.zeros(3)
    quad_R = jnp.eye(3)  # 无旋转
    quad_omega = jnp.zeros(3)
    
    # 创建 quadrotor state
    quadrotor_state = env.unwrapped.quadrotor.create_state(
        quad_p, quad_R, quad_v, 
        omega=quad_omega, 
        dr_key=key
    )
    
    # 目标：正前方0.5m处，静止
    target_pos = jnp.array([1, 0.0, -2.0])
    target_vel = jnp.zeros(3)
    target_direction = jnp.array([1.0, 0.0, 0.0])  # x方向
    
    # 计算悬停动作（使用随机化后的参数）
    thrust_hover = quad_params.mass * quad_params.gravity
    hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
    
    # 初始化last_actions
    num_last_actions = env.unwrapped.num_last_actions
    last_actions = jnp.tile(hovering_action, (num_last_actions, 1))
    
    # 创建初始state（使用环境的TrackStateVer10类）
    from aquila.envs.target_trackVer10 import TrackStateVer10
    state = TrackStateVer10(
        time=0.0,
        step_idx=0,
        quadrotor_state=quadrotor_state,
        last_actions=last_actions,
        target_pos=target_pos,
        target_vel=target_vel,
        target_direction=target_direction,
        quad_params=quad_params,
        target_speed_max=0.0,  # 目标静止
        action_raw=jnp.zeros(4),
        filtered_acc=jnp.array([0.0, 0.0, 9.81]),
        filtered_thrust=jnp.array(thrust_hover),
        has_exceeded_distance=False,
    )
    
    # 通过reset获取正确处理（归一化）的观测
    state, obs = env.reset(key, state)
    
    print(f"Initial quad position: {state.quadrotor_state.p}")
    print(f"Initial target position: {state.target_pos}")
    print(f"Initial distance: {jnp.linalg.norm(state.quadrotor_state.p - state.target_pos):.3f}m")
    print(f"Target velocity: {state.target_vel}")
    print(f"\nStarting simulation...\n")
    
    # ==================== 初始化动作-状态缓冲区（与训练时完全一致）====================
    # 获取归一化的悬停动作
    thrust_min = env.unwrapped.thrust_min
    thrust_max = quad_params.thrust_max
    omega_max_val = quad_params.omega_max
    if isinstance(omega_max_val, jnp.ndarray) and omega_max_val.ndim > 0:
        omega_max_val = omega_max_val[0]
    
    hovering_action_normalized = normalize_hovering_action(
        hovering_action, thrust_min, thrust_max, omega_max_val
    )
    
    # 初始化缓冲区：使用零向量作为观测填充（与训练时一致）
    zero_obs = jnp.zeros_like(obs)  # 使用零向量代替实际观测
    action_obs_combined = jnp.concatenate([hovering_action_normalized, zero_obs])
    action_obs_buffer = jnp.tile(action_obs_combined[None, :], (buffer_size, 1))
    
    # 在初始化时获取第一个动作（使用填充的缓冲区）
    initial_action = get_action_from_policy(policy, params, action_obs_buffer)
    
    # 初始化动作计数器
    action_counter = 0
    current_action = initial_action
    
    # ==================== Simulation Loop ====================
    max_steps = 1000
    
    # 数据记录
    positions = []
    target_positions = []
    velocities = []
    accelerations = []
    actions_thrust = []
    actions_omega = []
    rewards = []
    distances = []
    times = []
    heights = []
    
    for step in range(max_steps):
        # ==================== 动作选择（与训练时完全一致）====================
        # 每action_repeat步才获取一次新动作
        if action_counter % action_repeat == 0:
            # 步骤1：用空动作+当前观测更新缓冲区
            action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            empty_action = jnp.zeros(4)
            action_obs_combined_empty = jnp.concatenate([empty_action, obs])
            action_obs_buffer_for_input = action_obs_buffer_for_input.at[-1, :].set(action_obs_combined_empty)
            
            # 步骤2：获取新动作
            current_action = get_action_from_policy(policy, params, action_obs_buffer_for_input)
            
            # 步骤3：用新动作+当前观测更新缓冲区
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            action_obs_combined_new = jnp.concatenate([current_action, obs])
            action_obs_buffer = action_obs_buffer.at[-1, :].set(action_obs_combined_new)
            
            # 重置计数器为1
            action_counter = 1
        else:
            # 使用上一个动作，计数器+1
            action_counter += 1
        
        # ==================== Environment Step ====================
        # 执行动作（current_action已经是归一化的，环境会处理）
        transition = env.step(state, current_action, key)
        state, obs, reward, terminated, truncated, info = transition
        
        # 记录数据
        positions.append(np.array(state.quadrotor_state.p))
        target_positions.append(np.array(state.target_pos))
        velocities.append(np.array(state.quadrotor_state.v))
        accelerations.append(np.array(state.quadrotor_state.acc))
        actions_thrust.append(float(state.action_raw[0]))  # 使用归一化的原始输出 [-1, 1]
        actions_omega.append(np.array(state.action_raw[1:]))  # 使用归一化的原始输出
        rewards.append(float(reward))
        distances.append(float(info['distance_to_target']))
        times.append(float(state.time))
        heights.append(float(state.quadrotor_state.p[2]))
        
        # 打印进度
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{max_steps}, "
                  f"Distance: {info['distance_to_target']:.3f}m, "
                  f"Reward: {reward:.3f}, "
                  f"Height: {state.quadrotor_state.p[2]:.3f}m")
        
        # 检查是否终止
        if terminated or truncated:
            print(f"\nSimulation terminated at step {step + 1}")
            if terminated:
                print("Reason: Distance exceeded reset threshold")
            if truncated:
                print("Reason: Max steps reached")
            break
        
        # 更新key
        key, _ = jax.random.split(key)
    
    # ==================== Data Processing ====================
    positions = np.array(positions)
    target_positions = np.array(target_positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    actions_thrust = np.array(actions_thrust)
    actions_omega = np.array(actions_omega)
    rewards = np.array(rewards)
    distances = np.array(distances)
    times = np.array(times)
    heights = np.array(heights)
    
    # ==================== Save Data ====================
    output_data = {
        'positions': positions,
        'target_positions': target_positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'actions_thrust': actions_thrust,
        'actions_omega': actions_omega,
        'rewards': rewards,
        'distances': distances,
        'times': times,
        'heights': heights,
        'env_config': env_config,
        'action_repeat': action_repeat,
        'buffer_size': buffer_size,
    }
    
    os.makedirs('aquila/output', exist_ok=True)
    output_file = 'aquila/output/test_trackVer10_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\n✅ Test data saved to: {output_file}")
    
    # ==================== Visualization ====================
    print("\nGenerating visualizations...")
    
    # 图1: 3D轨迹图
    fig1 = plt.figure(figsize=(12, 10))
    ax1 = fig1.add_subplot(111, projection='3d')
    
    # 绘制无人机轨迹
    ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
             'b-', linewidth=2, label='Drone Trajectory', alpha=0.8)
    
    # 绘制起点和终点
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
                c='green', s=100, marker='o', label='Start Position')
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
                c='red', s=100, marker='x', label='End Position')
    
    # 绘制目标轨迹
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
             'r--', linewidth=2, label='Target Position', alpha=0.6)
    
    # 绘制目标初始位置
    ax1.scatter(target_positions[0, 0], target_positions[0, 1], target_positions[0, 2], 
                c='orange', s=100, marker='*', label='Target Start')
    
    # NED坐标系：z轴向下为正，需要翻转以便可视化
    ax1.set_xlabel('X (North) [m]')
    ax1.set_ylabel('Y (East) [m]')
    ax1.set_zlabel('Z (Down) [m]')
    ax1.set_title('TrackVer10: Drone Tracking Trajectory (3D)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 设置相同的刻度范围
    all_pos = np.vstack([positions, target_positions])
    x_range = [all_pos[:, 0].min() - 0.5, all_pos[:, 0].max() + 0.5]
    y_range = [all_pos[:, 1].min() - 0.5, all_pos[:, 1].max() + 0.5]
    z_range = [all_pos[:, 2].min() - 0.5, all_pos[:, 2].max() + 0.5]
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    
    plt.tight_layout()
    trajectory_file = 'aquila/output/trackVer10_trajectory_3d.png'
    plt.savefig(trajectory_file, dpi=150, bbox_inches='tight')
    print(f"✅ 3D trajectory plot saved to: {trajectory_file}")
    plt.close()
    
    # 图2: 数据分析图（动作、reward、高度、距离等）
    fig2, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # 子图1: 推力动作（归一化原始输出）
    axes[0, 0].plot(times, actions_thrust, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Thrust (normalized)')
    axes[0, 0].set_title('Thrust Command over Time (Normalized Output)')
    axes[0, 0].set_ylim([-1.1, 1.1])
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2: 角速度动作（归一化原始输出）
    axes[0, 1].plot(times, actions_omega[:, 0], 'r-', linewidth=1.5, label='ωx (Roll rate)')
    axes[0, 1].plot(times, actions_omega[:, 1], 'g-', linewidth=1.5, label='ωy (Pitch rate)')
    axes[0, 1].plot(times, actions_omega[:, 2], 'b-', linewidth=1.5, label='ωz (Yaw rate)')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Angular Velocity (normalized)')
    axes[0, 1].set_title('Angular Velocity Commands over Time (Normalized Output)')
    axes[0, 1].set_ylim([-1.1, 1.1])
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3: Reward
    axes[1, 0].plot(times, rewards, 'purple', linewidth=1.5)
    axes[1, 0].set_xlabel('Time [s]')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].set_title('Reward over Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 子图4: 距离到目标
    axes[1, 1].plot(times, distances, 'orange', linewidth=1.5)
    axes[1, 1].set_xlabel('Time [s]')
    axes[1, 1].set_ylabel('Distance [m]')
    axes[1, 1].set_title('Distance to Target over Time')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target distance (1m)')
    axes[1, 1].legend()
    
    # 子图5: 高度（Z坐标，NED系）
    axes[2, 0].plot(times, heights, 'brown', linewidth=1.5, label='Drone height')
    axes[2, 0].plot(times, target_positions[:, 2], 'r--', linewidth=1.5, label='Target height')
    axes[2, 0].set_xlabel('Time [s]')
    axes[2, 0].set_ylabel('Z (Down) [m]')
    axes[2, 0].set_title('Height (Z coordinate in NED) over Time')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 子图6: 速度模长
    velocity_norm = np.linalg.norm(velocities, axis=1)
    axes[2, 1].plot(times, velocity_norm, 'teal', linewidth=1.5)
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Velocity [m/s]')
    axes[2, 1].set_title('Drone Velocity Magnitude over Time')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_file = 'aquila/output/trackVer10_data_analysis.png'
    plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
    print(f"✅ Data analysis plot saved to: {analysis_file}")
    plt.close()
    
    # ==================== Summary Statistics ====================
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Total steps: {len(times)}")
    print(f"Total time: {times[-1]:.2f}s")
    print(f"Final distance to target: {distances[-1]:.3f}m")
    print(f"Average distance to target: {np.mean(distances):.3f}m")
    print(f"Min distance to target: {np.min(distances):.3f}m")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Total reward: {np.sum(rewards):.3f}")
    print(f"Final position: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]")
    print(f"Target position: [{target_positions[-1, 0]:.3f}, {target_positions[-1, 1]:.3f}, {target_positions[-1, 2]:.3f}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_policy()

