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
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer5 import TrackEnvVer5
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


def safe_norm(x, eps=1e-8):
    """计算向量的安全范数"""
    return jnp.sqrt(jnp.sum(x * x) + eps)


def compute_angle_between_body_z_and_target(quad_R, quad_pos, target_pos):
    """计算无人机z轴与到目标物体方向的夹角（度数）"""
    # 机体z轴在世界坐标系中的方向（NED坐标系中，机体z轴向下）
    body_z_world = quad_R @ jnp.array([0.0, 0.0, 1.0])
    
    # 从无人机到目标的方向向量
    direction_to_target = target_pos - quad_pos
    direction_to_target_normalized = direction_to_target / safe_norm(direction_to_target)
    
    # 计算夹角的余弦值
    cos_angle = jnp.dot(body_z_world, direction_to_target_normalized)
    cos_angle = jnp.clip(cos_angle, -1.0, 1.0)
    
    # 转换为角度
    angle_rad = jnp.arccos(cos_angle)
    angle_deg = jnp.degrees(angle_rad)
    
    return angle_deg


def run_episode(env, policy, params, buffer_size, action_repeat, max_steps=1000, key=None):
    """运行一个episode并记录数据"""
    if key is None:
        key = jax.random.key(42)
    
    # 重置环境
    key, subkey = jax.random.split(key)
    state, obs = env.reset(subkey)
    
    # 数据记录列表
    data = {
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
        'angle_body_z_target': [],
    }
    
    # 初始化动作-状态缓冲区（与bptt.py训练代码保持一致）
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    
    # 获取原始环境并归一化悬停动作
    # ⚠️ 修复：使用该episode实际的thrust_max和omega_max（参数随机化后的值），而不是默认值
    original_env = env._env._env
    hovering_action_raw = original_env.hovering_action
    
    # 从state中获取实际的quad_params（支持参数随机化）
    # 注意：state在reset后包含实际的随机化参数
    actual_thrust_max = state.quad_params.thrust_max
    actual_omega_max = state.quad_params.omega_max
    
    # 确保omega_max维度正确
    # omega_max在QuadrotorParams中是长度为3的数组，取第一个值用于归一化
    if actual_omega_max.ndim > 0:
        actual_omega_max_scalar = actual_omega_max[0] if len(actual_omega_max) > 0 else actual_omega_max
    else:
        actual_omega_max_scalar = actual_omega_max
    
    # 计算归一化范围
    action_low_thrust = original_env.thrust_min * 4
    action_high_thrust = actual_thrust_max * 4
    
    # 归一化悬停动作
    hovering_thrust_raw = hovering_action_raw[0]
    hovering_thrust_normalized = 2.0 * (hovering_thrust_raw - action_low_thrust) / (action_high_thrust - action_low_thrust) - 1.0
    hovering_omega_normalized = jnp.zeros(3)  # 角速度分量为0
    
    hovering_action_normalized = jnp.concatenate([
        jnp.array([hovering_thrust_normalized]),
        hovering_omega_normalized
    ])
    
    # 创建初始的动作-状态组合缓冲区（使用归一化的悬停动作）
    action_obs_combined = jnp.concatenate([hovering_action_normalized, obs])
    action_obs_buffer = jnp.tile(action_obs_combined[None, :], (buffer_size, 1))
    
    # 获取初始动作（使用初始化的缓冲区通过网络生成）
    action_obs_buffer_flat = action_obs_buffer.flatten()
    initial_action = policy.apply(params, action_obs_buffer_flat[None, :])[0]
    
    # 动作计数器和当前动作
    action_counter = 0
    current_action = initial_action
    
    terminated = False
    truncated = False
    step = 0
    
    print(f"\n{'='*60}")
    print(f"开始运行测试 episode...")
    print(f"悬停动作（原始）: {hovering_action_raw}")
    print(f"悬停动作（归一化）: {hovering_action_normalized}")
    print(f"初始动作（网络输出）: {initial_action}")
    print(f"动作-状态缓冲区形状: {action_obs_buffer.shape}")
    print(f"{'='*60}\n")
    
    while not (terminated or truncated) and step < max_steps:
        # 记录当前状态
        data['time'].append(float(state.time))
        data['quad_pos'].append(np.array(state.quadrotor_state.p))
        data['quad_vel'].append(np.array(state.quadrotor_state.v))
        data['quad_R'].append(np.array(state.quadrotor_state.R))
        data['quad_omega'].append(np.array(state.quadrotor_state.omega))
        data['target_pos'].append(np.array(state.target_pos))
        data['target_vel'].append(np.array(state.target_vel))
        data['action'].append(np.array(current_action))
        
        # 计算距离
        distance = float(safe_norm(state.quadrotor_state.p - state.target_pos))
        data['distance'].append(distance)
        
        # 计算高度（NED坐标系，z为负表示在地面以上）
        height = float(-state.quadrotor_state.p[2])
        data['height'].append(height)
        
        # 计算机体z轴与到目标方向的夹角
        angle = float(compute_angle_between_body_z_and_target(
            state.quadrotor_state.R, 
            state.quadrotor_state.p, 
            state.target_pos
        ))
        data['angle_body_z_target'].append(angle)
        
        # 每action_repeat步获取一次新动作（与bptt.py训练代码逻辑一致）
        if action_counter % action_repeat == 0:
            # 步骤1：创建临时缓冲区用于获取新动作（用空动作+当前观测）
            action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            empty_action = jnp.zeros(action_dim)
            action_obs_combined_empty = jnp.concatenate([empty_action, obs])
            action_obs_buffer_for_input = action_obs_buffer_for_input.at[-1].set(action_obs_combined_empty)
            
            # 步骤2：使用临时缓冲区获取新动作
            action_obs_buffer_flat = action_obs_buffer_for_input.flatten()
            current_action = policy.apply(params, action_obs_buffer_flat[None, :])[0]
            
            # 步骤3：用新动作更新原始缓冲区（用于下次使用）
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            action_obs_combined_new = jnp.concatenate([current_action, obs])
            action_obs_buffer = action_obs_buffer.at[-1].set(action_obs_combined_new)
            
            action_counter = 1
        else:
            # 不需要新动作时，缓冲区保持不变（与训练一致）
            action_counter += 1
        
        # 执行动作
        key, subkey = jax.random.split(key)
        transition = env.step(state, current_action, subkey)
        state, obs, reward, terminated, truncated, info = transition
        
        # 记录奖励
        data['reward'].append(float(reward))
        
        step += 1
        
        # 每100步打印一次进度
        if step % 100 == 0:
            print(f"Step {step}: Distance={distance:.3f}m, Height={height:.3f}m, Reward={float(reward):.3f}")
    
    # 记录最后一个状态（不记录reward，因为没有新的动作执行）
    data['time'].append(float(state.time))
    data['quad_pos'].append(np.array(state.quadrotor_state.p))
    data['quad_vel'].append(np.array(state.quadrotor_state.v))
    data['quad_R'].append(np.array(state.quadrotor_state.R))
    data['quad_omega'].append(np.array(state.quadrotor_state.omega))
    data['target_pos'].append(np.array(state.target_pos))
    data['target_vel'].append(np.array(state.target_vel))
    data['action'].append(np.array(current_action))
    distance = float(safe_norm(state.quadrotor_state.p - state.target_pos))
    data['distance'].append(distance)
    height = float(-state.quadrotor_state.p[2])
    data['height'].append(height)
    angle = float(compute_angle_between_body_z_and_target(
        state.quadrotor_state.R, 
        state.quadrotor_state.p, 
        state.target_pos
    ))
    data['angle_body_z_target'].append(angle)
    # 不记录最后的reward，因为没有执行新的动作
    
    print(f"\n{'='*60}")
    print(f"Episode 结束!")
    print(f"总步数: {step}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"最终距离: {distance:.3f}m")
    print(f"最终高度: {height:.3f}m")
    print(f"{'='*60}\n")
    
    return data


def visualize_trajectory(data, output_dir):
    """可视化轨迹（3D图）"""
    quad_pos = np.array(data['quad_pos'])
    target_pos = np.array(data['target_pos'])
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制无人机轨迹
    ax.plot(quad_pos[:, 0], quad_pos[:, 1], -quad_pos[:, 2], 
            'b-', linewidth=2, label='Quadrotor trajectory', alpha=0.8)
    
    # 绘制目标轨迹
    ax.plot(target_pos[:, 0], target_pos[:, 1], -target_pos[:, 2], 
            'r--', linewidth=2, label='Target trajectory', alpha=0.8)
    
    # 标记起点和终点
    ax.scatter(quad_pos[0, 0], quad_pos[0, 1], -quad_pos[0, 2], 
              c='green', marker='o', s=100, label='Start (quad)')
    ax.scatter(quad_pos[-1, 0], quad_pos[-1, 1], -quad_pos[-1, 2], 
              c='blue', marker='s', s=100, label='End (quad)')
    ax.scatter(target_pos[0, 0], target_pos[0, 1], -target_pos[0, 2], 
              c='orange', marker='o', s=100, label='Start (target)')
    ax.scatter(target_pos[-1, 0], target_pos[-1, 1], -target_pos[-1, 2], 
              c='red', marker='s', s=100, label='End (target)')
    
    ax.set_xlabel('X (North) [m]', fontsize=12)
    ax.set_ylabel('Y (East) [m]', fontsize=12)
    ax.set_zlabel('Z (Height) [m]', fontsize=12)
    ax.set_title('3D Trajectory - Quadrotor Tracking Task', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置相同的坐标轴比例
    max_range = np.array([
        quad_pos[:, 0].max() - quad_pos[:, 0].min(),
        quad_pos[:, 1].max() - quad_pos[:, 1].min(),
        quad_pos[:, 2].max() - quad_pos[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (quad_pos[:, 0].max() + quad_pos[:, 0].min()) * 0.5
    mid_y = (quad_pos[:, 1].max() + quad_pos[:, 1].min()) * 0.5
    mid_z = (quad_pos[:, 2].max() + quad_pos[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(-mid_z - max_range, -mid_z + max_range)
    
    plt.tight_layout()
    trajectory_path = os.path.join(output_dir, 'trajectory_3d.png')
    plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
    print(f"✅ 轨迹图已保存: {trajectory_path}")
    plt.close()


def visualize_data(data, output_dir):
    """可视化数据（多子图）"""
    # reward数组比time数组少一个元素（最后一个状态没有对应的reward）
    # 所以我们使用time[:-1]来匹配reward的长度
    time = np.array(data['time'][:-1])
    actions = np.array(data['action'][:-1])
    height = np.array(data['height'][:-1])
    reward = np.array(data['reward'])
    angle = np.array(data['angle_body_z_target'][:-1])
    distance = np.array(data['distance'][:-1])
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 子图1: 无人机高度随时间变化
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, height, 'b-', linewidth=2, label='Quadrotor height')
    ax1.axhline(y=2.0, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Target height (2m)')
    ax1.set_xlabel('Time [s]', fontsize=11)
    ax1.set_ylabel('Height [m]', fontsize=11)
    ax1.set_title('Quadrotor Height vs Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 无人机reward随时间变化
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, reward, 'g-', linewidth=2)
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Reward', fontsize=11)
    ax2.set_title('Reward vs Time', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 无人机动作随时间变化（4个动作分量）
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, actions[:, 0], 'r-', linewidth=1.5, label='Thrust (normalized)', alpha=0.8)
    ax3.plot(time, actions[:, 1], 'g-', linewidth=1.5, label='ωx (roll rate)', alpha=0.8)
    ax3.plot(time, actions[:, 2], 'b-', linewidth=1.5, label='ωy (pitch rate)', alpha=0.8)
    ax3.plot(time, actions[:, 3], 'm-', linewidth=1.5, label='ωz (yaw rate)', alpha=0.8)
    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('Action (normalized [-1, 1])', fontsize=11)
    ax3.set_title('Action Commands vs Time', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, ncol=4, loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 无人机z轴与到目标物体方向夹角随时间变化
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(time, angle, 'purple', linewidth=2)
    ax4.axhline(y=90, color='r', linestyle='--', linewidth=1, alpha=0.7, 
                label='Perpendicular (90°)')
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('Angle [degrees]', fontsize=11)
    ax4.set_title('Body Z-axis to Target Angle vs Time', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 无人机到目标物体距离随时间变化
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(time, distance, 'orange', linewidth=2)
    ax5.axhline(y=1.0, color='r', linestyle='--', linewidth=1, alpha=0.7, 
                label='Target distance (1m)')
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Distance [m]', fontsize=11)
    ax5.set_title('Distance to Target vs Time', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('TrackVer5 Test Results - Data Analysis', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    data_path = os.path.join(output_dir, 'data_analysis.png')
    plt.savefig(data_path, dpi=300, bbox_inches='tight')
    print(f"✅ 数据分析图已保存: {data_path}")
    plt.close()


def main():
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # ==================== Load Policy ====================
    policy_file = 'aquila/param/trackVer5_policy.pkl'
    
    if not os.path.exists(policy_file):
        print(f"❌ 错误: 找不到训练好的模型文件: {policy_file}")
        print(f"   请先运行 train_trackVer5.py 进行训练")
        return
    
    params, env_config, action_repeat, buffer_size = load_trained_policy(policy_file)
    
    # ==================== Environment Setup ====================
    # 使用与训练相同的环境配置
    env = TrackEnvVer5(
        max_steps_in_episode=env_config.get('max_steps_in_episode', 1000),
        dt=env_config.get('dt', 0.01),
        delay=env_config.get('delay', 0.03),
        omega_std=0.1,
        action_penalty_weight=env_config.get('action_penalty_weight', 0.5),
        obs_tau_pos=env_config.get('obs_tau_pos', 0.3),
        obs_tau_vel=env_config.get('obs_tau_vel', 0.2),
        obs_tau_R=env_config.get('obs_tau_R', 0.02),
        target_height=env_config.get('target_height', 2.0),
        target_init_distance_min=env_config.get('target_init_distance_min', 0.5),
        target_init_distance_max=env_config.get('target_init_distance_max', 1.5),
        target_speed_max=env_config.get('target_speed_max', 1.0),
        reset_distance=env_config.get('reset_distance', 100.0),
        max_speed=env_config.get('max_speed', 20.0),
        thrust_to_weight_min=1.5,
        thrust_to_weight_max=3.0,
    )
    
    # 应用相同的wrapper
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # ==================== Model Setup ====================
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)
    
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    print(f"\n{'='*60}")
    print(f"环境配置:")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")
    print(f"  缓冲区大小: {buffer_size}")
    print(f"  输入维度: {input_dim}")
    print(f"  动作重复: {action_repeat}")
    print(f"  最大步数: {env.max_steps_in_episode}")
    print(f"{'='*60}")
    
    # ==================== Run Test ====================
    key = jax.random.key(42)  # 使用固定的随机种子以便复现
    data = run_episode(env, policy, params, buffer_size, action_repeat, 
                       max_steps=env.max_steps_in_episode, key=key)
    
    # ==================== Create Output Directory ====================
    output_dir = 'aquila/output'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    
    # ==================== Visualize Results ====================
    print(f"\n{'='*60}")
    print(f"开始生成可视化结果...")
    print(f"{'='*60}\n")
    
    visualize_trajectory(data, output_dir)
    visualize_data(data, output_dir)
    
    # ==================== Save Data ====================
    data_file = os.path.join(output_dir, 'test_data.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"✅ 测试数据已保存: {data_file}")
    
    print(f"\n{'='*60}")
    print(f"测试完成！所有结果已保存到: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

