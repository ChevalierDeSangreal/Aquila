#!/usr/bin/env python
# coding: utf-8

import os
import sys

# ==================== GPU Configuration ====================
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import time
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer11 import TrackEnvVer11
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP
from aquila.utils.trajectory_utils import (
    TrajectoryGenerator,
    CircularTrajectory,
    create_trajectory
)


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
        action_repeat = data.get('action_repeat', 1)
        buffer_size = data.get('action_obs_buffer_size', 1)
    else:
        # 兼容旧格式
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
        action_repeat = 1
        buffer_size = 1
    
    print("✅ Policy parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    print(f"   Action repeat: {action_repeat}")
    print(f"   Action-obs buffer size: {buffer_size}")
    
    return params, env_config, action_repeat, buffer_size


def compute_distance_to_boundary(position, boundary_half_x, boundary_half_y, boundary_z):
    """计算位置到边界的最小距离（正值表示在边界内，负值表示超出边界）"""
    dist_to_px = boundary_half_x - position[0]  # 距离+X面
    dist_to_nx = position[0] + boundary_half_x  # 距离-X面
    dist_to_py = boundary_half_y - position[1]  # 距离+Y面
    dist_to_ny = position[1] + boundary_half_y  # 距离-Y面
    dist_to_pz = 0.0 - position[2]  # 距离+Z面（地面，z=0）
    dist_to_nz = position[2] + boundary_z  # 距离-Z面（顶部，z=-boundary_z）
    
    min_distance = min(dist_to_px, dist_to_nx, dist_to_py, dist_to_ny, dist_to_pz, dist_to_nz)
    return min_distance


def ensure_trajectory_within_boundary(trajectory: CircularTrajectory,
                                     boundary_half_x: float,
                                     boundary_half_y: float,
                                     boundary_z: float,
                                     safety_margin: float = 0.5) -> CircularTrajectory:
    """
    确保圆形轨迹在边界内（考虑安全边距）
    
    Args:
        trajectory: 原始圆形轨迹生成器
        boundary_half_x: X方向边界的一半（NED坐标系）
        boundary_half_y: Y方向边界的一半
        boundary_z: Z方向边界（从-boundary_z到0）
        safety_margin: 安全边距（米）
        
    Returns:
        调整后的轨迹生成器（新实例，不修改原实例）
    """
    # 对于圆形轨迹，确保圆心位置和半径的组合使得整个圆都在边界内
    # 圆的任何点都在: [center_x ± radius, center_y ± radius]
    
    # 先确定圆心位置的范围（考虑半径）
    # 如果圆心位置超出边界，先调整它（居中）
    center_x = 0.0  # 居中
    center_y = 0.0  # 居中
    
    # 计算最大允许半径（基于居中的圆心位置）
    max_radius_x = boundary_half_x - safety_margin
    max_radius_y = boundary_half_y - safety_margin
    max_radius = min(max_radius_x, max_radius_y)
    
    # 如果半径太大，缩小它
    adjusted_radius = min(trajectory.radius, max_radius)
    
    # 如果半径太小，使用一个合理的最小值
    if adjusted_radius < 0.5:
        adjusted_radius = min(1.0, max_radius)
    
    # 确保z坐标在边界内（居中在高度的一半）
    center_z = np.clip(
        trajectory.center_z,
        -boundary_z + safety_margin + 1.0,  # 确保有足够空间
        -safety_margin - 1.0
    )
    
    # 如果center_z没有设置或超出范围，使用边界中间值
    if center_z > -safety_margin or center_z < -boundary_z + safety_margin:
        center_z = -boundary_z * 0.5  # 在边界高度的一半处
    
    # 创建新的圆形轨迹
    return CircularTrajectory(
        center=(float(center_x), float(center_y), float(center_z)),
        radius=float(adjusted_radius),
        num_circles=trajectory.num_circles,
        ramp_up_time=trajectory.ramp_up_time,
        ramp_down_time=trajectory.ramp_down_time,
        circle_duration=trajectory.circle_duration,
        init_phase=trajectory.init_phase,
        max_speed=trajectory.max_speed
    )


def run_test_episode(env, policy_apply, params, key, action_repeat, buffer_size, 
                     trajectory: CircularTrajectory = None, verbose=False):
    """
    运行一个测试episode，记录跟踪和边界信息
    
    Args:
        env: 环境实例
        policy_apply: 策略应用函数
        params: 策略参数
        key: JAX随机数生成器密钥
        action_repeat: 动作重复次数
        buffer_size: 动作-观测缓冲区大小
        trajectory: 圆形轨迹生成器，如果提供则覆盖环境的默认目标运动
        verbose: 是否打印详细信息
    """
    # Reset environment（先reset获取基本状态结构）
    state, obs = env.reset(key)
    
    # 如果提供了轨迹生成器，先确保轨迹在边界内，然后根据轨迹初始化
    if trajectory is not None:
        # 1. 确保轨迹在边界内
        trajectory = ensure_trajectory_within_boundary(
            trajectory,
            state.boundary_half_x,
            state.boundary_half_y,
            state.boundary_z,
            safety_margin=0.5
        )
        
        # 2. 获取轨迹的初始位置（t=0时）
        target_initial_pos, target_initial_vel = trajectory.get_state(0.0)
        target_initial_pos = np.array(target_initial_pos)  # 转换为numpy以便计算
        target_initial_vel = jnp.array(target_initial_vel)
        
        # 3. 根据目标初始位置，初始化无人机在目标正后方1m处（NED坐标系，-X方向）
        # 这样目标就在无人机正前方1m处
        quad_initial_pos_np = target_initial_pos - np.array([1.0, 0.0, 0.0])  # 正后方1m
        
        # 确保无人机位置也在边界内（考虑安全边距）
        quad_initial_pos_np = np.clip(
            quad_initial_pos_np,
            np.array([-state.boundary_half_x + 1.0, -state.boundary_half_y + 1.0, -state.boundary_z + 1.0]),
            np.array([state.boundary_half_x - 1.0, state.boundary_half_y - 1.0, -1.0])
        )
        quad_initial_pos = jnp.array(quad_initial_pos_np)
        target_initial_pos = jnp.array(target_initial_pos)
        
        # 4. 更新state中的目标位置、速度和无人机位置
        import dataclasses
        from aquila.objects.quadrotor_obj import QuadrotorState
        
        # 更新无人机状态（保持其他属性不变，只更新位置）
        new_quadrotor_state = dataclasses.replace(
            state.quadrotor_state,
            p=quad_initial_pos
        )
        
        # 更新整个state
        state = dataclasses.replace(
            state,
            quadrotor_state=new_quadrotor_state,
            target_pos=target_initial_pos,
            target_vel=target_initial_vel
        )
        
        # 5. 重新计算观测
        obs = env._get_obs(state)
    
    # 初始化动作-状态缓冲区
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # ⚠️ 重要：缓冲区格式必须与训练时一致：[action, obs]（先动作，后观测）
    # 缓冲区形状：(buffer_size, action_dim + obs_dim)
    action_obs_buffer = jnp.zeros((buffer_size, action_dim + obs_dim))
    
    # 初始化：填充零动作和零观测（与训练时的初始化一致）
    zero_action = jnp.zeros(action_dim)
    zero_obs = jnp.zeros(obs_dim)
    action_obs_combined = jnp.concatenate([zero_action, zero_obs])
    action_obs_buffer = jnp.tile(action_obs_combined[None, :], (buffer_size, 1))
    
    # Episode statistics
    episode_data = {
        'quad_positions': [],
        'target_positions': [],
        'distances': [],
        'boundary_distances': [],
        'rewards': [],
        'actions': [],
        'velocities': [],
        'terminated': False,
        'truncated': False,
        'num_steps': 0,
        'boundary_violations': 0,
        'min_boundary_distance': float('inf'),
    }
    
    done = False
    step_count = 0
    action = jnp.zeros(action_dim)  # 初始动作为零
    action_counter = 0  # 动作计数器，用于action_repeat
    
    # 如果提供了轨迹生成器，记录轨迹起始时间
    trajectory_start_time = 0.0 if trajectory else None
    
    while not done and step_count < env.max_steps_in_episode:
        # 每action_repeat步获取一次新动作
        if action_counter % action_repeat == 0:
            # ⚠️ 与训练时一致的逻辑：
            # 步骤1：先用空动作+当前观测更新缓冲区（为获取新动作做准备）
            action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            empty_action = jnp.zeros(action_dim)
            action_obs_combined_empty = jnp.concatenate([empty_action, obs])
            action_obs_buffer_for_input = action_obs_buffer_for_input.at[-1].set(action_obs_combined_empty)
            
            # 步骤2：展平缓冲区作为网络输入：[action[0], obs[0], action[1], obs[1], ...]
            network_input = action_obs_buffer_for_input.flatten()
            
            # 步骤3：获取新动作
            action = policy_apply(params, network_input)
            
            # 步骤4：用获取到的新动作更新缓冲区（用于下次使用）
            action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
            action_obs_combined_new = jnp.concatenate([action, obs])
            action_obs_buffer = action_obs_buffer.at[-1].set(action_obs_combined_new)
        
        # 执行动作
        key, subkey = jax.random.split(key)
        transition = env.step(state, action, subkey)
        next_state, next_obs, reward, terminated, truncated, info = transition
        
        # 如果提供了轨迹生成器，覆盖目标位置和速度
        if trajectory is not None:
            current_time = trajectory_start_time + step_count * env.dt
            traj_pos, traj_vel = trajectory.get_state(current_time)
            
            # 确保目标位置在边界内（运行时安全检查）
            traj_pos_clipped = jnp.clip(
                traj_pos,
                jnp.array([-state.boundary_half_x, -state.boundary_half_y, -state.boundary_z]),
                jnp.array([state.boundary_half_x, state.boundary_half_y, 0.0])
            )
            
            # 更新状态中的目标位置和速度
            import dataclasses
            next_state = dataclasses.replace(
                next_state,
                target_pos=traj_pos_clipped,
                target_vel=traj_vel
            )
            
            # 重新计算观测（因为目标位置改变了）
            next_obs = env._get_obs(next_state)
        
        # 记录数据
        quad_pos = np.array(info['quad_p'])
        target_pos = np.array(info['target_p'])
        distance = np.array(info['distance_to_target'])
        
        # 计算到边界的距离
        boundary_dist = compute_distance_to_boundary(
            quad_pos, 
            state.boundary_half_x, 
            state.boundary_half_y, 
            state.boundary_z
        )
        
        episode_data['quad_positions'].append(quad_pos)
        episode_data['target_positions'].append(target_pos)
        episode_data['distances'].append(distance)
        episode_data['boundary_distances'].append(boundary_dist)
        episode_data['rewards'].append(float(reward))
        episode_data['actions'].append(np.array(action))
        episode_data['velocities'].append(np.array(info['quad_v']))
        episode_data['target_velocities'] = episode_data.get('target_velocities', [])
        episode_data['target_velocities'].append(np.array(info['target_v']))
        
        # 检查边界违规（超出边界）
        if boundary_dist < 0:
            episode_data['boundary_violations'] += 1
        
        # 更新最小边界距离
        episode_data['min_boundary_distance'] = min(
            episode_data['min_boundary_distance'], 
            boundary_dist
        )
        
        # 更新状态和观测
        state = next_state
        obs = next_obs  # 更新obs以便下次获取动作时使用
        done = terminated or truncated
        step_count += 1
        action_counter += 1
        
        if verbose and step_count % 100 == 0:
            print(f"  Step {step_count}: Distance={distance:.3f}m, Boundary={boundary_dist:.3f}m, Reward={reward:.3f}")
    
    episode_data['terminated'] = bool(terminated)
    episode_data['truncated'] = bool(truncated)
    episode_data['num_steps'] = step_count
    
    return episode_data


def visualize_episode(episode_data, boundary_half_x, boundary_half_y, boundary_z, episode_idx=0, save_path=None):
    """可视化单个episode的结果"""
    fig = plt.figure(figsize=(24, 16))
    
    # 1. 3D轨迹图
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    quad_positions = np.array(episode_data['quad_positions'])
    target_positions = np.array(episode_data['target_positions'])
    
    ax1.plot(quad_positions[:, 0], quad_positions[:, 1], quad_positions[:, 2], 
             'b-', label='Quadrotor', linewidth=2, alpha=0.7)
    ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
             'r--', label='Target', linewidth=2, alpha=0.7)
    
    # 绘制边界框（NED坐标系：z范围[-boundary_z, 0]）
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices = [
        [-boundary_half_x, -boundary_half_y, -boundary_z],
        [boundary_half_x, -boundary_half_y, -boundary_z],
        [boundary_half_x, boundary_half_y, -boundary_z],
        [-boundary_half_x, boundary_half_y, -boundary_z],
        [-boundary_half_x, -boundary_half_y, 0],
        [boundary_half_x, -boundary_half_y, 0],
        [boundary_half_x, boundary_half_y, 0],
        [-boundary_half_x, boundary_half_y, 0],
    ]
    
    # 绘制边界框的边
    edges = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # 顶面
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # 底面
        [vertices[0], vertices[4]], [vertices[1], vertices[5]],  # 垂直边
        [vertices[2], vertices[6]], [vertices[3], vertices[7]],
    ]
    
    for edge in edges[:2]:
        poly = Poly3DCollection([edge], alpha=0.1, facecolor='gray', edgecolor='black', linewidth=2)
        ax1.add_collection3d(poly)
    
    ax1.set_xlabel('X (North) [m]')
    ax1.set_ylabel('Y (East) [m]')
    ax1.set_zlabel('Z (Down) [m]')
    ax1.set_title(f'Episode {episode_idx}: 3D Trajectory (NED frame)')
    ax1.legend()
    ax1.grid(True)
    
    # 设置相等的坐标轴比例
    max_range = max(boundary_half_x, boundary_half_y, boundary_z)
    ax1.set_xlim([-max_range, max_range])
    ax1.set_ylim([-max_range, max_range])
    ax1.set_zlim([-max_range, max_range])
    
    # 2. XY平面投影（俯视图）
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(quad_positions[:, 0], quad_positions[:, 1], 'b-', label='Quadrotor', linewidth=2, alpha=0.7)
    ax2.plot(target_positions[:, 0], target_positions[:, 1], 'r--', label='Target', linewidth=2, alpha=0.7)
    
    # 绘制边界
    boundary_rect = Rectangle(
        (-boundary_half_x, -boundary_half_y), 
        2 * boundary_half_x, 
        2 * boundary_half_y,
        fill=False, edgecolor='black', linewidth=2, linestyle='--'
    )
    ax2.add_patch(boundary_rect)
    
    ax2.set_xlabel('X (North) [m]')
    ax2.set_ylabel('Y (East) [m]')
    ax2.set_title(f'Episode {episode_idx}: Top View (XY plane)')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')
    
    # 3. 跟踪距离随时间变化
    ax3 = fig.add_subplot(3, 3, 3)
    distances = np.array(episode_data['distances'])
    ax3.plot(distances, 'b-', linewidth=2)
    ax3.axhline(y=1.0, color='r', linestyle='--', label='Target Distance (1m)')
    ax3.axhline(y=1.3, color='orange', linestyle=':', label='Acceptable Range (±30cm)')
    ax3.axhline(y=0.7, color='orange', linestyle=':')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Distance to Target [m]')
    ax3.set_title(f'Episode {episode_idx}: Tracking Distance')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 到边界的距离随时间变化
    ax4 = fig.add_subplot(3, 3, 4)
    boundary_distances = np.array(episode_data['boundary_distances'])
    ax4.plot(boundary_distances, 'g-', linewidth=2)
    ax4.axhline(y=0, color='r', linestyle='--', label='Boundary')
    ax4.axhline(y=1.0, color='orange', linestyle=':', label='Warning Zone (1m)')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Distance to Boundary [m]')
    ax4.set_title(f'Episode {episode_idx}: Boundary Distance (negative=violation)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 奖励随时间变化
    ax5 = fig.add_subplot(3, 3, 5)
    rewards = np.array(episode_data['rewards'])
    ax5.plot(rewards, 'purple', linewidth=2)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Reward')
    ax5.set_title(f'Episode {episode_idx}: Rewards')
    ax5.grid(True)
    
    # 6. 目标物体速度随时间变化
    ax6 = fig.add_subplot(3, 3, 6)
    target_velocities = np.array(episode_data.get('target_velocities', []))
    if len(target_velocities) > 0:
        target_speed = np.linalg.norm(target_velocities, axis=1)
        ax6.plot(target_speed, 'r-', linewidth=2, label='Target Speed')
        ax6.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, label='Max Speed (1 m/s)')
        ax6.set_xlabel('Step')
        ax6.set_ylabel('Speed [m/s]')
        ax6.set_title(f'Episode {episode_idx}: Target Speed')
        ax6.legend()
        ax6.grid(True)
        ax6.set_ylim(bottom=0)
    
    # 7. 无人机速度随时间变化
    ax7 = fig.add_subplot(3, 3, 7)
    velocities = np.array(episode_data['velocities'])
    quad_speed = np.linalg.norm(velocities, axis=1)
    ax7.plot(quad_speed, 'b-', linewidth=2, label='Quad Speed')
    ax7.set_xlabel('Step')
    ax7.set_ylabel('Speed [m/s]')
    ax7.set_title(f'Episode {episode_idx}: Quadrotor Speed')
    ax7.legend()
    ax7.grid(True)
    ax7.set_ylim(bottom=0)
    
    # 8. 目标速度向量（3个分量）
    ax8 = fig.add_subplot(3, 3, 8)
    if len(target_velocities) > 0:
        ax8.plot(target_velocities[:, 0], 'r-', alpha=0.7, label='Vx (North)', linewidth=1.5)
        ax8.plot(target_velocities[:, 1], 'g-', alpha=0.7, label='Vy (East)', linewidth=1.5)
        ax8.plot(target_velocities[:, 2], 'b-', alpha=0.7, label='Vz (Down)', linewidth=1.5)
        ax8.set_xlabel('Step')
        ax8.set_ylabel('Velocity [m/s]')
        ax8.set_title(f'Episode {episode_idx}: Target Velocity Components')
        ax8.legend()
        ax8.grid(True)
    
    # 9. 统计信息文本
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)
    mean_reward = np.mean(rewards)
    total_reward = np.sum(rewards)
    min_boundary_dist = episode_data['min_boundary_distance']
    boundary_violations = episode_data['boundary_violations']
    
    # 计算跟踪成功率（距离在目标±30cm内的比例）
    tracking_success_rate = np.mean(np.abs(distances - 1.0) < 0.3) * 100
    
    # 计算目标速度统计
    target_velocities = np.array(episode_data.get('target_velocities', []))
    if len(target_velocities) > 0:
        target_speed = np.linalg.norm(target_velocities, axis=1)
        mean_target_speed = np.mean(target_speed)
        max_target_speed = np.max(target_speed)
    else:
        mean_target_speed = 0.0
        max_target_speed = 0.0
    
    stats_text = f"""
    Episode {episode_idx} Statistics:
    
    Steps: {episode_data['num_steps']}
    Terminated: {episode_data['terminated']}
    Truncated: {episode_data['truncated']}
    
    Tracking Performance:
    • Mean Distance: {mean_distance:.3f} m
    • Std Distance: {std_distance:.3f} m
    • Min Distance: {min_distance:.3f} m
    • Max Distance: {max_distance:.3f} m
    • Success Rate (±30cm): {tracking_success_rate:.1f}%
    
    Boundary Avoidance:
    • Min Boundary Dist: {min_boundary_dist:.3f} m
    • Boundary Violations: {boundary_violations}
    • Violation Rate: {boundary_violations/episode_data['num_steps']*100:.1f}%
    
    Target Motion:
    • Mean Speed: {mean_target_speed:.3f} m/s
    • Max Speed: {max_target_speed:.3f} m/s
    
    Rewards:
    • Mean Reward: {mean_reward:.3f}
    • Total Reward: {total_reward:.3f}
    """
    
    ax9.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: {save_path}")
    
    return fig


def main():
    # ==================== Configuration ====================
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # ==================== Circular Trajectory Setup ====================
    # 创建居中的圆形轨迹
    # 参数会在环境创建后根据边界进行调整
    trajectory = create_trajectory(
        'circular',
        center=(0.0, 0.0, -2.0),  # 临时中心，会根据边界调整
        radius=2.0,                # 半径（米），会根据边界调整
        num_circles=2,             # 圆圈数量
        ramp_up_time=3.0,          # 加速时间（秒）
        ramp_down_time=3.0,        # 减速时间（秒）
        circle_duration=20.0,      # 单圈名义持续时间（秒）
        init_phase=0.0,            # 初始相位（弧度）
        max_speed=1.0              # 最大速度限制（m/s）
    )
    
    print(f"\n{'='*60}")
    print(f"Circular Trajectory Configuration:")
    print(f"{'='*60}")
    traj_info = trajectory.get_info()
    for key, value in traj_info.items():
        print(f"  {key}: {value}")
    print(f"Note: Trajectory will be adjusted to fit within boundaries")
    print(f"{'='*60}\n")
    
    # Load trained policy
    policy_file = 'aquila/param/trackVer11_policy.pkl'
    
    if not os.path.exists(policy_file):
        print(f"❌ Error: Policy file not found: {policy_file}")
        print("   Please train the model first using train_trackVer11.py")
        return
    
    params, env_config, action_repeat, buffer_size = load_trained_policy(policy_file)
    
    # ==================== Environment Setup ====================
    # Create env with same configuration as training
    env = TrackEnvVer11(
        max_steps_in_episode=env_config.get('max_steps_in_episode', 1000),
        dt=env_config.get('dt', 0.01),
        delay=env_config.get('delay', 0.03),
        omega_std=0.1,
        action_penalty_weight=env_config.get('action_penalty_weight', 0.5),
        target_height=env_config.get('target_height', 2.0),
        target_init_distance_min=env_config.get('target_init_distance_min', 0.5),
        target_init_distance_max=env_config.get('target_init_distance_max', 1.5),
        target_speed_max=env_config.get('target_speed_max', 2.0),
        reset_distance=env_config.get('reset_distance', 100.0),
        max_speed=env_config.get('max_speed', 20.0),
        boundary_x=env_config.get('boundary_x', 10.0),
        boundary_y=env_config.get('boundary_y', 10.0),
        boundary_z=env_config.get('boundary_z', 10.0),
        boundary_penalty_distance=env_config.get('boundary_penalty_distance', 1.0),
        boundary_penalty_max=env_config.get('boundary_penalty_max', 100.0),
        thrust_to_weight_min=1.2,
        thrust_to_weight_max=5.0,
        disturbance_mag=0.0,  # 测试时关闭扰动
    )
    
    # Apply wrappers
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # ==================== Model Setup ====================
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)
    
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    print(f"\n{'='*60}")
    print(f"Test Configuration:")
    print(f"{'='*60}")
    print(f"Environment: TrackEnvVer11 (with boundary constraints)")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action repeat: {action_repeat}")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Boundary: x=[{-env.boundary_half_x:.1f}, {env.boundary_half_x:.1f}], "
          f"y=[{-env.boundary_half_y:.1f}, {env.boundary_half_y:.1f}], "
          f"z=[{-env.boundary_z:.1f}, 0.0]")
    print(f"Disturbance: DISABLED (for testing)")
    print(f"{'='*60}\n")
    
    # ==================== Run Test Episodes ====================
    num_test_episodes = 1
    print(f"Running {num_test_episodes} test episode...\n")
    
    key = jax.random.key(42)  # 使用固定种子以便复现
    
    all_episode_data = []
    
    for episode_idx in range(num_test_episodes):
        print(f"Episode {episode_idx + 1}/{num_test_episodes}:")
        key, subkey = jax.random.split(key)
        
        episode_data = run_test_episode(
            env, policy.apply, params, subkey, 
            action_repeat, buffer_size, 
            trajectory=trajectory,  # 传递轨迹生成器
            verbose=True
        )
        
        all_episode_data.append(episode_data)
        
        # Print episode summary
        mean_distance = np.mean(episode_data['distances'])
        min_boundary_dist = episode_data['min_boundary_distance']
        boundary_violations = episode_data['boundary_violations']
        tracking_success_rate = np.mean(np.abs(np.array(episode_data['distances']) - 1.0) < 0.3) * 100
        
        print(f"  ✓ Completed {episode_data['num_steps']} steps")
        print(f"    Mean tracking distance: {mean_distance:.3f}m")
        print(f"    Tracking success rate (±30cm): {tracking_success_rate:.1f}%")
        print(f"    Min boundary distance: {min_boundary_dist:.3f}m")
        print(f"    Boundary violations: {boundary_violations} ({boundary_violations/episode_data['num_steps']*100:.1f}%)")
        print(f"    Terminated: {episode_data['terminated']}")
        print()
    
    # ==================== Aggregate Statistics ====================
    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"{'='*60}\n")
    
    # Tracking performance
    all_distances = np.concatenate([np.array(ep['distances']) for ep in all_episode_data])
    mean_distance_all = np.mean(all_distances)
    std_distance_all = np.std(all_distances)
    tracking_success_rate_all = np.mean(np.abs(all_distances - 1.0) < 0.3) * 100
    
    print("📊 Tracking Performance:")
    print(f"   • Mean distance to target: {mean_distance_all:.3f} ± {std_distance_all:.3f} m")
    print(f"   • Success rate (±30cm from 1m): {tracking_success_rate_all:.1f}%")
    print(f"   • Min distance: {np.min(all_distances):.3f} m")
    print(f"   • Max distance: {np.max(all_distances):.3f} m")
    
    # Boundary avoidance
    total_violations = sum(ep['boundary_violations'] for ep in all_episode_data)
    total_steps = sum(ep['num_steps'] for ep in all_episode_data)
    min_boundary_distances = [ep['min_boundary_distance'] for ep in all_episode_data]
    mean_min_boundary_dist = np.mean(min_boundary_distances)
    
    print(f"\n🛡️  Boundary Avoidance:")
    print(f"   • Total boundary violations: {total_violations} / {total_steps} steps")
    print(f"   • Violation rate: {total_violations/total_steps*100:.2f}%")
    print(f"   • Mean minimum boundary distance: {mean_min_boundary_dist:.3f} m")
    print(f"   • Episodes with violations: {sum(1 for ep in all_episode_data if ep['boundary_violations'] > 0)}")
    
    # Termination statistics
    num_terminated = sum(1 for ep in all_episode_data if ep['terminated'])
    num_truncated = sum(1 for ep in all_episode_data if ep['truncated'])
    
    print(f"\n📈 Episode Statistics:")
    print(f"   • Episode terminated early: {num_terminated} (tracking lost)")
    print(f"   • Episode completed fully: {num_truncated} (reached max steps)")
    print(f"   • Episode length: {total_steps} steps")
    
    # Overall assessment
    print(f"\n{'='*60}")
    print(f"Overall Assessment:")
    print(f"{'='*60}")
    
    tracking_passed = tracking_success_rate_all >= 70  # 至少70%的时间在±30cm内
    boundary_passed = (total_violations / total_steps) < 0.01  # 违规率低于1%
    
    print(f"✓ Tracking Test: {'PASSED ✅' if tracking_passed else 'FAILED ❌'}")
    print(f"  (Success rate {tracking_success_rate_all:.1f}% {'≥' if tracking_passed else '<'} 70%)")
    
    print(f"✓ Boundary Avoidance Test: {'PASSED ✅' if boundary_passed else 'FAILED ❌'}")
    print(f"  (Violation rate {total_violations/total_steps*100:.2f}% {'<' if boundary_passed else '≥'} 1%)")
    
    if tracking_passed and boundary_passed:
        print(f"\n🎉 All tests PASSED! The policy successfully tracks the target and avoids boundaries.")
    else:
        print(f"\n⚠️  Some tests FAILED. The policy needs further training or tuning.")
    
    # ==================== Visualization ====================
    print(f"\n{'='*60}")
    print(f"Generating visualizations...")
    print(f"{'='*60}\n")
    
    # Create output directory
    output_dir = 'aquila/output/trackVer11'
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize episode
    print(f"Visualizing episode...")
    save_path = f'{output_dir}/episode_1.png'
    fig = visualize_episode(
        all_episode_data[0], 
        env.boundary_half_x, 
        env.boundary_half_y, 
        env.boundary_z,
        episode_idx=1,
        save_path=save_path
    )
    plt.close(fig)
    
    # Create summary plot
    print("Creating summary plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distance distribution
    ax = axes[0, 0]
    ax.hist(all_distances, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Target (1m)')
    ax.axvline(x=0.7, color='orange', linestyle=':', linewidth=2, label='Acceptable Range')
    ax.axvline(x=1.3, color='orange', linestyle=':', linewidth=2)
    ax.set_xlabel('Distance to Target [m]')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Tracking Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Boundary distance distribution
    ax = axes[0, 1]
    all_boundary_distances = np.concatenate([np.array(ep['boundary_distances']) for ep in all_episode_data])
    ax.hist(all_boundary_distances, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Boundary')
    ax.axvline(x=1.0, color='orange', linestyle=':', linewidth=2, label='Warning Zone (1m)')
    ax.set_xlabel('Distance to Boundary [m]')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Boundary Distances')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Tracking distance over time
    ax = axes[1, 0]
    distances = np.array(all_episode_data[0]['distances'])
    ax.plot(distances, 'b-', linewidth=2, alpha=0.7)
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Target (1m)')
    ax.axhline(y=1.3, color='orange', linestyle=':', linewidth=1, label='Acceptable Range')
    ax.axhline(y=0.7, color='orange', linestyle=':', linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Target [m]')
    ax.set_title('Tracking Distance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Boundary distance over time
    ax = axes[1, 1]
    boundary_distances = np.array(all_episode_data[0]['boundary_distances'])
    ax.plot(boundary_distances, 'g-', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Boundary')
    ax.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, label='Warning Zone (1m)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Distance to Boundary [m]')
    ax.set_title('Boundary Distance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = f'{output_dir}/summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"  Summary plot saved to: {summary_path}")
    plt.close(fig)
    
    print(f"\n✅ All visualizations saved to: {output_dir}/")
    print(f"\nTest completed! 🎉")


if __name__ == "__main__":
    main()

