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
import dataclasses
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer9 import TrackEnvVer9, ExtendedQuadrotorParams
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP


# ==================== Target Motion Pattern Interface ====================
class TargetMotionPattern(ABC):
    """目标运动模式的抽象基类"""
    
    @abstractmethod
    def get_initial_position(self, drone_pos):
        """返回目标的初始位置
        
        Args:
            drone_pos: 无人机初始位置 (3,)
            
        Returns:
            target_pos: 目标初始位置 (3,)
        """
        pass
    
    @abstractmethod
    def update(self, current_pos, current_vel, dt):
        """更新目标位置和速度
        
        Args:
            current_pos: 当前目标位置 (3,)
            current_vel: 当前目标速度 (3,)
            dt: 时间步长
            
        Returns:
            new_pos: 新的目标位置 (3,)
            new_vel: 新的目标速度 (3,)
        """
        pass


class CircularMotion(TargetMotionPattern):
    """圆周运动模式"""
    
    def __init__(self, speed=2.0, radius=2.0, center=None, plane_normal=None, acceleration_time=2.0):
        """
        Args:
            speed: 目标运动速度 (m/s)
            radius: 圆周半径 (m)
            center: 圆心位置，默认为原点
            plane_normal: 圆周平面的法向量，默认为 [0, 0, 1] (水平面)
            acceleration_time: 加速到目标速度所需的时间 (s)，默认2秒
        """
        self.speed = speed
        self.radius = radius
        self.center = np.array(center) if center is not None else np.array([0.0, 0.0, -2.0])
        self.acceleration_time = acceleration_time
        
        # 法向量归一化
        if plane_normal is None:
            plane_normal = np.array([0.0, 0.0, 1.0])  # 默认水平面
        self.plane_normal = np.array(plane_normal)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        
        # 计算平面内的两个正交基向量
        self._compute_plane_basis()
        
        self.time = 0.0
        self.current_angle = 0.0  # 当前角度（弧度）
        
    def _compute_plane_basis(self):
        """计算圆周平面内的两个正交基向量"""
        # 找一个不平行于法向量的向量
        if abs(self.plane_normal[0]) < 0.9:
            arbitrary = np.array([1.0, 0.0, 0.0])
        else:
            arbitrary = np.array([0.0, 1.0, 0.0])
        
        # 第一个基向量：法向量叉乘任意向量
        self.basis1 = np.cross(self.plane_normal, arbitrary)
        self.basis1 = self.basis1 / np.linalg.norm(self.basis1)
        
        # 第二个基向量：法向量叉乘第一个基向量
        self.basis2 = np.cross(self.plane_normal, self.basis1)
        self.basis2 = self.basis2 / np.linalg.norm(self.basis2)
    
    def get_initial_position(self, drone_pos):
        """返回目标的初始位置：在无人机正前方1m处（使用全局坐标系x轴方向）"""
        # 确保目标在无人机正前方1m处（使用全局坐标系的x轴正方向）
        # 注意：这里使用全局坐标，后续会根据无人机朝向调整
        initial_offset = np.array([1.0, 0.0, 0.0])  # x轴正方向1m
        return np.array(drone_pos) + initial_offset
    
    def update(self, current_pos, current_vel, dt):
        """更新目标位置和速度（圆周运动，带加速过程）
        
        注意：圆周运动以初始位置为起点，需要重新计算圆心位置，使得：
        1. 初始位置（无人机正前方1m）在圆周上
        2. 圆周在指定的平面内（由plane_normal定义）
        """
        # 如果是第一次更新，需要根据初始位置重新计算圆心
        if not hasattr(self, '_center_initialized'):
            # 初始位置应该是无人机正前方1m处（在current_pos中）
            # 我们需要让初始位置成为圆周上的一个点
            # 
            # 将初始位置投影到圆周平面上，确保圆周在指定平面内
            initial_pos = np.array(current_pos)
            
            # 计算从用户指定的center到初始位置的向量在平面内的投影
            # 这个向量确定了初始角度为0时的方向（basis1方向）
            vec_to_initial = initial_pos - self.center
            
            # 将vec_to_initial投影到圆周平面上（垂直于plane_normal）
            # vec_in_plane = vec_to_initial - (vec_to_initial · plane_normal) * plane_normal
            dot_product = np.dot(vec_to_initial, self.plane_normal)
            vec_in_plane = vec_to_initial - dot_product * self.plane_normal
            
            # 如果投影向量的模长太小，使用basis1作为方向
            vec_norm = np.linalg.norm(vec_in_plane)
            if vec_norm < 1e-6:
                # 使用basis1方向
                direction_from_center = self.basis1
            else:
                # 归一化投影向量
                direction_from_center = vec_in_plane / vec_norm
            
            # 重新计算圆心：圆心 = 初始位置 - direction_from_center * radius
            # 这样初始位置就在圆周上
            self.center = initial_pos - direction_from_center * self.radius
            
            # 重新计算平面基向量，使得basis1指向初始位置的方向（从圆心指向初始位置）
            self.basis1 = direction_from_center
            # basis2需要与basis1正交，并且在平面内
            self.basis2 = np.cross(self.plane_normal, self.basis1)
            self.basis2 = self.basis2 / (np.linalg.norm(self.basis2) + 1e-8)
            
            self._center_initialized = True
            # 初始角度为0（对应basis1方向，即从圆心指向初始位置）
            self.current_angle = 0.0
        
        self.time += dt
        
        # 计算当前实际速度（从0加速到目标速度）
        if self.acceleration_time > 0 and self.time < self.acceleration_time:
            # 加速阶段：线性加速
            current_speed = self.speed * (self.time / self.acceleration_time)
        else:
            # 已达到目标速度
            current_speed = self.speed
        
        # 计算当前角速度（根据实际速度）
        current_angular_velocity = current_speed / self.radius
        
        # 更新角度（使用数值积分）
        self.current_angle += current_angular_velocity * dt
        
        # 计算圆周上的位置
        pos_offset = self.radius * (np.cos(self.current_angle) * self.basis1 + 
                                    np.sin(self.current_angle) * self.basis2)
        new_pos = self.center + pos_offset
        
        # 计算切向速度方向
        tangent_direction = -np.sin(self.current_angle) * self.basis1 + \
                           np.cos(self.current_angle) * self.basis2
        
        # 使用当前实际速度
        new_vel = current_speed * tangent_direction
        
        return new_pos, new_vel


class StaticTarget(TargetMotionPattern):
    """静止目标"""
    
    def __init__(self, position=None):
        """
        Args:
            position: 目标位置，默认为 [1, 0, -2]
        """
        self.position = np.array(position) if position is not None else np.array([1.0, 0.0, -2.0])
    
    def get_initial_position(self, drone_pos):
        """返回目标的初始位置：在无人机正前方1m处"""
        # 确保目标在无人机正前方1m处（使用全局坐标系的x轴正方向）
        initial_offset = np.array([1.0, 0.0, 0.0])  # x轴正方向1m
        return np.array(drone_pos) + initial_offset
    
    def update(self, current_pos, current_vel, dt):
        """静止目标不移动（保持在初始位置）"""
        # 返回当前初始位置（在无人机正前方1m处）
        return np.array(current_pos), np.zeros(3)


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


def build_rotation_matrix_to_target(target_direction, preferred_up=None):
    """构建旋转矩阵，使得无人机x轴指向目标方向
    
    Args:
        target_direction: 从无人机指向目标的方向向量（归一化）
        preferred_up: 首选的上方向（默认[0,0,1]向下，NED坐标系）
    
    Returns:
        R: 3x3旋转矩阵，使得R[:, 0]（x轴）指向target_direction
    """
    if preferred_up is None:
        preferred_up = jnp.array([0.0, 0.0, 1.0])  # NED坐标系，向下为正
    
    # x轴 = 目标方向（归一化）
    x_axis = target_direction / (jnp.linalg.norm(target_direction) + 1e-8)
    
    # 计算y轴：使用preferred_up与x轴的叉积
    y_axis_candidate = jnp.cross(preferred_up, x_axis)
    y_norm = jnp.linalg.norm(y_axis_candidate)
    
    # 如果叉积结果太小（x轴与preferred_up几乎平行），使用备用方向
    backup_up = jnp.array([0.0, 1.0, 0.0])  # y轴方向作为备用
    y_axis_candidate2 = jnp.cross(backup_up, x_axis)
    y_norm2 = jnp.linalg.norm(y_axis_candidate2)
    
    # 选择较大的叉积结果（使用Python条件，因为这是初始化代码，不需要JIT）
    if y_norm < 0.1:
        y_axis = y_axis_candidate2 / (y_norm2 + 1e-8)
    else:
        y_axis = y_axis_candidate / (y_norm + 1e-8)
    
    # z轴 = x轴 × y轴（确保右手坐标系）
    z_axis = jnp.cross(x_axis, y_axis)
    z_axis = z_axis / (jnp.linalg.norm(z_axis) + 1e-8)
    
    # 重新计算y轴以确保正交性
    y_axis = jnp.cross(z_axis, x_axis)
    y_axis = y_axis / (jnp.linalg.norm(y_axis) + 1e-8)
    
    # 构建旋转矩阵（列向量：x, y, z轴）
    R = jnp.stack([x_axis, y_axis, z_axis], axis=1)
    
    return R


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


def test_policy(motion_pattern=None, test_duration=10.0):
    """测试训练好的策略
    
    Args:
        motion_pattern: TargetMotionPattern实例，定义目标的运动模式
        test_duration: 测试总时间（秒），默认10秒
    """
    
    # ==================== Target Motion Pattern Setup ====================
    if motion_pattern is None:
        # 默认：速度2m/s的圆周运动，半径2m，在水平面上，2秒加速到目标速度
        motion_pattern = CircularMotion(
            speed=2.0,
            radius=2.0,
            center=np.array([0.0, 0.0, -2.0]),
            plane_normal=np.array([0.0, 0.0, 1.0]),  # 水平面
            acceleration_time=2.0  # 2秒加速到目标速度
        )
    
    # ==================== Load Policy ====================
    policy_file = 'aquila/param/trackVer9_policy.pkl'
    params, env_config, action_repeat, buffer_size = load_trained_policy(policy_file)
    
    # ==================== Environment Setup ====================
    # 根据测试时间计算环境的最大步数
    dt = 0.01
    max_steps_in_episode = int(test_duration / dt)
    
    # 创建环境（与训练时相同的配置）
    env = TrackEnvVer9(
        max_steps_in_episode=max_steps_in_episode,  # 根据测试时间动态设置
        dt=dt,
        delay=0.03,
        omega_std=0.1,
        action_penalty_weight=0.5,
        target_height=2.0,
        target_init_distance_min=0.5,
        target_init_distance_max=1.5,
        target_speed_max=1.0,  # 与训练时一致（train_trackVer9.py中使用1.0）
        reset_distance=100.0,
        max_speed=20.0,
        thrust_to_weight_min=1.2,
        thrust_to_weight_max=5.0,
        disturbance_mag=0.0,  # 测试时不添加扰动
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
    print(f"Testing TrackVer9 Policy")
    print(f"{'='*60}")
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Action repeat: {action_repeat}")
    print(f"Target motion pattern: {motion_pattern.__class__.__name__}")
    print(f"Test duration: {test_duration:.1f} seconds")
    print(f"{'='*60}\n")
    
    # ==================== Custom Initialization ====================
    key = jax.random.key(42)
    
    # 创建固定的quadrotor参数（使用默认值，不随机化）
    default_params_base = env.unwrapped.quadrotor.default_params()
    quad_params = ExtendedQuadrotorParams(
        thrust_max=default_params_base.thrust_max,
        omega_max=default_params_base.omega_max,
        motor_tau=default_params_base.motor_tau,
        mass=env.unwrapped.quadrotor._mass,
        gravity=9.81
    )
    
    # 创建自定义初始状态
    # 无人机：原点，静止，高度2m（NED坐标系中z=-2）
    quad_p = jnp.array([0.0, 0.0, -2.0])
    quad_v = jnp.zeros(3)
    quad_omega = jnp.zeros(3)
    
    # 使用运动模式获取目标初始位置
    target_pos_np = motion_pattern.get_initial_position(np.array(quad_p))
    target_pos = jnp.array(target_pos_np)
    target_vel_np = np.zeros(3)
    target_vel = jnp.array(target_vel_np)
    
    # 计算目标方向（从无人机指向目标）
    direction = target_pos - quad_p
    direction_norm = jnp.linalg.norm(direction)
    if direction_norm > 1e-6:
        target_direction = direction / direction_norm
    else:
        target_direction = jnp.array([1.0, 0.0, 0.0])  # 默认x方向
    
    # 构建旋转矩阵，使得无人机x轴正对目标
    quad_R = build_rotation_matrix_to_target(target_direction)
    
    # 创建 quadrotor state
    quadrotor_state = env.unwrapped.quadrotor.create_state(
        quad_p, quad_R, quad_v, 
        omega=quad_omega, 
        dr_key=key
    )
    
    # 计算悬停动作
    thrust_hover = env.unwrapped.quadrotor._mass * 9.81
    hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
    
    # 初始化last_actions
    num_last_actions = env.unwrapped.num_last_actions
    last_actions = jnp.tile(hovering_action, (num_last_actions, 1))
    
    # 创建初始state（使用环境的TrackStateVer9类）
    from aquila.envs.target_trackVer9 import TrackStateVer9
    state = TrackStateVer9(
        time=0.0,
        step_idx=0,
        quadrotor_state=quadrotor_state,
        last_actions=last_actions,
        target_pos=target_pos,
        target_vel=target_vel,
        target_direction=target_direction,
        quad_params=quad_params,
        target_speed_max=1.0,  # 与训练时一致（train_trackVer9.py中使用1.0）
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
    
    # 验证x轴是否正对目标
    drone_x_axis = state.quadrotor_state.R[:, 0]
    to_target = state.target_pos - state.quadrotor_state.p
    to_target_normalized = to_target / (jnp.linalg.norm(to_target) + 1e-8)
    cos_angle = jnp.dot(drone_x_axis, to_target_normalized)
    angle_deg = float(jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0)) * 180.0 / jnp.pi)
    print(f"Initial angle between drone x-axis and target: {angle_deg:.3f}°")
    if angle_deg < 1.0:
        print("✅ Drone x-axis is aligned with target direction")
    else:
        print(f"⚠️  Warning: Drone x-axis is not perfectly aligned (angle: {angle_deg:.3f}°)")
    
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
    
    # 初始化缓冲区：所有位置都使用零观测初始化（和训练时保持一致）
    zero_obs = jnp.zeros_like(obs)  # 使用零向量代替实际观测
    action_obs_combined_zero = jnp.concatenate([hovering_action_normalized, zero_obs])
    action_obs_buffer = jnp.tile(action_obs_combined_zero[None, :], (buffer_size, 1))
    
    # 获取初始动作（和训练时一致）
    action_obs_buffer_flat = action_obs_buffer.reshape(-1)
    current_action = policy.apply(params, action_obs_buffer_flat)
    
    # 初始化动作计数器（和训练时一致：从0开始）
    action_counter = 0
    
    # ==================== Simulation Loop ====================
    # 根据测试总时间计算最大步数
    max_steps = int(test_duration / env.dt)
    print(f"Simulation will run for {test_duration:.1f} seconds ({max_steps} steps at dt={env.dt:.3f}s)\n")
    
    # 数据记录
    positions = []
    target_positions = []
    target_velocities = []  # 记录目标速度
    velocities = []
    accelerations = []
    actions_thrust = []
    actions_omega = []
    rewards = []
    distances = []
    times = []
    heights = []
    orientations = []  # 记录旋转矩阵
    angles_to_target = []  # 记录无人机x轴与目标物体的夹角
    
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
        
        # ==================== Update Target Position ====================
        # 使用运动模式更新目标位置和速度
        target_pos_np = np.array(state.target_pos)
        target_vel_np = np.array(state.target_vel)
        new_target_pos_np, new_target_vel_np = motion_pattern.update(
            target_pos_np, target_vel_np, env.dt
        )
        
        # 更新state中的目标位置和速度
        state = dataclasses.replace(
            state,
            target_pos=jnp.array(new_target_pos_np),
            target_vel=jnp.array(new_target_vel_np)
        )
        
        # 重新计算目标方向
        direction = state.target_pos - state.quadrotor_state.p
        direction_norm = jnp.linalg.norm(direction)
        if direction_norm > 1e-6:
            state = dataclasses.replace(state, target_direction=direction / direction_norm)
        
        # 记录数据
        positions.append(np.array(state.quadrotor_state.p))
        target_positions.append(np.array(state.target_pos))
        target_velocities.append(np.array(state.target_vel))  # 记录目标速度
        velocities.append(np.array(state.quadrotor_state.v))
        accelerations.append(np.array(state.quadrotor_state.acc))
        # 记录归一化的原始推力（-1到1）
        actions_thrust.append(float(current_action[0]))
        actions_omega.append(np.array(current_action[1:]))
        rewards.append(float(reward))
        distances.append(float(jnp.linalg.norm(state.quadrotor_state.p - state.target_pos)))
        times.append(float(state.time))
        heights.append(float(state.quadrotor_state.p[2]))
        
        # 记录旋转矩阵
        orientations.append(np.array(state.quadrotor_state.R))
        
        # 计算无人机x轴与目标物体的夹角
        # 无人机x轴方向（在全局坐标系中）：旋转矩阵的第一列
        drone_x_axis = state.quadrotor_state.R[:, 0]
        
        # 从无人机指向目标的方向向量
        to_target = state.target_pos - state.quadrotor_state.p
        to_target_norm = jnp.linalg.norm(to_target)
        
        if to_target_norm > 1e-6:
            to_target_normalized = to_target / to_target_norm
            # 计算夹角（使用点积）
            cos_angle = jnp.clip(jnp.dot(drone_x_axis, to_target_normalized), -1.0, 1.0)
            angle_rad = jnp.arccos(cos_angle)
            angle_deg = float(angle_rad * 180.0 / jnp.pi)
        else:
            angle_deg = 0.0
        
        angles_to_target.append(angle_deg)
        
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
    target_velocities = np.array(target_velocities)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)
    actions_thrust = np.array(actions_thrust)
    actions_omega = np.array(actions_omega)
    rewards = np.array(rewards)
    distances = np.array(distances)
    times = np.array(times)
    heights = np.array(heights)
    orientations = np.array(orientations)
    angles_to_target = np.array(angles_to_target)
    
    # ==================== Save Data ====================
    output_data = {
        'positions': positions,
        'target_positions': target_positions,
        'target_velocities': target_velocities,
        'velocities': velocities,
        'accelerations': accelerations,
        'actions_thrust': actions_thrust,
        'actions_omega': actions_omega,
        'rewards': rewards,
        'distances': distances,
        'times': times,
        'heights': heights,
        'orientations': orientations,
        'angles_to_target': angles_to_target,
        'env_config': env_config,
        'action_repeat': action_repeat,
        'buffer_size': buffer_size,
        'motion_pattern': motion_pattern.__class__.__name__,
    }
    
    os.makedirs('aquila/output', exist_ok=True)
    output_file = 'aquila/output/test_trackVer9_data.pkl'
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
    ax1.set_title(f'TrackVer9: Drone Tracking Trajectory - {motion_pattern.__class__.__name__}', 
                  fontsize=14, fontweight='bold')
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
    trajectory_file = 'aquila/output/trackVer9_trajectory_3d.png'
    plt.savefig(trajectory_file, dpi=150, bbox_inches='tight')
    print(f"✅ 3D trajectory plot saved to: {trajectory_file}")
    plt.close()
    
    # 图2: 数据分析图（动作、reward、高度、距离等）
    fig2, axes = plt.subplots(4, 2, figsize=(14, 16))
    
    # 子图1: 推力动作（归一化，-1到1）
    axes[0, 0].plot(times, actions_thrust, 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time [s]')
    axes[0, 0].set_ylabel('Normalized Thrust')
    axes[0, 0].set_title('Normalized Thrust Command over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.3, label='Max (1.0)')
    axes[0, 0].axhline(y=-1.0, color='r', linestyle='--', alpha=0.3, label='Min (-1.0)')
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.2)
    axes[0, 0].set_ylim([-1.2, 1.2])
    axes[0, 0].legend(fontsize=9)
    
    # 子图2: 角速度动作（归一化，-1到1）
    axes[0, 1].plot(times, actions_omega[:, 0], 'r-', linewidth=1.5, label='ωx (Roll rate)')
    axes[0, 1].plot(times, actions_omega[:, 1], 'g-', linewidth=1.5, label='ωy (Pitch rate)')
    axes[0, 1].plot(times, actions_omega[:, 2], 'b-', linewidth=1.5, label='ωz (Yaw rate)')
    axes[0, 1].set_xlabel('Time [s]')
    axes[0, 1].set_ylabel('Normalized Angular Velocity')
    axes[0, 1].set_title('Normalized Angular Velocity Commands over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=1.0, color='k', linestyle='--', alpha=0.2)
    axes[0, 1].axhline(y=-1.0, color='k', linestyle='--', alpha=0.2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.1)
    axes[0, 1].set_ylim([-1.2, 1.2])
    
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
    
    # 子图6: 速度模长（无人机和目标）
    velocity_norm = np.linalg.norm(velocities, axis=1)
    target_velocity_norm = np.linalg.norm(target_velocities, axis=1)
    axes[2, 1].plot(times, velocity_norm, 'teal', linewidth=1.5, label='Drone velocity')
    axes[2, 1].plot(times, target_velocity_norm, 'orange', linewidth=1.5, linestyle='--', label='Target velocity')
    axes[2, 1].set_xlabel('Time [s]')
    axes[2, 1].set_ylabel('Velocity [m/s]')
    axes[2, 1].set_title('Velocity Magnitude over Time')
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    
    # 子图7: 无人机x轴与目标物体的夹角
    axes[3, 0].plot(times, angles_to_target, 'crimson', linewidth=1.5)
    axes[3, 0].set_xlabel('Time [s]')
    axes[3, 0].set_ylabel('Angle [degrees]')
    axes[3, 0].set_title('Angle between Drone X-axis and Target Direction')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3, label='0° (aligned)')
    axes[3, 0].axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90° (perpendicular)')
    axes[3, 0].axhline(y=180, color='r', linestyle='--', alpha=0.3, label='180° (opposite)')
    axes[3, 0].legend(fontsize=9)
    axes[3, 0].set_ylim([0, 180])
    
    # 子图8: 加速度模长
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    axes[3, 1].plot(times, acceleration_norm, 'darkviolet', linewidth=1.5)
    axes[3, 1].set_xlabel('Time [s]')
    axes[3, 1].set_ylabel('Acceleration [m/s²]')
    axes[3, 1].set_title('Drone Acceleration Magnitude over Time')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].axhline(y=9.81, color='orange', linestyle='--', alpha=0.5, label='Gravity (9.81 m/s²)')
    axes[3, 1].legend(fontsize=9)
    
    plt.tight_layout()
    analysis_file = 'aquila/output/trackVer9_data_analysis.png'
    plt.savefig(analysis_file, dpi=150, bbox_inches='tight')
    print(f"✅ Data analysis plot saved to: {analysis_file}")
    plt.close()
    
    # ==================== Summary Statistics ====================
    print(f"\n{'='*60}")
    print(f"Test Summary")
    print(f"{'='*60}")
    print(f"Planned test duration: {test_duration:.2f}s")
    print(f"Total steps: {len(times)}")
    print(f"Actual time: {times[-1]:.2f}s")
    print(f"Final distance to target: {distances[-1]:.3f}m")
    print(f"Average distance to target: {np.mean(distances):.3f}m")
    print(f"Min distance to target: {np.min(distances):.3f}m")
    print(f"Average reward: {np.mean(rewards):.3f}")
    print(f"Total reward: {np.sum(rewards):.3f}")
    print(f"Final position: [{positions[-1, 0]:.3f}, {positions[-1, 1]:.3f}, {positions[-1, 2]:.3f}]")
    print(f"Target position: [{target_positions[-1, 0]:.3f}, {target_positions[-1, 1]:.3f}, {target_positions[-1, 2]:.3f}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # ==================== 选择目标运动模式 ====================
    
    # 模式1：圆周运动（水平面）- 默认，2秒加速到2m/s
    circular_motion_horizontal = CircularMotion(
        speed=1.0,
        radius=4.0,
        center=np.array([0.0, 0.0, -2.0]),
        plane_normal=np.array([0.0, 0.0, 1.0]),  # 水平面
        acceleration_time=2.0  # 2秒加速到目标速度
    )
    
    # 模式2：圆周运动（竖直平面，XZ平面）
    circular_motion_vertical_xz = CircularMotion(
        speed=0.5,
        radius=2.0,
        center=np.array([0.0, 0.0, -2.0]),
        plane_normal=np.array([0.0, 1.0, 0.0]),  # 竖直平面（XZ）
        acceleration_time=2.0  # 2秒加速到目标速度
    )
    
    # 模式3：圆周运动（竖直平面，YZ平面）
    circular_motion_vertical_yz = CircularMotion(
        speed=2.0,
        radius=2.0,
        center=np.array([0.0, 0.0, -2.0]),
        plane_normal=np.array([1.0, 0.0, 0.0]),  # 竖直平面（YZ）
        acceleration_time=2.0  # 2秒加速到目标速度
    )
    
    # 模式4：圆周运动（倾斜平面）
    circular_motion_tilted = CircularMotion(
        speed=2.0,
        radius=2.0,
        center=np.array([0.0, 0.0, -2.0]),
        plane_normal=np.array([1.0, 1.0, 1.0]),  # 倾斜平面
        acceleration_time=2.0  # 2秒加速到目标速度
    )
    
    # 模式5：静止目标
    static_target = StaticTarget(position=np.array([2.0, 0.0, -2.0]))
    
    # ==================== 运行测试 ====================
    # 选择要使用的运动模式
    # selected_motion = circular_motion_vertical_xz
    selected_motion = circular_motion_horizontal
    
    # 设置测试总时间（秒）
    test_duration = 25.0  # 默认10秒，可以根据需要修改
    
    print(f"\n{'='*60}")
    print(f"Selected Motion Pattern: {selected_motion.__class__.__name__}")
    if isinstance(selected_motion, CircularMotion):
        print(f"  Speed: {selected_motion.speed} m/s")
        print(f"  Radius: {selected_motion.radius} m")
        print(f"  Center: {selected_motion.center}")
        print(f"  Plane normal: {selected_motion.plane_normal}")
        print(f"  Acceleration time: {selected_motion.acceleration_time} s")
    print(f"Test duration: {test_duration} s")
    print(f"{'='*60}\n")
    
    test_policy(motion_pattern=selected_motion, test_duration=test_duration)

