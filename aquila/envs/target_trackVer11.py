"""
Track Environment Version 11
Full quadrotor tracking environment with a moving target and boundary constraints.
Ver11 modifications: Based on Ver10 with boundary constraints:
- Boundary: Rectangular box with configurable dimensions (center at origin)
- Boundary observation: Distance vectors to nearest points on 6 faces (in body frame)
- Boundary reward: Exponential penalty within 1m, constant max penalty outside
- Drone initialization: Randomized within boundary with diverse positions
- Target initialization: Randomized within boundary
- Target movement: Constrained to stay within boundary with diverse movement patterns

Uses NED (North-East-Down) coordinate system:
- X axis: North (positive forward)
- Y axis: East (positive right)  
- Z axis: Down (positive downward)
"""
import functools
from typing import Optional
import math

import chex
import jax
import jax.numpy as jnp
import numpy as np
import jax_dataclasses as jdc

from aquila.objects.quadrotor_obj import Quadrotor, QuadrotorState, QuadrotorParams
from aquila.objects.world_box_obj import WorldBox
from aquila.utils import spaces
from aquila.utils.pytrees import field_jnp
from aquila.utils.math import smooth_l1
import aquila.envs.env_base as env_base
from aquila.envs.env_base import EnvTransition
import dataclasses


@jdc.pytree_dataclass
class ExtendedQuadrotorParams(QuadrotorParams):
    """扩展的QuadrotorParams，包含mass、gravity和external_force以便支持动态扰动"""
    mass: float = 1.0  # [kg]
    gravity: float = 9.81  # [m/s^2]
    external_force: jax.Array = field_jnp([0.0, 0.0, 0.0])  # [N] 外部作用力（如风力）


@jdc.pytree_dataclass
class TrackStateVer11:
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    target_pos: jax.Array
    target_vel: jax.Array
    target_direction: jax.Array  # 目标速度方向（单位向量，用于随机方向运动）
    quad_params: ExtendedQuadrotorParams  # 添加quadrotor参数（扩展版，包含mass和gravity）
    target_speed_max: float = 1.0  # 当前episode的目标最大速度（每次reset随机化）
    action_raw: jax.Array = field_jnp(jnp.zeros(4))
    filtered_acc: jax.Array = field_jnp([0.0, 0.0, 9.8])
    filtered_thrust: float = field_jnp(9.8)
    # Flag to track if distance has ever exceeded reset_distance
    has_exceeded_distance: bool = False
    # Boundary dimensions (half-lengths in x and y, full height in z)
    # In NED: x, y centered at 0; z from -boundary_z (upward) to 0 (ground level)
    boundary_half_x: float = 3.0
    boundary_half_y: float = 4.0
    boundary_z: float = 10.0  # z范围: [-boundary_z, 0]（完全在z轴负方向，空中）


@jax.jit
def first_order_filter(current_value, last_value, alpha):
    """一阶惯性滤波器
    Args:
        current_value: 当前值
        last_value: 上一次的滤波值
        alpha: 滤波系数 (0-1), 越大表示对新值的权重越大
    """
    current_value = jnp.asarray(current_value, dtype=jnp.float32)
    last_value = jnp.asarray(last_value, dtype=jnp.float32)
    alpha = jnp.asarray(alpha, dtype=jnp.float32)
    return jnp.where(
        jnp.isfinite(last_value),
        alpha * current_value + (1 - alpha) * last_value,
        current_value
    )


def safe_norm(x, eps=1e-8):
    x = jnp.asarray(x, dtype=jnp.float32)
    return jnp.sqrt(jnp.sum(x * x) + eps)


class TrackEnvVer11(env_base.Env[TrackStateVer11]):
    """Quadrotor tracking environment Ver11 - with boundary constraints."""
    
    def __init__(
        self,
        *,
        max_steps_in_episode=1000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        drone_path=None,
        action_penalty_weight=0.1,
        # Tracking specific parameters
        target_height=2.0,  # m (height above ground, positive value)
        target_init_distance_min=0.5,  # m (x轴上的初始距离最小值)
        target_init_distance_max=1.5,  # m (x轴上的初始距离最大值)
        target_speed_max=1.0,  # m/s (目标最大速度)
        target_acceleration=0.5,  # m/s² (目标加速度，从0加速到最大速度)
        reset_distance=100.0,  # m (重置距离阈值)
        max_speed=20.0,  # m/s
        # Boundary parameters (full dimensions of the rectangular box)
        boundary_x=10.0,  # m (full length in x direction)
        boundary_y=10.0,  # m (full width in y direction)
        boundary_z=10.0,  # m (full height in z direction)
        boundary_penalty_distance=1.0,  # m (distance threshold for boundary penalty)
        boundary_penalty_max=100.0,  # maximum penalty for being outside boundary
        # Parameter randomization (quadrotor)
        thrust_to_weight_min=1.2,  # 最小推重比
        thrust_to_weight_max=5.0,  # 最大推重比
        disturbance_mag=2.0,  # [N] 常值扰动力大小（训练时>0，测试时=0）
    ):
        self.world_box = WorldBox(
            jnp.array([-5000.0, -5000.0, -5000.0]), jnp.array([5000.0, 5000.0, 5000.0])
        )
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        
        self.omega_std = omega_std
        
        # quadrotor - 使用完整的四旋翼模型（基于agilicious framework）
        self.quadrotor = Quadrotor(mass=1.0, disturbance_mag=disturbance_mag)
        
        # 获取四旋翼参数
        default_params = self.quadrotor.default_params()
        self.omega_min = -default_params.omega_max
        self.omega_max = default_params.omega_max
        self.thrust_min = self.quadrotor._thrust_min  # 完整模型的最小推力
        self.thrust_max = default_params.thrust_max
        
        # Set bounds based on max_speed parameter
        self.max_speed = max_speed
        self.v_min = jnp.array([-max_speed, -max_speed, -max_speed])
        self.v_max = jnp.array([max_speed, max_speed, max_speed])
        self.acc_min = jnp.array([-20.0, -20.0, -20.0])
        self.acc_max = jnp.array([20.0, 20.0, 20.0])

        assert delay >= 0.0, "Delay must be non-negative"
        self.delay = np.array(delay)
        self.num_last_actions = int(np.ceil(delay / dt)) + 1

        self.action_penalty_weight = action_penalty_weight

        # 计算悬停推力：mass * gravity
        thrust_hover = self.quadrotor._mass * 9.81
        self.hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
        
        # Tracking specific parameters
        self.target_height = target_height
        self.target_init_distance_min = target_init_distance_min
        self.target_init_distance_max = target_init_distance_max
        self.target_speed_max = target_speed_max
        self.target_acceleration = target_acceleration
        self.reset_distance = reset_distance
        
        # Boundary parameters
        # In NED coordinate system, the box is:
        # - x: [-boundary_half_x, boundary_half_x] (centered at 0)
        # - y: [-boundary_half_y, boundary_half_y] (centered at 0)
        # - z: [-boundary_z, 0] (completely above ground level, z-negative is upward)
        self.boundary_half_x = boundary_x / 2.0
        self.boundary_half_y = boundary_y / 2.0
        self.boundary_z = boundary_z  # 完整高度，从z=-boundary_z到z=0
        self.boundary_penalty_distance = boundary_penalty_distance
        self.boundary_penalty_max = boundary_penalty_max
        
        # Parameter randomization
        self.thrust_to_weight_min = thrust_to_weight_min
        self.thrust_to_weight_max = thrust_to_weight_max
        self.disturbance_mag = disturbance_mag

    def reset(
        self, key: chex.PRNGKey, state: Optional[TrackStateVer11] = None, quad_params: Optional[ExtendedQuadrotorParams] = None):
        """Reset environment with tracking-specific initialization.
        
        Args:
            key: Random key for initialization
            state: Optional state to reset to
            quad_params: Optional quadrotor parameters. If None, will use default or randomize based on key.
        """
        if state is not None:
            return state, self._get_obs(state)
        
        # 分割随机数key
        keys = jax.random.split(key, 15)
        key_target_pos, key_target_dir, key_target_speed, key_drone_pos, key_roll, key_pitch, key_yaw, key_omega, key_quad, key_randomize, key_vel_dir, key_vel_mag, key_target_change_dir, key_drone_x, key_drone_y = keys
        
        # 获取quadrotor参数（如果没有提供则使用默认参数）
        if quad_params is None:
            # # 暂时禁用参数随机化以加快训练速度
            # # 使用默认参数
            # quad_params = self.quadrotor.default_params()
            
            # 启用参数随机化（质量固定，推力和角速度随机化）
            base_params = self.quadrotor.default_params()
            randomized_params = Quadrotor.randomize_params(
                base_params,
                self.quadrotor._mass,  # Quadrotor.randomize_params 需要 mass 参数
                key_randomize,
                thrust_to_weight_min=self.thrust_to_weight_min,
                thrust_to_weight_max=self.thrust_to_weight_max
            )
            # 转换为扩展的参数类，添加mass和gravity
            quad_params = ExtendedQuadrotorParams(
                thrust_max=randomized_params.thrust_max,
                omega_max=randomized_params.omega_max,
                motor_tau=randomized_params.motor_tau,
                mass=self.quadrotor._mass,
                gravity=9.81
            )
        
        # ========== 目标物体初始化 ==========
        # 在边界内随机初始化目标物体位置（留有0.5m的安全边距）
        # NED坐标系：z范围为[-boundary_z, 0]，完全在地面以上
        target_pos_keys = jax.random.split(key_target_pos, 3)
        target_x = jax.random.uniform(
            target_pos_keys[0], shape=(), 
            minval=-self.boundary_half_x + 0.5, 
            maxval=self.boundary_half_x - 0.5
        )
        target_y = jax.random.uniform(
            target_pos_keys[1], shape=(),
            minval=-self.boundary_half_y + 0.5,
            maxval=self.boundary_half_y - 0.5
        )
        # z在边界内随机，偏向靠近地面（较小的z值）
        # z范围: [0.5, boundary_z*0.5] 偏向上半部分（z值较小）
        target_z = jax.random.uniform(
            target_pos_keys[2], shape=(),
            minval=-self.boundary_z * 0.5,
            maxval=-0.5
        )
        target_pos = jnp.array([target_x, target_y, target_z])
        
        # 随机生成速度方向（单位向量，任意方向）
        # 使用球面均匀分布生成随机单位向量
        # 生成3个独立的标准正态分布随机数，然后归一化
        random_vec = jax.random.normal(key_target_dir, shape=(3,))
        target_direction = random_vec / jnp.linalg.norm(random_vec)
        
        # 每次episode随机化目标最大速度（0到1m/s之间）
        episode_target_speed_max = jax.random.uniform(
            key_target_speed, shape=(),
            minval=0.0,
            maxval=1.0
        )
        
        # 初始速度为0，将加速到episode的目标最大速度（沿随机方向）
        target_vel = jnp.array([0.0, 0.0, 0.0])
        
        # ========== 无人机初始化 ==========
        # 在目标物体周围2m范围内随机初始化无人机位置（同时保证在边界内）
        # NED坐标系：z范围为[-boundary_z, 0]，完全在地面以上
        drone_pos_keys = jax.random.split(key_drone_pos, 3)
        
        # 在目标周围球形区域内随机采样（半径0.5-2.0m）
        # 使用球坐标系生成均匀分布
        radius = jax.random.uniform(
            drone_pos_keys[0], shape=(),
            minval=0.5,  # 最小距离0.5m，避免太近
            maxval=2.0   # 最大距离2.0m
        )
        theta = jax.random.uniform(drone_pos_keys[1], shape=(), minval=0.0, maxval=2.0 * jnp.pi)
        phi = jnp.arccos(jax.random.uniform(drone_pos_keys[2], shape=(), minval=-1.0, maxval=1.0))
        
        # 转换为笛卡尔坐标偏移
        offset_x = radius * jnp.sin(phi) * jnp.cos(theta)
        offset_y = radius * jnp.sin(phi) * jnp.sin(theta)
        offset_z = radius * jnp.cos(phi)
        
        # 无人机位置 = 目标位置 + 偏移
        drone_x = target_x + offset_x
        drone_y = target_y + offset_y
        drone_z = target_z + offset_z
        
        # 确保无人机位置在边界内（带1.0m安全边距）
        drone_x = jnp.clip(drone_x, -self.boundary_half_x + 1.0, self.boundary_half_x - 1.0)
        drone_y = jnp.clip(drone_y, -self.boundary_half_y + 1.0, self.boundary_half_y - 1.0)
        drone_z = jnp.clip(drone_z, -self.boundary_z + 1.0, -1.0)
        
        p = jnp.array([drone_x, drone_y, drone_z])
        
        # 速度：随机方向，大小在0~0.5m/s范围内随机
        vel_keys = jax.random.split(key_vel_dir, 2)
        vel_theta = jax.random.uniform(vel_keys[0], shape=(), minval=0.0, maxval=2.0 * jnp.pi)
        vel_phi = jnp.arccos(jax.random.uniform(vel_keys[1], shape=(), minval=-1.0, maxval=1.0))
        
        vel_direction = jnp.array([
            jnp.sin(vel_phi) * jnp.cos(vel_theta),
            jnp.sin(vel_phi) * jnp.sin(vel_theta),
            jnp.cos(vel_phi)
        ])
        vel_magnitude = jax.random.uniform(key_vel_mag, shape=(), minval=0.0, maxval=0.5)
        v = vel_magnitude * vel_direction
        
        # roll和pitch在±30°范围内随机，yaw完全随机
        max_tilt_angle = jnp.pi / 6  # ±30°
        init_roll = jax.random.uniform(key_roll, shape=(), minval=-max_tilt_angle, maxval=max_tilt_angle)
        init_pitch = jax.random.uniform(key_pitch, shape=(), minval=-max_tilt_angle, maxval=max_tilt_angle)
        init_yaw = jax.random.uniform(key_yaw, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        
        # 将欧拉角转换为旋转矩阵
        c1, s1 = jnp.cos(init_roll), jnp.sin(init_roll)
        c2, s2 = jnp.cos(init_pitch), jnp.sin(init_pitch)
        c3, s3 = jnp.cos(init_yaw), jnp.sin(init_yaw)

        R_x = jnp.array([[1.0, 0.0, 0.0],
                        [0.0, c1, -s1],
                        [0.0, s1, c1]])
        
        R_y = jnp.array([[c2, 0.0, s2],
                        [0.0, 1.0, 0.0],
                        [-s2, 0.0, c2]])
        
        R_z = jnp.array([[c3, -s3, 0.0],
                        [s3, c3, 0.0],
                        [0.0, 0.0, 1.0]])

        R = R_z @ R_y @ R_x
        
        # 随机角速度
        omega = jax.random.normal(key_omega, (3,)) * self.omega_std * 0

        # Initialize quadrotor state
        # Quadrotor.create_state 使用位置参数 p, R, v，其他参数通过 kwargs 传递
        quadrotor_state = self.quadrotor.create_state(p, R, v, omega=omega, dr_key=key_quad)
        
        # Calculate hovering action based on current episode's quad_params
        # 悬停推力 = mass * gravity（使用当前episode的实际质量）
        thrust_hover = self.quadrotor._mass * 9.81  # QuadrotorParams 不包含 mass 和 gravity，使用实例的 _mass
        hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
        
        # Initialize action history
        last_actions = jax.device_put(jnp.tile(hovering_action, (self.num_last_actions, 1)))
        action_raw = jax.device_put(jnp.zeros(4))
        filtered_acc = jax.device_put(jnp.array([0.0, 0.0, 9.81]))  # NED坐标系，Down为正
        filtered_thrust = jax.device_put(jnp.array(thrust_hover))

        state = TrackStateVer11(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            target_pos=target_pos,
            target_vel=target_vel,
            target_direction=target_direction,
            quad_params=quad_params,
            target_speed_max=episode_target_speed_max,
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            has_exceeded_distance=False,
            boundary_half_x=self.boundary_half_x,
            boundary_half_y=self.boundary_half_y,
            boundary_z=self.boundary_z,
        )
        
        return state, self._get_obs(state)

    def _compute_boundary_distances(self, state: TrackStateVer11) -> jax.Array:
        """计算无人机到边界六个面最近点的距离向量（机体系下）
        
        边界定义（NED坐标系）：
        - X: [-boundary_half_x, boundary_half_x]
        - Y: [-boundary_half_y, boundary_half_y]
        - Z: [-boundary_z, 0]（完全在地面以上）
        
        Args:
            state: 当前状态
            
        Returns:
            距离向量数组 (6x3)，每行代表一个面的最近点的距离向量（机体系）
            顺序：+X面, -X面, +Y面, -Y面, +Z面（地面，z=0）, -Z面（顶部，z=-boundary_z）
        """
        quad_pos = state.quadrotor_state.p
        quad_R = state.quadrotor_state.R
        R_transpose = jnp.transpose(quad_R)
        
        # 计算到六个面的最近点（世界系）
        # +X面 (x = boundary_half_x)
        closest_point_px = jnp.array([
            state.boundary_half_x,
            jnp.clip(quad_pos[1], -state.boundary_half_y, state.boundary_half_y),
            jnp.clip(quad_pos[2], -state.boundary_z, 0.0)
        ])
        
        # -X面 (x = -boundary_half_x)
        closest_point_nx = jnp.array([
            -state.boundary_half_x,
            jnp.clip(quad_pos[1], -state.boundary_half_y, state.boundary_half_y),
            jnp.clip(quad_pos[2], -state.boundary_z, 0.0)
        ])
        
        # +Y面 (y = boundary_half_y)
        closest_point_py = jnp.array([
            jnp.clip(quad_pos[0], -state.boundary_half_x, state.boundary_half_x),
            state.boundary_half_y,
            jnp.clip(quad_pos[2], -state.boundary_z, 0.0)
        ])
        
        # -Y面 (y = -boundary_half_y)
        closest_point_ny = jnp.array([
            jnp.clip(quad_pos[0], -state.boundary_half_x, state.boundary_half_x),
            -state.boundary_half_y,
            jnp.clip(quad_pos[2], -state.boundary_z, 0.0)
        ])
        
        # +Z面 (z = 0, 地面)
        closest_point_pz = jnp.array([
            jnp.clip(quad_pos[0], -state.boundary_half_x, state.boundary_half_x),
            jnp.clip(quad_pos[1], -state.boundary_half_y, state.boundary_half_y),
            0.0
        ])
        
        # -Z面 (z = -boundary_z, 顶部，高空)
        closest_point_nz = jnp.array([
            jnp.clip(quad_pos[0], -state.boundary_half_x, state.boundary_half_x),
            jnp.clip(quad_pos[1], -state.boundary_half_y, state.boundary_half_y),
            -state.boundary_z
        ])
        
        # 计算距离向量（世界系）
        dist_vec_px = closest_point_px - quad_pos
        dist_vec_nx = closest_point_nx - quad_pos
        dist_vec_py = closest_point_py - quad_pos
        dist_vec_ny = closest_point_ny - quad_pos
        dist_vec_pz = closest_point_pz - quad_pos
        dist_vec_nz = closest_point_nz - quad_pos
        
        # 转换到机体系
        dist_vec_px_body = R_transpose @ dist_vec_px
        dist_vec_nx_body = R_transpose @ dist_vec_nx
        dist_vec_py_body = R_transpose @ dist_vec_py
        dist_vec_ny_body = R_transpose @ dist_vec_ny
        dist_vec_pz_body = R_transpose @ dist_vec_pz
        dist_vec_nz_body = R_transpose @ dist_vec_nz
        
        # 组合成 (6, 3) 数组
        boundary_distances = jnp.stack([
            dist_vec_px_body,
            dist_vec_nx_body,
            dist_vec_py_body,
            dist_vec_ny_body,
            dist_vec_pz_body,
            dist_vec_nz_body
        ])
        
        return boundary_distances

    def _get_obs(self, state: TrackStateVer11) -> jax.Array:
        """Get observation from state.
        
        Ver11修改：在Ver10基础上添加边界距离向量观测
        
        观测组成：
        1. 无人机机体系自身速度向量 (3)
        2. 无人机机体系重力方向 (3)
        3. 无人机机体系目标物体坐标 (3)
        4. 无人机到边界六个面最近点的距离向量（机体系） (18 = 6x3)
        """
        # 直接使用真实状态（无延迟）
        quad_pos = state.quadrotor_state.p
        quad_vel = state.quadrotor_state.v
        quad_R = state.quadrotor_state.R
        R_transpose = jnp.transpose(quad_R)
        
        # 1. 无人机机体系自身速度向量
        v_body = R_transpose @ quad_vel
        
        # 2. 无人机机体系重力方向
        g_world = jnp.array([0.0, 0.0, 1.0])  # NED坐标系中重力方向 (Down为正)
        g_body = R_transpose @ g_world
        
        # 3. 无人机机体系目标物体坐标（相对位置）
        target_pos_world = state.target_pos
        target_pos_relative_world = target_pos_world - quad_pos
        target_pos_body = R_transpose @ target_pos_relative_world
        
        # 4. 无人机到边界六个面最近点的距离向量（机体系）
        boundary_distances = self._compute_boundary_distances(state)  # (6, 3)
        boundary_distances_flat = boundary_distances.flatten()  # (18,)

        # Combine all observations
        components = [
            v_body,                                # 机体系速度 (3)
            g_body,                                # 机体系重力方向 (3)
            target_pos_body,                       # 机体系目标位置 (3)
            boundary_distances_flat,               # 边界距离向量 (18)
        ]  
        obs = jnp.concatenate(components)
        return obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: TrackStateVer11, action: jax.Array, key: chex.PRNGKey
    ) -> EnvTransition:
        # 保存原始action (tanh输出为[-1,1]范围)
        action_raw = action
        
        # 将tanh输出的action [-1, 1] 映射到实际范围
        # thrust: [-1, 1] -> [thrust_min*4, thrust_max*4]（保持从0开始映射）
        # omega: [-1, 1] -> [-omega_max, omega_max]（对称映射，0对应静止）
        thrust_normalized = action[0]
        omega_normalized = action[1:]
        
        # Thrust映射：[-1, 1] -> [thrust_min*4, thrust_max*4]
        # ⚠️  使用当前状态的实际thrust_max（参数随机化后的值）
        # tanh输出-1 -> thrust_min, 0 -> 中间值, 1 -> thrust_max
        actual_thrust_max = state.quad_params.thrust_max
        thrust_denormalized = 0.5 * (thrust_normalized + 1.0) * (actual_thrust_max * 4 - self.thrust_min * 4) + self.thrust_min * 4
        
        # Omega映射：[-1, 1] -> [-omega_max, omega_max]（对称映射）
        # ⚠️  使用当前状态的实际omega_max（参数随机化后的值）
        # tanh输出-1 -> -omega_max, 0 -> 0(静止), 1 -> omega_max
        actual_omega_max = state.quad_params.omega_max
        omega_denormalized = omega_normalized * actual_omega_max
        
        action = jnp.concatenate([jnp.array([thrust_denormalized]), omega_denormalized])
        
        # clip action to physical limits (使用实际参数)
        action_low = jnp.concatenate([jnp.array([self.thrust_min * 4]), -actual_omega_max])
        action_high = jnp.concatenate([jnp.array([actual_thrust_max * 4]), actual_omega_max])
        action = jnp.clip(action, action_low, action_high)

        # add action to last actions
        last_actions = jnp.roll(state.last_actions, shift=-1, axis=0)
        last_actions = last_actions.at[-1].set(action)

        # 1 step
        dt_1 = self.delay % self.dt
        action_1 = last_actions[0]
        f_1, omega_1 = action_1[0], action_1[1:]
        # 直接传递ExtendedQuadrotorParams（包含mass、gravity、external_force）
        # Quadrotor的_dynamics方法现在支持ExtendedQuadrotorParams
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, dt_1, 
            drag_params=None,  # 使用默认drag_params
            quad_params=state.quad_params  # 使用ExtendedQuadrotorParams
        )

        if self.delay > 0:
            # 2 step
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, f_2, omega_2, dt_2,
                drag_params=None,  # 使用默认drag_params
                quad_params=state.quad_params  # 使用ExtendedQuadrotorParams
            )

        # 更新滤波值
        alpha_acc = jnp.array(0.05, dtype=jnp.float32)
        alpha_thrust = jnp.array(0.05, dtype=jnp.float32)
        
        # 计算比力加速度 (specific force in body frame)
        gravity_world = jnp.array([0., 0., 9.81])
        R = quadrotor_state.R
        R_transpose = jnp.transpose(R)
        specific_force_world = quadrotor_state.acc - gravity_world
        specific_force_world = jnp.clip(specific_force_world, -100.0, 100.0)
        specific_force = jnp.matmul(R_transpose, specific_force_world)

        # 使用比力加速度进行滤波
        filtered_acc = first_order_filter(specific_force, state.filtered_acc, alpha_acc)
        filtered_thrust = first_order_filter(action_1[0], state.filtered_thrust, alpha_thrust)

        # 分割key用于目标物体运动
        key_target_motion, key_direction_change = jax.random.split(key, 2)
        
        # 目标物体运动（保持在边界内，带有随机方向变化）
        current_speed_vec = state.target_vel
        current_speed = safe_norm(current_speed_vec, eps=1e-8)
        episode_target_speed_max = state.target_speed_max  # 使用当前episode的目标最大速度
        target_acc = self.target_acceleration
        
        # 如果当前速度小于最大速度，则加速
        new_speed = jnp.minimum(
            current_speed + target_acc * self.dt,
            episode_target_speed_max
        )
        
        # 随机改变方向（小概率事件，约1%每步）
        should_change_direction = jax.random.uniform(key_direction_change) < 0.01
        
        # 如果需要改变方向，生成新的随机方向
        random_vec = jax.random.normal(key_target_motion, shape=(3,))
        new_direction = random_vec / safe_norm(random_vec, eps=1e-8)
        
        # 使用条件判断是否更新方向
        target_direction = jnp.where(
            should_change_direction,
            new_direction,
            state.target_direction
        )
        
        # 计算新的速度向量
        target_vel = new_speed * target_direction
        
        # 预测下一步位置
        predicted_pos = state.target_pos + target_vel * self.dt
        
        # 检查是否会超出边界，如果超出则反弹（反转相应方向的速度分量）
        # X方向检查
        out_of_bounds_px = predicted_pos[0] > state.boundary_half_x
        out_of_bounds_nx = predicted_pos[0] < -state.boundary_half_x
        reflect_x = out_of_bounds_px | out_of_bounds_nx
        
        # Y方向检查
        out_of_bounds_py = predicted_pos[1] > state.boundary_half_y
        out_of_bounds_ny = predicted_pos[1] < -state.boundary_half_y
        reflect_y = out_of_bounds_py | out_of_bounds_ny
        
        # Z方向检查（边界为[-boundary_z, 0]）
        out_of_bounds_pz = predicted_pos[2] > 0.0  # 超出地面（向下）
        out_of_bounds_nz = predicted_pos[2] < -state.boundary_z  # 超出顶部（向上）
        reflect_z = out_of_bounds_pz | out_of_bounds_nz
        
        # 反转相应方向的速度分量
        target_vel_x = jnp.where(reflect_x, -target_vel[0], target_vel[0])
        target_vel_y = jnp.where(reflect_y, -target_vel[1], target_vel[1])
        target_vel_z = jnp.where(reflect_z, -target_vel[2], target_vel[2])
        target_vel = jnp.array([target_vel_x, target_vel_y, target_vel_z])
        
        # 同时更新方向向量（反转后的方向）
        target_direction_x = jnp.where(reflect_x, -target_direction[0], target_direction[0])
        target_direction_y = jnp.where(reflect_y, -target_direction[1], target_direction[1])
        target_direction_z = jnp.where(reflect_z, -target_direction[2], target_direction[2])
        target_direction = jnp.array([target_direction_x, target_direction_y, target_direction_z])
        # 重新归一化方向向量
        target_direction = target_direction / safe_norm(target_direction, eps=1e-8)
        
        # 计算最终位置，并限制在边界内
        target_pos = state.target_pos + target_vel * self.dt
        target_pos = jnp.clip(
            target_pos,
            jnp.array([-state.boundary_half_x, -state.boundary_half_y, -state.boundary_z]),
            jnp.array([state.boundary_half_x, state.boundary_half_y, 0.0])
        )
        
        # 检查距离是否超过10m，更新标志位
        distance_to_target = safe_norm(quadrotor_state.p - target_pos, eps=1e-8)
        has_exceeded_distance = state.has_exceeded_distance | (distance_to_target > self.reset_distance)
        
        next_state = dataclasses.replace(
            state,
            time=state.time + self.dt,
            step_idx=state.step_idx + 1,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            quad_params=state.quad_params,  # 保持quad_params不变
            target_speed_max=state.target_speed_max,  # 保持target_speed_max不变
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            target_pos=target_pos,
            target_vel=target_vel,
            target_direction=target_direction,  # 使用更新后的方向（包含反弹后的方向）
            has_exceeded_distance=has_exceeded_distance,
        )

        obs = self._get_obs(next_state)
        reward = self._compute_reward(state, next_state)
        
        # 检查是否需要重置（距离大于10m）
        distance_to_target = safe_norm(next_state.quadrotor_state.p - next_state.target_pos, eps=1e-8)
        terminated = distance_to_target > self.reset_distance
        
        truncated = jnp.greater_equal(
            next_state.step_idx, self.max_steps_in_episode
        )
        
        info = {
            "quad_p": next_state.quadrotor_state.p,
            "quad_v": next_state.quadrotor_state.v,
            "quad_acc": next_state.quadrotor_state.acc,
            "quad_R": next_state.quadrotor_state.R,
            "target_p": next_state.target_pos,
            "target_v": next_state.target_vel,
            "action": next_state.last_actions[-1],
            "distance_to_target": distance_to_target,
        }

        return EnvTransition(
            next_state, obs, reward, terminated, truncated, info
        )

    def _compute_reward(
        self, last_state: TrackStateVer11, next_state: TrackStateVer11
    ) -> jax.Array:
        """计算奖励 - 基于 agile_lossVer7 算法，加入推力惩罚和边界惩罚
        奖励设计：
        1. 方向损失：使用余弦相似度计算完整3D方向
        2. 距离损失：水平距离与目标距离的绝对差值
        3. 高度损失：无人机高度与目标高度的绝对差值
        4. 速度损失：相对速度模长
        5. 姿态损失：基于机体z轴方向的惩罚
        6. 动作损失：当前动作与上一动作的L2范数
        7. 角速度损失：惩罚旋转运动（防止roll持续旋转）
        8. 推力超限损失：动作推力与悬停推力的偏差（Ver10新增）
        9. 边界损失：距离边界的惩罚（Ver11新增）
        """
        # 获取状态信息
        quad_pos = next_state.quadrotor_state.p
        quad_vel = next_state.quadrotor_state.v
        quad_R = next_state.quadrotor_state.R
        quad_omega = next_state.quadrotor_state.omega  # 获取角速度用于惩罚旋转
        target_pos = next_state.target_pos
        target_vel = next_state.target_vel
        
        # 计算相对位置和速度
        p_rel = target_pos - quad_pos
        v_rel = target_vel - quad_vel
        
        # 1. 方向损失 (direction) - 使用余弦相似度计算完整3D方向
        # 将相对位置向量转换到机体坐标系
        R_transpose = jnp.transpose(quad_R)
        direction_vector_body = R_transpose @ p_rel
        direction_vector_body_unit = direction_vector_body / (safe_norm(direction_vector_body, eps=1e-6))
        
        init_vec = jnp.array([1.0, 0.0, 0.0])  # 机体前向方向（完整3D）
        cos_similarity = jnp.dot(init_vec, direction_vector_body_unit)
        cos_similarity = jnp.clip(cos_similarity, -1.0, 1.0)
        
        # 零惩罚范围：方向 < 15° 时损失为0
        # cos(15°) ≈ 0.966
        cos_threshold = jnp.cos(jnp.deg2rad(15.0))
        # 🔧 FIX: exp input is in [0, 2] since cos_similarity in [-1, 1], which is safe
        # but clip for consistency and numerical stability
        exp_input = jnp.clip(1 - cos_similarity, 0.0, 2.0)
        direction_loss_base = jnp.exp(exp_input) - 1
        # 计算阈值处的损失值，用于保持连续性
        threshold_loss = jnp.exp(1 - cos_threshold) - 1
        # 在阈值内损失为0，超出后从阈值处开始线性增加
        direction_loss = jnp.where(
            cos_similarity >= cos_threshold,
            0.0,
            direction_loss_base - threshold_loss  # 减去阈值处的损失，使在阈值处连续
        )
        
        
        # 2. 距离损失 (distance) - 水平距离与目标距离的绝对差值
        norm_hor_dis = safe_norm(p_rel[:2], eps=1e-8)
        target_distance = 1.0  # 目标距离1米
        distance_error = jnp.abs(norm_hor_dis - target_distance)
        # 零惩罚范围：位置 < 30cm 时损失为0
        position_threshold = 0.3  # 30cm
        distance_loss = jnp.where(
            distance_error < position_threshold,
            0.0,
            distance_error - position_threshold  # 超出后从0开始线性增加
        )
        
        # 3. 高度损失 (h) - 无人机高度与目标高度的绝对差值
        height_error = jnp.abs(quad_pos[2] - target_pos[2])
        # 零惩罚范围：位置 < 30cm 时损失为0
        height_loss = jnp.where(
            height_error < position_threshold,
            0.0,
            height_error - position_threshold  # 超出后从0开始线性增加
        )
        
        # 4. 速度损失 (vel) - 相对速度模长
        velocity_error = safe_norm(v_rel, eps=1e-8)
        # 零惩罚范围：速度 < 0.3m/s 时损失为0
        velocity_threshold = 0.3  # 0.3m/s
        velocity_loss = jnp.where(
            velocity_error < velocity_threshold,
            0.0,
            velocity_error - velocity_threshold  # 超出后从0开始线性增加
        )
        
        # 5. 姿态损失 (ori) - 基于机体z轴方向的惩罚，改为指数增长
        body_z_world = quad_R @ jnp.array([0.0, 0.0, -1.0])  # 机体z轴在世界系中的方向
        # 理想情况下，机体z轴应该指向上方（-z方向），body_z_world应该接近[0, 0, -1]
        # 惩罚当body_z_world[2]偏离-1的情况（即偏离垂直）
        # 使用指数增长：exp(偏离度) - 1
        ori_deviation = (body_z_world[2] + 1.0) ** 2  # 偏离度（0到4之间）
        # 🔧 FIX: clip ori_deviation for safety, though it should be in [0, 4] naturally
        ori_deviation_clipped = jnp.clip(ori_deviation, 0.0, 4.0)
        ori_loss = 10 * (jnp.exp(ori_deviation_clipped) - 1.0)  # 指数增长
        
        # 6. 动作损失 (aux) - 当前动作与上一动作的L2范数，改为指数增长
        action_current = next_state.action_raw
        action_last = jnp.where(
            last_state.step_idx == 0,
            next_state.action_raw,  # step 0: 使用当前动作，变化为0
            last_state.action_raw   # step > 0: 使用真实的上一个动作
        )
        action_change = action_current - action_last
        action_error = safe_norm(action_change, eps=1e-8)
        # 🔧 FIX: clip action_error to avoid exp overflow
        # action_error could be large if network outputs change drastically
        action_error_clipped = jnp.clip(action_error, 0.0, 10.0)
        action_loss = jnp.exp(action_error_clipped) - 1.0  # 指数增长
        
        # 7. 角速度损失 - 防止持续旋转（只惩罚roll和pitch，不惩罚yaw），改为指数增长
        omega_roll_pitch = quad_omega[:2]  # 只取roll和pitch角速度，忽略yaw
        omega_error = safe_norm(omega_roll_pitch, eps=1e-8)
        # 🔧 FIX: clip omega_error to avoid exp overflow
        # omega_max can be large (e.g., 50 rad/s), so L2 norm could be ~70
        omega_error_clipped = jnp.clip(omega_error, 0.0, 10.0)
        omega_loss = jnp.exp(omega_error_clipped) - 1.0  # 指数增长
        
        # 8. 推力超限损失 - 约束推力，动作推力与悬停推力的偏差（Ver10新增，参考hoverVer1）
        # 使用当前动作（归一化后的值），需要去归一化
        thrust_normalized = action_current[0]
        # 去归一化推力：[-1, 1] -> [thrust_min*4, thrust_max*4]
        actual_thrust_max = next_state.quad_params.thrust_max
        action_thrust = 0.5 * (thrust_normalized + 1.0) * (actual_thrust_max * 4 - self.thrust_min * 4) + self.thrust_min * 4
        # 计算悬停推力：mass * gravity
        thrust_hover = next_state.quad_params.mass * next_state.quad_params.gravity
        # 计算推力偏差的L2范数（对于标量，L2范数就是绝对差值）
        thrust_error = action_thrust - thrust_hover
        thrust_loss = safe_norm(jnp.array([thrust_error]), eps=1e-8)
        
        # 9. 边界损失 - 距离边界的惩罚（Ver11新增）
        # 计算到六个面的最小距离
        # 边界定义（NED坐标系）：X: [-half_x, +half_x], Y: [-half_y, +half_y], Z: [-boundary_z, 0]
        quad_pos = next_state.quadrotor_state.p
        
        # 计算到各个面的距离（带符号，正值表示在边界内，负值表示超出边界）
        dist_to_px = next_state.boundary_half_x - quad_pos[0]  # 距离+X面
        dist_to_nx = quad_pos[0] + next_state.boundary_half_x  # 距离-X面
        dist_to_py = next_state.boundary_half_y - quad_pos[1]  # 距离+Y面
        dist_to_ny = quad_pos[1] + next_state.boundary_half_y  # 距离-Y面
        dist_to_pz = 0.0 - quad_pos[2]  # 距离+Z面（地面，z=0）
        dist_to_nz = quad_pos[2] + next_state.boundary_z  # 距离-Z面（顶部，z=-boundary_z）
        
        # 找到最近的面的距离
        min_distance = jnp.minimum(
            jnp.minimum(dist_to_px, dist_to_nx),
            jnp.minimum(
                jnp.minimum(dist_to_py, dist_to_ny),
                jnp.minimum(dist_to_pz, dist_to_nz)
            )
        )
        
        # 边界惩罚计算
        # 情况1：距离面 > 1m，不受惩罚
        # 情况2：距离面 <= 1m 且在边界内，指数增加的惩罚
        # 情况3：超出边界（距离 < 0），惩罚恒定为最大值
        boundary_threshold = self.boundary_penalty_distance  # 1.0m
        
        # 计算惩罚
        # 在边界内且距离 <= 1m 时：exp((1 - distance) * 3) - 1（指数增长）
        # 超出边界时：使用最大惩罚值
        inside_penalty_zone = (min_distance <= boundary_threshold) & (min_distance >= 0.0)
        outside_boundary = min_distance < 0.0
        
        # 指数惩罚：当距离从1m减少到0m时，惩罚从0增加到exp(3)-1≈19.09
        # 🔧 FIX: clip min_distance to avoid exp overflow when drone is far outside boundary
        # If min_distance < -10, exp((1-(-10))*3) = exp(33) would overflow
        safe_min_distance = jnp.clip(min_distance, -10.0, boundary_threshold)
        exp_penalty = jnp.exp((boundary_threshold - safe_min_distance) * 3.0) - 1.0
        
        boundary_loss = jnp.where(
            outside_boundary,
            self.boundary_penalty_max,  # 超出边界：恒定最大惩罚
            jnp.where(
                inside_penalty_zone,
                exp_penalty,  # 在惩罚区内：指数增长
                0.0  # 距离面 > 1m：无惩罚
            )
        )
        
        # 总损失 - 根据新的损失函数特性调整权重
        # 权重调整说明：
        # - 方向损失：有零惩罚范围(<15°)，超出后指数增长，权重降低到40
        # - 位置损失（距离和高度）：有零惩罚范围(<30cm)，超出后线性增长，保持较高权重80
        # - 速度损失：有零惩罚范围(<0.3m/s)，超出后线性增长，权重提高到3
        # - 姿态损失：改为指数增长，权重降低到0.5（指数增长本身会快速增加）
        # - 动作损失：改为指数增长，权重降低到4
        # - 角速度损失：改为指数增长，权重降低到4
        # - 推力超限损失：中等权重，约束推力接近悬停推力（Ver10新增）
        # - 边界损失：高权重，强制无人机远离边界（Ver11新增）
        total_loss = (
            0.5 * ori_loss +           # 姿态损失：指数增长，权重降低
            150 * distance_loss +        # 距离损失：零惩罚范围后线性增长，保持较高权重
            3 * velocity_loss +         # 速度损失：零惩罚范围后线性增长，权重提高
            40 * direction_loss +       # 方向损失：零惩罚范围后指数增长，权重稍微降低
            80 * height_loss +          # 高度损失：零惩罚范围后线性增长，保持较高权重
            4 * action_loss +           # 动作损失：指数增长，权重降低
            10 * omega_loss +            # 角速度损失：指数增长，权重降低
            4 * thrust_loss +           # 推力超限损失：中等权重，约束推力接近悬停推力（Ver10新增）
            100 * boundary_loss          # 边界损失：高权重，强制无人机远离边界（Ver11新增）
        )
        
        # 转换为奖励（负的损失）
        reward = -total_loss
        
        return reward

    def _compute_action_cost(self, action: jax.Array) -> jax.Array:
        """计算动作超限的惩罚
        Args:
            action: 动作数组 [thrust, wx, wy, wz]
        Returns:
            cost: 惩罚值
        """
        # 偏离悬停动作的惩罚
        omega_dev = action[1:] - self.hovering_action[1:]
        action_bias_cost = smooth_l1(safe_norm(omega_dev * jnp.array([1.0, 1.0, 2.0])))
        
        return action_bias_cost

    @property
    def action_space(self) -> spaces.Box:
        # Action space is now normalized to [-1, 1] for all dimensions
        # to match the tanh output from the neural network
        low = -jnp.ones(4)
        high = jnp.ones(4)
        return spaces.Box(low, high, shape=(4,))

    @property
    def observation_space(self) -> spaces.Box:
        """Get observation space.
        
        Ver11修改：在Ver10基础上添加边界距离向量观测
        
        观测组成：
        1. 机体系速度 (3)
        2. 机体系重力方向 (3)
        3. 机体系目标位置 (3)
        4. 机体系边界距离向量 (18 = 6x3)
        """
        obs_dim = 3 + 3 + 3 + 18  # 总维度27
        
        # 边界距离向量的最大范围（保守估计）
        # 边界定义：X: [-half_x, +half_x], Y: [-half_y, +half_y], Z: [-boundary_z, 0]
        max_boundary_distance = jnp.sqrt(
            self.boundary_half_x**2 + self.boundary_half_y**2 + self.boundary_z**2
        ) * 2  # 对角线长度的2倍作为安全值
        
        low = jnp.concatenate([
            self.v_min,                       # 机体系速度最小值
            -jnp.ones(3),                     # 重力方向最小值
            jnp.array([-100.0, -100.0, -100.0]),  # 目标位置最小值（相对）
            -jnp.ones(18) * max_boundary_distance,  # 边界距离向量最小值
        ])
        high = jnp.concatenate([
            self.v_max,                       # 机体系速度最大值
            jnp.ones(3),                      # 重力方向最大值
            jnp.array([100.0, 100.0, 100.0]), # 目标位置最大值（相对）
            jnp.ones(18) * max_boundary_distance,  # 边界距离向量最大值
        ])
        return spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=jnp.float32)


if __name__ == "__main__":
    from aquila.utils.random import key_generator
    
    key_gen = key_generator(0)

    env = TrackEnvVer11()

    state, obs = env.reset(next(key_gen))
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial quad position: {state.quadrotor_state.p}")
    print(f"Initial target position: {state.target_pos}")
    print(f"Initial distance: {jnp.linalg.norm(state.quadrotor_state.p - state.target_pos)}")
    print(f"Boundary dimensions: x=[{-state.boundary_half_x:.1f}, {state.boundary_half_x:.1f}], "
          f"y=[{-state.boundary_half_y:.1f}, {state.boundary_half_y:.1f}], "
          f"z=[{-state.boundary_z:.1f}, 0.0] (NED: z-negative is upward)")
    
    random_action = env.action_space.sample(next(key_gen))
    transition = env.step(state, random_action, next(key_gen))
    state, obs, reward, terminated, truncated, info = transition
    print(f"\nAfter step:")
    print(f"Observation shape: {obs.shape}")
    print(f"Reward: {reward}")
    print(f"Distance to target: {info['distance_to_target']}")
    print(f"Quad position: {state.quadrotor_state.p}")
    print(f"Target position: {state.target_pos}")
    print(f"Terminated: {terminated}")