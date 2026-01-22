"""
Track Environment Version 13
Full quadrotor tracking environment with a moving target.
Ver13 modifications: Based on Ver12, used for testing more aggressive reward weights and target acceleration randomization.
The main differences from Ver12:
- Uses more aggressive reward weights for testing different training dynamics
- Target acceleration is randomized per episode (0 to max acceleration) instead of fixed

Ver12 features (inherited):
- Target initial position: randomized in x, y (±0.2m), and z (-2.2 to -1.8m)
- Target max speed: randomized from 0 to 1 m/s per episode
- Target acceleration: randomized from 0 to max acceleration per episode (Ver13 new)
- Drone initial orientation: roll and pitch randomized within ±30°, yaw fully randomized
- Drone initial velocity: random direction, magnitude in [0, 0.5] m/s
- Removed observation delay - uses true state values directly
- Reward includes thrust penalty (based on hoverVer1)

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
class TrackStateVer13:
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    target_pos: jax.Array
    target_vel: jax.Array
    target_direction: jax.Array  # 目标速度方向（单位向量，用于随机方向运动）
    quad_params: ExtendedQuadrotorParams  # 添加quadrotor参数（扩展版，包含mass和gravity）
    target_speed_max: float = 1.0  # 当前episode的目标最大速度（每次reset随机化）
    target_acceleration: float = 0.5  # 当前episode的目标加速度（每次reset随机化，0到最大加速度之间）
    action_raw: jax.Array = field_jnp(jnp.zeros(4))
    filtered_acc: jax.Array = field_jnp([0.0, 0.0, 9.8])
    filtered_thrust: float = field_jnp(9.8)
    # Flag to track if distance has ever exceeded reset_distance
    has_exceeded_distance: bool = False


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


class TrackEnvVer13(env_base.Env[TrackStateVer13]):
    """Quadrotor tracking environment Ver13 - based on Ver12, used for testing more aggressive reward weights and target acceleration randomization."""
    
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
        target_acceleration_max=0.5,  # m/s² (目标最大加速度，每次reset时在0到此值之间随机生成实际加速度)
        reset_distance=100.0,  # m (重置距离阈值)
        max_speed=20.0,  # m/s
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
        self.target_acceleration_max = target_acceleration_max
        self.reset_distance = reset_distance
        
        # Parameter randomization
        self.thrust_to_weight_min = thrust_to_weight_min
        self.thrust_to_weight_max = thrust_to_weight_max
        self.disturbance_mag = disturbance_mag

    def reset(
        self, key: chex.PRNGKey, state: Optional[TrackStateVer13] = None, quad_params: Optional[ExtendedQuadrotorParams] = None):
        """Reset environment with tracking-specific initialization.
        
        Args:
            key: Random key for initialization
            state: Optional state to reset to
            quad_params: Optional quadrotor parameters. If None, will use default or randomize based on key.
        """
        if state is not None:
            return state, self._get_obs(state)
        
        # 分割随机数key
        keys = jax.random.split(key, 14)
        key_target_x, key_target_y, key_target_z, key_target_dir, key_target_speed, key_target_acc, key_roll, key_pitch, key_yaw, key_omega, key_quad, key_randomize, key_vel_dir, key_vel_mag = keys
        
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
        # x正半轴上0.5～1.5m处
        target_x = jax.random.uniform(
            key_target_x, shape=(), 
            minval=self.target_init_distance_min, 
            maxval=self.target_init_distance_max
        )
        # y在(-0.2, 0.2)范围内随机
        target_y = jax.random.uniform(
            key_target_y, shape=(),
            minval=-0.2,
            maxval=0.2
        )
        # z在(-2.2, -1.8)范围内随机 (NED坐标系，高度在天上所以是负值)
        target_z = jax.random.uniform(
            key_target_z, shape=(),
            minval=-2.2,
            maxval=-1.8
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
        
        # 每次episode随机化目标加速度（0到最大加速度之间）
        episode_target_acceleration = jax.random.uniform(
            key_target_acc, shape=(),
            minval=0.0,
            maxval=self.target_acceleration_max
        )
        
        # 初始速度为0，将加速到episode的目标最大速度（沿随机方向）
        target_vel = jnp.array([0.0, 0.0, 0.0])
        
        # ========== 无人机初始化 ==========
        # 位置在原点，高度为2m（NED坐标系中z=-2）
        p = jnp.array([0.0, 0.0, -self.target_height])
        
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

        state = TrackStateVer13(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            target_pos=target_pos,
            target_vel=target_vel,
            target_direction=target_direction,
            quad_params=quad_params,
            target_speed_max=episode_target_speed_max,
            target_acceleration=episode_target_acceleration,
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            has_exceeded_distance=False,
        )
        
        return state, self._get_obs(state)

    def _get_obs(self, state: TrackStateVer13) -> jax.Array:
        """Get observation from state.
        
        Ver13修改：继承Ver12，移除观测延迟，直接使用真实状态值
        
        观测组成：
        1. 无人机机体系自身速度向量 (3)
        2. 无人机机体系重力方向 (3)
        3. 无人机机体系目标物体坐标 (3)
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

        # Combine all observations
        components = [
            v_body,                                # 机体系速度 (3)
            g_body,                                # 机体系重力方向 (3)
            target_pos_body,                       # 机体系目标位置 (3)
        ]  
        obs = jnp.concatenate(components)
        return obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: TrackStateVer13, action: jax.Array, key: chex.PRNGKey
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

        # 目标物体加速运动（从0加速到当前episode的最大速度，沿随机方向）
        current_speed_vec = state.target_vel
        current_speed = safe_norm(current_speed_vec, eps=1e-8)
        episode_target_speed_max = state.target_speed_max  # 使用当前episode的目标最大速度
        episode_target_acceleration = state.target_acceleration  # 使用当前episode的目标加速度（随机生成）
        
        # 如果当前速度小于最大速度，则加速
        new_speed = jnp.minimum(
            current_speed + episode_target_acceleration * self.dt,
            episode_target_speed_max
        )
        # 速度向量 = 速度大小 * 方向单位向量
        target_vel = new_speed * state.target_direction
        target_pos = state.target_pos + target_vel * self.dt
        
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
            target_acceleration=state.target_acceleration,  # 保持target_acceleration不变
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            target_pos=target_pos,
            target_vel=target_vel,
            target_direction=state.target_direction,  # 保持方向不变
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
        self, last_state: TrackStateVer13, next_state: TrackStateVer13
    ) -> jax.Array:
        """计算奖励 - 基于Ver12，使用更激进的奖励权重进行测试
        奖励设计：
        1. 方向损失：使用余弦相似度计算完整3D方向
        2. 距离损失：水平距离与目标距离的绝对差值
        3. 高度损失：无人机高度与目标高度的绝对差值
        4. 速度损失：相对速度模长
        5. 姿态损失：基于机体z轴方向的惩罚
        6. 动作损失：当前动作与上一动作的L2范数
        7. 角速度损失：惩罚旋转运动（防止roll持续旋转）
        8. 推力超限损失：动作推力与悬停推力的偏差（Ver10新增）
        
        Ver13与Ver12的区别：使用更激进的奖励权重进行测试
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
        direction_loss_base = jnp.exp(1 - cos_similarity) - 1
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
        ori_loss = 10 * (jnp.exp(ori_deviation) - 1.0)  # 指数增长
        
        # 6. 动作损失 (aux) - 当前动作与上一动作的L2范数，改为指数增长
        action_current = next_state.action_raw
        action_last = jnp.where(
            last_state.step_idx == 0,
            next_state.action_raw,  # step 0: 使用当前动作，变化为0
            last_state.action_raw   # step > 0: 使用真实的上一个动作
        )
        action_change = action_current - action_last
        action_error = safe_norm(action_change, eps=1e-8)
        action_loss = jnp.exp(action_error) - 1.0  # 指数增长
        
        # 7. 角速度损失 - 防止持续旋转（只惩罚roll和pitch，不惩罚yaw），改为指数增长
        omega_roll_pitch = quad_omega[:2]  # 只取roll和pitch角速度，忽略yaw
        omega_error = safe_norm(omega_roll_pitch, eps=1e-8)
        omega_loss = jnp.exp(omega_error) - 1.0  # 指数增长
        
        # 8. 推力超限损失 - 约束推力，动作推力与悬停推力的偏差（Ver10新增，参考hoverVer1，Ver12继承）
        # 使用当前动作（归一化后的值），需要去归一化
        thrust_normalized = action_current[0]
        # 去归一化推力：[-1, 1] -> [thrust_min*4, thrust_max*4]
        actual_thrust_max = next_state.quad_params.thrust_max
        action_thrust = 0.5 * (thrust_normalized + 1.0) * (actual_thrust_max * 4 - self.thrust_min * 4) + self.thrust_min * 4
        # 计算悬停推力：mass * gravity
        thrust_hover = next_state.quad_params.mass * next_state.quad_params.gravity
        # 计算推力偏差的绝对值，然后使用指数增长
        thrust_error = jnp.abs(action_thrust - thrust_hover)
        thrust_loss = jnp.exp(thrust_error) - 1.0  # 指数增长
        
        # 总损失 - 根据新的损失函数特性调整权重
        # 权重调整说明：
        # - 方向损失：有零惩罚范围(<15°)，超出后指数增长，权重降低到40
        # - 位置损失（距离和高度）：有零惩罚范围(<30cm)，超出后线性增长，保持较高权重80
        # - 速度损失：有零惩罚范围(<0.3m/s)，超出后线性增长，权重提高到3
        # - 姿态损失：改为指数增长，权重降低到0.5（指数增长本身会快速增加）
        # - 动作损失：改为指数增长，权重降低到4
        # - 角速度损失：改为指数增长，权重降低到4
        # - 推力超限损失：中等权重，约束推力接近悬停推力（Ver10新增，Ver12继承，Ver13继承）
        total_loss = (
            0.5 * ori_loss +           # 姿态损失：指数增长，权重降低
            150 * distance_loss +        # 距离损失：零惩罚范围后线性增长，保持较高权重
            3 * velocity_loss +         # 速度损失：零惩罚范围后线性增长，权重提高
            40 * direction_loss +       # 方向损失：零惩罚范围后指数增长，权重稍微降低
            8 * height_loss +          # 高度损失：零惩罚范围后线性增长，保持较高权重
            0.04 * action_loss +           # 动作损失：指数增长，权重降低
            0.01 * omega_loss +            # 角速度损失：指数增长，权重降低
            0.04 * thrust_loss            # 推力超限损失：中等权重，约束推力接近悬停推力（Ver10新增，Ver12继承，Ver13继承）
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
        
        Ver13修改：继承Ver12，移除观测延迟，直接使用真实状态值
        
        观测组成：
        1. 机体系速度 (3)
        2. 机体系重力方向 (3)
        3. 机体系目标位置 (3)
        """
        obs_dim = 3 + 3 + 3  # 总维度9
        
        low = jnp.concatenate([
            self.v_min,                       # 机体系速度最小值
            -jnp.ones(3),                     # 重力方向最小值
            jnp.array([-100.0, -100.0, -100.0]),  # 目标位置最小值（相对）
        ])
        high = jnp.concatenate([
            self.v_max,                       # 机体系速度最大值
            jnp.ones(3),                      # 重力方向最大值
            jnp.array([100.0, 100.0, 100.0]), # 目标位置最大值（相对）
        ])
        return spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=jnp.float32)


if __name__ == "__main__":
    from aquila.utils.random import key_generator
    
    key_gen = key_generator(0)

    env = TrackEnvVer13()

    state, obs = env.reset(next(key_gen))
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial quad position: {state.quadrotor_state.p}")
    print(f"Initial target position: {state.target_pos}")
    print(f"Initial distance: {jnp.linalg.norm(state.quadrotor_state.p - state.target_pos)}")
    
    random_action = env.action_space.sample(next(key_gen))
    transition = env.step(state, random_action, next(key_gen))
    state, obs, reward, terminated, truncated, info = transition
    print(f"\nAfter step:")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Distance to target: {info['distance_to_target']}")
    print(f"Terminated: {terminated}")