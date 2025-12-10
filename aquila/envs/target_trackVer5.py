"""
Track Environment Version 5
Full quadrotor tracking environment with a target moving on x-axis.
Ver5 modification: Based on Ver3 but using full Quadrotor model instead of QuadrotorSimple, with bpttVer3 compatibility.

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
class TrackStateVer5:
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    target_pos: jax.Array
    target_vel: jax.Array
    target_direction: jax.Array  # 目标速度方向（单位向量，用于随机方向运动）
    quad_params: ExtendedQuadrotorParams  # 添加quadrotor参数（扩展版，包含mass和gravity）
    action_raw: jax.Array = field_jnp(jnp.zeros(4))
    filtered_acc: jax.Array = field_jnp([0.0, 0.0, 9.8])
    filtered_thrust: float = field_jnp(9.8)
    # Observed (sensor) dynamics states
    obs_p: jax.Array = field_jnp(jnp.zeros(3))
    obs_v: jax.Array = field_jnp(jnp.zeros(3))
    obs_R: jax.Array = field_jnp(jnp.eye(3))
    # Flag to track if distance has ever exceeded 10m
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


class TrackEnvVer5(env_base.Env[TrackStateVer5]):
    """Quadrotor tracking environment Ver5 - based on Ver3 but using full Quadrotor model instead of QuadrotorSimple."""
    
    def __init__(
        self,
        *,
        max_steps_in_episode=1000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        drone_path=None,
        action_penalty_weight=0.1,
        # Observation dynamics time constants (seconds)
        obs_tau_pos=0.3,
        obs_tau_vel=0.2,
        obs_tau_R=0.02,
        # Tracking specific parameters
        target_height=2.0,  # m (height above ground, positive value)
        target_init_distance_min=0.5,  # m (x轴上的初始距离最小值)
        target_init_distance_max=1.5,  # m (x轴上的初始距离最大值)
        target_speed_max=1.0,  # m/s (目标最大速度)
        target_acceleration=0.5,  # m/s² (目标加速度，从0加速到最大速度)
        reset_distance=100.0,  # m (重置距离阈值)
        max_speed=20.0,  # m/s
        # Parameter randomization (quadrotor)
        thrust_to_weight_min=1.5,  # 最小推重比
        thrust_to_weight_max=3.0,  # 最大推重比
        disturbance_mag=0.0,  # [N] 常值扰动力大小（训练时>0，测试时=0）
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

        # Observation dynamics params
        self.obs_tau_pos = float(obs_tau_pos)
        self.obs_tau_vel = float(obs_tau_vel)
        self.obs_tau_R = float(obs_tau_R)
        
        # Tracking specific parameters
        self.target_height = target_height
        self.target_init_distance_min = target_init_distance_min
        self.target_init_distance_max = target_init_distance_max
        self.target_speed_max = target_speed_max
        self.target_acceleration = target_acceleration
        self.reset_distance = reset_distance
        
        # Parameter randomization
        self.thrust_to_weight_min = thrust_to_weight_min
        self.thrust_to_weight_max = thrust_to_weight_max
        self.disturbance_mag = disturbance_mag

    def reset(
        self, key: chex.PRNGKey, state: Optional[TrackStateVer5] = None, quad_params: Optional[ExtendedQuadrotorParams] = None):
        """Reset environment with tracking-specific initialization.
        
        Args:
            key: Random key for initialization
            state: Optional state to reset to
            quad_params: Optional quadrotor parameters. If None, will use default or randomize based on key.
        """
        if state is not None:
            return state, self._get_obs(state)
        
        # 分割随机数key
        keys = jax.random.split(key, 7)
        key_target_pos, key_target_vel, key_target_dir, key_yaw, key_omega, key_quad, key_randomize = keys
        
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
        # x正半轴上1～5m处
        target_x = jax.random.uniform(
            key_target_pos, shape=(), 
            minval=self.target_init_distance_min, 
            maxval=self.target_init_distance_max
        )
        # y=0, z=-2m (NED坐标系，高度2m在天上所以是-2)
        target_pos = jnp.array([target_x, 0.0, -self.target_height])
        
        # 随机生成速度方向（单位向量，限制在水平面xy上）
        # 生成一个随机角度（0到2π），然后在xy平面上生成单位向量
        random_angle = jax.random.uniform(key_target_dir, shape=(), minval=0.0, maxval=2.0 * jnp.pi)
        target_direction = jnp.array([jnp.cos(random_angle), jnp.sin(random_angle), 0.0])
        
        # 初始速度为0，将加速到目标最大速度（沿随机方向）
        target_vel = jnp.array([0.0, 0.0, 0.0])
        
        # ========== 无人机初始化 ==========
        # 位置在原点，高度为2m（NED坐标系中z=-2）
        p = jnp.array([0.0, 0.0, -self.target_height])
        
        # 初始速度为零
        v = jnp.zeros(3)
        
        # roll=0, pitch=0, yaw随机
        init_roll = 0.0
        init_pitch = 0.0
        init_yaw = jax.random.uniform(key_yaw, shape=(), minval=-jnp.pi, maxval=jnp.pi)
        # init_yaw = 0.0
        
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

        state = TrackStateVer5(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            target_pos=target_pos,
            target_vel=target_vel,
            target_direction=target_direction,
            quad_params=quad_params,
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            obs_p=p,
            obs_v=v,
            obs_R=R,
            has_exceeded_distance=False,
        )
        
        return state, self._get_obs(state)

    def _get_obs(self, state: TrackStateVer5) -> jax.Array:
        """Get observation from state.
        
        Ver3修改：基于Ver2，移除了推重比、最大角速度和目标物体速度
        
        观测组成：
        1. 无人机机体系自身速度向量 (3)
        2. 无人机机体系重力方向 (3)
        3. 无人机机体系目标物体坐标 (3)
        """
        # Observed states
        p_obs = state.obs_p
        v_obs = state.obs_v
        R_obs = state.obs_R
        R_transpose = jnp.transpose(R_obs)
        
        # 1. 无人机机体系自身速度向量
        v_body_obs = R_transpose @ v_obs
        
        # 2. 无人机机体系重力方向
        g_world = jnp.array([0.0, 0.0, 1.0])  # NED坐标系中重力方向 (Down为正)
        g_body_obs = R_transpose @ g_world
        
        # 3. 无人机机体系目标物体坐标（相对位置）
        target_pos_world = state.target_pos
        target_pos_relative_world = target_pos_world - p_obs
        target_pos_body = R_transpose @ target_pos_relative_world

        # Combine all observations (Ver3: 移除了推重比、最大角速度和目标速度)
        components = [
            v_body_obs,                            # 机体系速度 (3)
            g_body_obs,                            # 机体系重力方向 (3)
            target_pos_body,                       # 机体系目标位置 (3)
        ]  
        obs = jnp.concatenate(components)
        return obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: TrackStateVer5, action: jax.Array, key: chex.PRNGKey
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
        
        # 观测一阶惯性系数（根据时间常数）
        alpha_pos = jnp.array(self.dt / (self.obs_tau_pos + self.dt), dtype=jnp.float32)
        alpha_vel = jnp.array(self.dt / (self.obs_tau_vel + self.dt), dtype=jnp.float32)
        
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
        
        # 观测：位置/速度一阶滤波
        obs_p_new = first_order_filter(quadrotor_state.p, state.obs_p, alpha_pos)
        obs_v_new = first_order_filter(quadrotor_state.v, state.obs_v, alpha_vel)
        
        # 观测：姿态一拍延时（一步delay）
        obs_R_new = state.quadrotor_state.R

        # 目标物体加速运动（从0加速到最大速度，沿随机方向）
        current_speed_vec = state.target_vel
        current_speed = safe_norm(current_speed_vec, eps=1e-8)
        target_speed_max = self.target_speed_max
        target_acc = self.target_acceleration
        
        # 如果当前速度小于最大速度，则加速
        new_speed = jnp.minimum(
            current_speed + target_acc * self.dt,
            target_speed_max
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
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            obs_p=obs_p_new,
            obs_v=obs_v_new,
            obs_R=obs_R_new,
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
        self, last_state: TrackStateVer5, next_state: TrackStateVer5
    ) -> jax.Array:
        """计算奖励 - 基于 agile_lossVer7 算法
        奖励设计：
        1. 方向损失：使用余弦相似度计算完整3D方向
        2. 距离损失：水平距离与目标距离的绝对差值
        3. 高度损失：无人机高度与目标高度的绝对差值
        4. 速度损失：相对速度模长
        5. 姿态损失：基于机体z轴方向的惩罚
        6. 动作损失：当前动作与上一动作的L2范数
        7. 角速度损失：惩罚旋转运动（防止roll持续旋转）
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
        direction_loss = jnp.exp(1 - cos_similarity) - 1
        
        
        # 2. 距离损失 (distance) - 水平距离与目标距离的绝对差值
        norm_hor_dis = safe_norm(p_rel[:2], eps=1e-8)
        target_distance = 1.0  # 目标距离1米
        distance_loss = jnp.abs(norm_hor_dis - target_distance)
        
        # 3. 高度损失 (h) - 无人机高度与目标高度的绝对差值
        height_loss = jnp.abs(quad_pos[2] - target_pos[2])
        
        # 4. 速度损失 (vel) - 相对速度模长
        velocity_loss = safe_norm(v_rel, eps=1e-8)
        
        # 5. 姿态损失 (ori) - 基于机体z轴方向的惩罚
        body_z_world = quad_R @ jnp.array([0.0, 0.0, -1.0])  # 机体z轴在世界系中的方向
        # 理想情况下，机体z轴应该指向上方（-z方向），body_z_world应该接近[0, 0, -1]
        # 惩罚当body_z_world[2]偏离-1的情况（即偏离垂直）
        # body_z_world[2] = -1时正常飞行(loss=0)，= +1时翻转(loss=40)，= 0时侧倾(loss=10)
        ori_loss = 10 * (body_z_world[2] + 1.0) ** 2
        
        # 6. 动作损失 (aux) - 当前动作与上一动作的L2范数
        action_current = next_state.action_raw
        action_last = jnp.where(
            last_state.step_idx == 0,
            next_state.action_raw,  # step 0: 使用当前动作，变化为0
            last_state.action_raw   # step > 0: 使用真实的上一个动作
        )
        action_change = action_current - action_last
        action_loss = safe_norm(action_change, eps=1e-8)
        
        # 7. 角速度损失 - 防止持续旋转
        omega_loss = safe_norm(quad_omega, eps=1e-8)
        
        # 总损失 - 添加角速度惩罚防止roll旋转
        total_loss = (
            1 * ori_loss + 
            100 * distance_loss + 
            1 * velocity_loss + 
            50 * direction_loss + 
            100 * height_loss + 
            1 * action_loss +
            1 * omega_loss  # 新增：惩罚角速度防止持续旋转
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
        
        Ver3修改：基于Ver2，移除了推重比、最大角速度和目标物体速度
        
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

    env = TrackEnvVer5()

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