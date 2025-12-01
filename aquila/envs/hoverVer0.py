"""
Hover Environment Version 0
Full quadrotor hovering environment at the origin.
Ver0 modifications: Based on TrackVer9, now focuses on hovering at origin:
- No target object - drone should hover at origin (0, 0, -2.0)
- Drone initial position: randomized within a ball of radius 0.5m around origin
- Drone initial orientation: roll, pitch, yaw fully randomized
- Drone initial velocity: random direction, magnitude in [0, 2] m/s
- Drone initial angular velocity: randomized within max angular velocity limits
- Observation includes relative position to origin instead of target
- Uses QuadrotorVer2 with PID control parameter randomization support

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

from aquila.objects.quadrotor_objVer2 import QuadrotorVer2, QuadrotorState, QuadrotorParams
from aquila.objects.world_box_obj import WorldBox
from aquila.utils import spaces
from aquila.utils.pytrees import field_jnp
from aquila.utils.math import smooth_l1
import aquila.envs.env_base as env_base
from aquila.envs.env_base import EnvTransition
import dataclasses


@jdc.pytree_dataclass
class ExtendedQuadrotorParams(QuadrotorParams):
    """扩展的QuadrotorParams Ver2，包含mass、gravity和external_force以便支持动态扰动
    继承自QuadrotorParams Ver2，包含Kp (PID比例增益)参数"""
    mass: float = 1.0  # [kg]
    gravity: float = 9.81  # [m/s^2]
    external_force: jax.Array = field_jnp([0.0, 0.0, 0.0])  # [N] 外部作用力（如风力）


@jdc.pytree_dataclass
class HoverStateVer0:
    time: float
    step_idx: int
    quadrotor_state: QuadrotorState
    last_actions: jax.Array
    quad_params: ExtendedQuadrotorParams  # 添加quadrotor参数（扩展版，包含mass和gravity）
    action_raw: jax.Array = field_jnp(jnp.zeros(4))
    filtered_acc: jax.Array = field_jnp([0.0, 0.0, 9.8])
    filtered_thrust: float = field_jnp(9.8)
    hover_origin: jax.Array = field_jnp([0.0, 0.0, -2.0])  # 悬停目标位置（原点）


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


class HoverEnvVer0(env_base.Env[HoverStateVer0]):
    """Quadrotor hovering environment Ver0 - hover at origin, based on TrackVer9 but with QuadrotorVer2 and PID parameter randomization."""
    
    def __init__(
        self,
        *,
        max_steps_in_episode=1000,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        drone_path=None,
        action_penalty_weight=0.1,
        # Hovering specific parameters
        hover_height=2.0,  # m (hover height above ground, positive value)
        init_pos_range=0.5,  # m (initial position randomization range in x, y)
        max_distance=10.0,  # m (maximum distance from origin before reset)
        max_speed=20.0,  # m/s
        # Parameter randomization (quadrotor)
        thrust_to_weight_min=1.2,  # 最小推重比
        thrust_to_weight_max=3.0,  # 最大推重比
    ):
        self.world_box = WorldBox(
            jnp.array([-5000.0, -5000.0, -5000.0]), jnp.array([5000.0, 5000.0, 5000.0])
        )
        self.max_steps_in_episode = max_steps_in_episode
        self.dt = np.array(dt)
        
        self.omega_std = omega_std
        
        # quadrotor - 使用完整的四旋翼模型 Ver2（基于agilicious framework，支持PID参数随机化）
        self.quadrotor = QuadrotorVer2()
        
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
        
        # Hovering specific parameters
        self.hover_height = hover_height
        self.init_pos_range = init_pos_range
        self.max_distance = max_distance
        self.hover_origin = jnp.array([0.0, 0.0, -hover_height])  # NED坐标系中的悬停目标位置
        
        # Parameter randomization
        self.thrust_to_weight_min = thrust_to_weight_min
        self.thrust_to_weight_max = thrust_to_weight_max

    def reset(
        self, key: chex.PRNGKey, state: Optional[HoverStateVer0] = None, quad_params: Optional[ExtendedQuadrotorParams] = None):
        """Reset environment with hovering-specific initialization.
        
        Randomization:
        - Position: Uniformly distributed within a ball of radius [0, 0.5]m around origin
        - Velocity: Random direction, magnitude uniformly distributed in [0, 2] m/s
        - Orientation: Roll and yaw in [-π, π], pitch in [-π/2, π/2]
        - Angular velocity: Uniformly distributed in [-omega_max, omega_max] for each axis
        
        Args:
            key: Random key for initialization
            state: Optional state to reset to
            quad_params: Optional quadrotor parameters. If None, will use default or randomize based on key.
        """
        if state is not None:
            return state, self._get_obs(state)
        
        # 分割随机数key
        keys = jax.random.split(key, 7)
        key_pos, key_angles, key_vel, key_omega, key_randomize, key_vel_dir, key_vel_mag = keys
        
        # 获取quadrotor参数（如果没有提供则使用默认参数）
        if quad_params is None:
            # 启用参数随机化（质量固定，推力、角速度、PID参数随机化）
            base_params = self.quadrotor.default_params()
            randomized_params = QuadrotorVer2.randomize_params(
                base_params,
                self.quadrotor._mass,  # QuadrotorVer2.randomize_params 需要 mass 参数
                key_randomize,
                thrust_to_weight_min=self.thrust_to_weight_min,
                thrust_to_weight_max=self.thrust_to_weight_max
            )
            # 转换为扩展的参数类，添加mass和gravity，保留Kp参数（Ver2新增）
            quad_params = ExtendedQuadrotorParams(
                thrust_max=randomized_params.thrust_max,
                omega_max=randomized_params.omega_max,
                motor_tau=randomized_params.motor_tau,
                Kp=randomized_params.Kp,  # Ver2: PID比例增益参数
                mass=self.quadrotor._mass,
                gravity=9.81
            )
        
        # ========== 无人机初始化 ==========
        # 位置：距离原点0~0.5m范围内随机
        # 生成随机单位向量（均匀分布在球面上）
        pos_keys = jax.random.split(key_pos, 3)
        # 使用球坐标系生成均匀分布
        theta = jax.random.uniform(pos_keys[0], shape=(), minval=0.0, maxval=2.0 * jnp.pi)  # 方位角
        phi = jnp.arccos(jax.random.uniform(pos_keys[1], shape=(), minval=-1.0, maxval=1.0))  # 极角
        
        # 球坐标转笛卡尔坐标
        random_direction = jnp.array([
            jnp.sin(phi) * jnp.cos(theta),
            jnp.sin(phi) * jnp.sin(theta),
            jnp.cos(phi)
        ])
        
        # 距离在0~0.5m之间随机
        random_distance = jax.random.uniform(pos_keys[2], shape=(), minval=0.0, maxval=0.5)
        
        # 位置 = 原点 + 随机距离 * 随机方向
        p = self.hover_origin + random_distance * random_direction
        
        # 速度：随机方向，大小在0~2m/s范围内随机
        vel_keys = jax.random.split(key_vel, 2)
        vel_theta = jax.random.uniform(vel_keys[0], shape=(), minval=0.0, maxval=0.1 * jnp.pi)
        vel_phi = jnp.arccos(jax.random.uniform(vel_keys[1], shape=(), minval=-1.0, maxval=1.0))
        
        vel_direction = jnp.array([
            jnp.sin(vel_phi) * jnp.cos(vel_theta),
            jnp.sin(vel_phi) * jnp.sin(vel_theta),
            jnp.cos(vel_phi)
        ])
        vel_magnitude = jax.random.uniform(key_vel_mag, shape=(), minval=0.0, maxval=2.0)
        v = vel_magnitude * vel_direction
        
        # 姿态角：roll, pitch, yaw都完全随机
        angle_keys = jax.random.split(key_angles, 3)
        init_roll = jax.random.uniform(angle_keys[0], shape=(), minval=-jnp.pi, maxval=jnp.pi)
        init_pitch = jax.random.uniform(angle_keys[1], shape=(), minval=-jnp.pi/2, maxval=jnp.pi/2)  # pitch限制在±90°避免万向锁
        init_yaw = jax.random.uniform(angle_keys[2], shape=(), minval=-jnp.pi, maxval=jnp.pi)
        
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
        
        # 角速度：在最大角速度范围内随机（使用当前episode的omega_max参数）
        actual_omega_max = quad_params.omega_max
        omega = jax.random.uniform(key_omega, shape=(3,), minval=-actual_omega_max, maxval=actual_omega_max)

        # Initialize quadrotor state
        # QuadrotorVer2.create_state 使用位置参数 p, R, v，其他参数通过 kwargs 传递
        quadrotor_state = self.quadrotor.create_state(p, R, v, omega=omega)
        
        # Calculate hovering action based on current episode's quad_params
        # 悬停推力 = mass * gravity（使用当前episode的实际质量）
        thrust_hover = self.quadrotor._mass * 9.81  # QuadrotorParams Ver2 不包含 mass 和 gravity，使用实例的 _mass
        hovering_action = jnp.array([thrust_hover, 0.0, 0.0, 0.0])
        
        # Initialize action history
        last_actions = jax.device_put(jnp.tile(hovering_action, (self.num_last_actions, 1)))
        action_raw = jax.device_put(jnp.zeros(4))
        filtered_acc = jax.device_put(jnp.array([0.0, 0.0, 9.81]))  # NED坐标系，Down为正
        filtered_thrust = jax.device_put(jnp.array(thrust_hover))

        state = HoverStateVer0(
            time=0.0,
            step_idx=0,
            quadrotor_state=quadrotor_state,
            last_actions=last_actions,
            quad_params=quad_params,
            action_raw=action_raw,
            filtered_acc=filtered_acc,
            filtered_thrust=filtered_thrust,
            hover_origin=self.hover_origin,
        )
        
        return state, self._get_obs(state)

    def _get_obs(self, state: HoverStateVer0) -> jax.Array:
        """Get observation from state.
        
        Ver0修改：基于TrackVer9，将目标位置改为悬停原点
        
        观测组成：
        1. 无人机机体系自身速度向量 (3)
        2. 无人机机体系重力方向 (3)
        3. 无人机机体系到原点的相对位置 (3)
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
        
        # 3. 无人机机体系到原点的相对位置（原点 - 当前位置）
        origin_pos_relative_world = state.hover_origin - quad_pos
        origin_pos_body = R_transpose @ origin_pos_relative_world

        # Combine all observations
        components = [
            v_body,                                # 机体系速度 (3)
            g_body,                                # 机体系重力方向 (3)
            origin_pos_body,                       # 机体系到原点的相对位置 (3)
        ]  
        obs = jnp.concatenate(components)
        return obs

    @functools.partial(jax.jit, static_argnums=(0,))
    def _step(
        self, state: HoverStateVer0, action: jax.Array, key: chex.PRNGKey
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
        # 直接传递ExtendedQuadrotorParams（包含mass、gravity、external_force、Kp）
        # QuadrotorVer2的step方法现在支持ExtendedQuadrotorParams（包含Kp参数）
        quadrotor_state = self.quadrotor.step(
            state.quadrotor_state, f_1, omega_1, dt_1, 
            drag_params=None,  # 使用默认drag_params
            quad_params=state.quad_params  # 使用ExtendedQuadrotorParams（Ver2包含Kp）
        )

        if self.delay > 0:
            # 2 step
            dt_2 = self.dt - dt_1
            action_2 = last_actions[1]
            f_2, omega_2 = action_2[0], action_2[1:]
            quadrotor_state = self.quadrotor.step(
                quadrotor_state, f_2, omega_2, dt_2,
                drag_params=None,  # 使用默认drag_params
                quad_params=state.quad_params  # 使用ExtendedQuadrotorParams（Ver2包含Kp）
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
            hover_origin=state.hover_origin,  # 保持悬停原点不变
        )

        obs = self._get_obs(next_state)
        reward = self._compute_reward(state, next_state)
        
        # 检查是否需要重置（距离原点大于max_distance）
        distance_to_origin = safe_norm(next_state.quadrotor_state.p - next_state.hover_origin, eps=1e-8)
        terminated = distance_to_origin > self.max_distance
        
        truncated = jnp.greater_equal(
            next_state.step_idx, self.max_steps_in_episode
        )
        
        info = {
            "quad_p": next_state.quadrotor_state.p,
            "quad_v": next_state.quadrotor_state.v,
            "quad_acc": next_state.quadrotor_state.acc,
            "quad_R": next_state.quadrotor_state.R,
            "action": next_state.last_actions[-1],
            "distance_to_origin": distance_to_origin,
        }

        return EnvTransition(
            next_state, obs, reward, terminated, truncated, info
        )

    def _compute_reward(
        self, last_state: HoverStateVer0, next_state: HoverStateVer0
    ) -> jax.Array:
        """计算奖励 - 悬停任务奖励函数
        奖励设计：
        1. 位置损失：到原点的水平距离
        2. 高度损失：与目标高度的偏差
        3. 速度损失：速度模长（应该接近0）
        4. 姿态损失：基于机体z轴方向的惩罚
        5. 动作损失：当前动作与上一动作的L2范数
        6. 角速度损失：惩罚旋转运动
        """
        # 获取状态信息
        quad_pos = next_state.quadrotor_state.p
        quad_vel = next_state.quadrotor_state.v
        quad_R = next_state.quadrotor_state.R
        quad_omega = next_state.quadrotor_state.omega
        hover_origin = next_state.hover_origin
        
        # 1. 位置损失 (position) - 到原点的水平距离
        p_rel = quad_pos - hover_origin
        horizontal_distance = safe_norm(p_rel[:2], eps=1e-8)
        # 零惩罚范围：位置 < 10cm 时损失为0
        position_threshold = 0.1  # 10cm
        position_loss = jnp.where(
            horizontal_distance < position_threshold,
            0.0,
            horizontal_distance - position_threshold  # 超出后从0开始线性增加
        )
        
        # 2. 高度损失 (height) - 与目标高度的偏差
        height_error = jnp.abs(quad_pos[2] - hover_origin[2])
        # 零惩罚范围：高度 < 10cm 时损失为0
        height_loss = jnp.where(
            height_error < position_threshold,
            0.0,
            height_error - position_threshold  # 超出后从0开始线性增加
        )
        
        # 3. 速度损失 (velocity) - 速度模长（悬停时应该接近0）
        velocity_error = safe_norm(quad_vel, eps=1e-8)
        # 零惩罚范围：速度 < 0.1m/s 时损失为0
        velocity_threshold = 0.1  # 0.1m/s
        velocity_loss = jnp.where(
            velocity_error < velocity_threshold,
            0.0,
            velocity_error - velocity_threshold  # 超出后从0开始线性增加
        )
        
        # 4. 姿态损失 (orientation) - 基于机体z轴方向的惩罚
        body_z_world = quad_R @ jnp.array([0.0, 0.0, -1.0])  # 机体z轴在世界系中的方向
        # 理想情况下，机体z轴应该指向上方（-z方向），body_z_world应该接近[0, 0, -1]
        ori_deviation = (body_z_world[2] + 1.0) ** 2  # 偏离度（0到4之间）
        ori_loss = 10 * (jnp.exp(ori_deviation) - 1.0)  # 指数增长
        
        # 5. 动作损失 (action) - 当前动作与上一动作的L2范数
        action_current = next_state.action_raw
        action_last = jnp.where(
            last_state.step_idx == 0,
            next_state.action_raw,  # step 0: 使用当前动作，变化为0
            last_state.action_raw   # step > 0: 使用真实的上一个动作
        )
        action_change = action_current - action_last
        action_error = safe_norm(action_change, eps=1e-8)
        action_loss = jnp.exp(action_error) - 1.0  # 指数增长
        
        # 6. 角速度损失 - 防止持续旋转
        omega_error = safe_norm(quad_omega, eps=1e-8)
        omega_loss = jnp.exp(omega_error) - 1.0  # 指数增长
        
        # 总损失 - 悬停任务权重调整
        # - 位置损失：最高权重，确保在原点附近悬停
        # - 高度损失：高权重，保持目标高度
        # - 速度损失：高权重，保持静止
        # - 姿态损失：中等权重，保持水平姿态
        # - 动作损失：低权重，平滑控制
        # - 角速度损失：低权重，减少旋转
        total_loss = (
            10 * position_loss +      # 位置损失：最高权重
            100 * height_loss +        # 高度损失：最高权重
            5 * velocity_loss +       # 速度损失：高权重
            10 * ori_loss +           # 姿态损失：中等权重
            10 * action_loss +          # 动作损失：低权重
            10 * omega_loss             # 角速度损失：低权重
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
        
        Ver0修改：基于TrackVer9，改为悬停任务观测空间
        
        观测组成：
        1. 机体系速度 (3)
        2. 机体系重力方向 (3)
        3. 机体系到原点的相对位置 (3)
        """
        obs_dim = 3 + 3 + 3  # 总维度9
        
        low = jnp.concatenate([
            self.v_min,                       # 机体系速度最小值
            -jnp.ones(3),                     # 重力方向最小值
            jnp.array([-100.0, -100.0, -100.0]),  # 到原点相对位置最小值
        ])
        high = jnp.concatenate([
            self.v_max,                       # 机体系速度最大值
            jnp.ones(3),                      # 重力方向最大值
            jnp.array([100.0, 100.0, 100.0]), # 到原点相对位置最大值
        ])
        return spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=jnp.float32)


if __name__ == "__main__":
    from aquila.utils.random import key_generator
    
    key_gen = key_generator(0)

    env = HoverEnvVer0()

    state, obs = env.reset(next(key_gen))
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial quad position: {state.quadrotor_state.p}")
    print(f"Hover origin: {state.hover_origin}")
    print(f"Initial distance to origin: {jnp.linalg.norm(state.quadrotor_state.p - state.hover_origin)}")
    
    random_action = env.action_space.sample(next(key_gen))
    transition = env.step(state, random_action, next(key_gen))
    state, obs, reward, terminated, truncated, info = transition
    print(f"\nAfter step:")
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Distance to origin: {info['distance_to_origin']}")
    print(f"Terminated: {terminated}")