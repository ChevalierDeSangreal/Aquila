import os
from functools import partial
from typing import Optional

import jax
import jax_dataclasses as jdc
import numpy as np
import chex
from jax import numpy as jnp
from aquila.utils.pytrees import field_jnp, CustomPyTree
from aquila.utils.math import rotation_matrix_from_vector


@jdc.pytree_dataclass
class SimpleQuadrotorParams:
    """简化的四旋翼参数，用于参数随机化（与JIT兼容）"""
    mass: float = 1.0  # [kg]
    inertia: jax.Array = field_jnp([0.00014, 0.00016, 0.0002])  # [kgm^2]
    arm_length: float = 0.04  # [m]
    motor_tau: float = 0.04  # [s]
    thrust_max: float = 9.0  # [N] per motor
    omega_max: jax.Array = field_jnp([5.0, 5.0, 5.0])  # [rad/s]
    kappa: float = 0.008  # [Nm/N]
    gravity: float = 9.81  # [m/s^2]


@jdc.pytree_dataclass
class SimpleQuadrotorState(CustomPyTree):
    """简化的四旋翼状态"""
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])  # position [m]
    R: jax.Array = field_jnp(jnp.eye(3))  # rotation matrix
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])  # velocity [m/s]
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])  # angular velocity [rad/s]
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])  # acceleration [m/s^2]
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))

    def detached(self):
        return SimpleQuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            omega=jax.lax.stop_gradient(self.omega),
            acc=jax.lax.stop_gradient(self.acc),
            dr_key=jax.lax.stop_gradient(self.dr_key)
        )

    def as_vector(self):
        return jnp.concatenate([self.p, self.R.flatten(), self.v, self.omega])

    @classmethod
    def from_vector(cls, vector):
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        return cls(p, R, v, omega)


class QuadrotorSimple:
    """
    简化的四旋翼模型，结合了dynamics_simple.py和quadrotor_obj.py的优点。
    使用JAX实现，适合强化学习环境。
    """

    def __init__(
        self,
        *,
        mass=1.0,  # [kg]
        inertia=jnp.array([0.00014, 0.00016, 0.0002]),  # [kgm^2]
        arm_length=0.04,  # [m]
        motor_tau=0.04,  # [s]
        thrust_max=9.0,  # [N] per motor
        omega_max=jnp.array([5.0, 5.0, 5.0]),  # [rad/s]
        kappa=0.008,  # [Nm/N]
        gravity=9.81,  # [m/s^2]
        dt=0.01,  # [s]
    ):
        self._mass = mass
        self._inertia = inertia
        self._arm_length = arm_length
        self._motor_tau = motor_tau
        self._thrust_max = thrust_max
        self._omega_max = omega_max
        self._kappa = kappa
        self._gravity = jnp.array([0, 0, gravity])  # 向下为正
        self._dt = dt

        # 推力映射参数（来自dynamics_simple.py）
        self._thrust_coeff_a = 0.962978
        self._thrust_coeff_b = 0.037022
        self._thrust_scale = 35.406905

        # 惯性矩阵
        self._inertia_matrix = jnp.diag(self._inertia)
        self._inertia_matrix_inv = jnp.linalg.inv(self._inertia_matrix)

        # 分配矩阵（用于将电机力转换为机体力和力矩）
        self._allocation_matrix = self._compute_allocation_matrix()
        self._allocation_matrix_inv = jnp.linalg.inv(self._allocation_matrix)

    def _compute_allocation_matrix(self) -> jnp.ndarray:
        """计算分配矩阵，将电机力映射到机体力和力矩"""
        # 电机位置（十字配置）
        motor_positions = jnp.array([
            [self._arm_length, self._arm_length, 0],    # FR
            [-self._arm_length, -self._arm_length, 0],  # BL
            [-self._arm_length, self._arm_length, 0],    # BR
            [self._arm_length, -self._arm_length, 0]     # FL
        ])

        x = motor_positions[:, 0]
        y = motor_positions[:, 1]

        # 分配矩阵：[总推力, tau_x, tau_y, tau_z] = A @ [f1, f2, f3, f4]
        allocation_matrix = jnp.array([
            jnp.ones(4),  # 总推力
            y,             # 滚转力矩 (tau_x)
            -x,            # 俯仰力矩 (tau_y)
            self._kappa * jnp.array([-1, -1, 1, 1])  # 偏航力矩 (tau_z)
        ])

        return allocation_matrix

    def default_state(self):
        """获取默认悬停状态"""
        return SimpleQuadrotorState()

    def create_state(self, p=None, R=None, v=None, omega=None, **kwargs):
        """创建四旋翼状态"""
        if p is None:
            p = jnp.zeros(3)
        if R is None:
            R = jnp.eye(3)
        if v is None:
            v = jnp.zeros(3)
        if omega is None:
            omega = jnp.zeros(3)

        return SimpleQuadrotorState(p=p, R=R, v=v, omega=omega, **kwargs)

    def default_params(self) -> SimpleQuadrotorParams:
        """获取默认四旋翼参数"""
        return SimpleQuadrotorParams(
            mass=self._mass,
            inertia=self._inertia,
            arm_length=self._arm_length,
            motor_tau=self._motor_tau,
            thrust_max=self._thrust_max,
            omega_max=self._omega_max,
            kappa=self._kappa,
            gravity=self._gravity[2]
        )


    def linear_dynamics(self, f_d: jnp.ndarray, R: jnp.ndarray, params: SimpleQuadrotorParams = None) -> jnp.ndarray:
        """
        计算线性加速度
        
        Args:
            f_d: 期望总推力 [N]
            R: 旋转矩阵
            params: 四旋翼参数
        
        Returns:
            线性加速度 [m/s^2]
        """
        if params is None:
            params = self.default_params()

        # 机体坐标系中的推力向量（向上）
        thrust_body = jnp.array([0, 0, f_d])

        # 将推力转换到世界坐标系
        thrust_world = R @ thrust_body

        # 添加重力
        acceleration = thrust_world / params.mass + self._gravity

        return acceleration

    def angular_dynamics(self, omega_d: jnp.ndarray, current_omega: jnp.ndarray, params: SimpleQuadrotorParams = None) -> jnp.ndarray:
        """
        计算角加速度
        
        Args:
            omega_d: 期望机体角速度 [rad/s]
            current_omega: 当前角速度 [rad/s]
            params: 四旋翼参数
        
        Returns:
            角加速度 [rad/s^2]
        """
        if params is None:
            params = self.default_params()

        # 简单的比例控制
        K_p = jnp.array([22.0, 22.0, 22.0])
        omega_error = omega_d - current_omega

        # 计算期望力矩
        inertia_matrix = jnp.diag(params.inertia)
        desired_torques = inertia_matrix @ (K_p * omega_error)

        # 注释掉陀螺效应 - 避免陀螺耦合导致的持续旋转
        # gyroscopic_torque = jnp.cross(current_omega, inertia_matrix @ current_omega)

        # 总力矩（不使用陀螺效应）
        total_torque = desired_torques  # - gyroscopic_torque

        # 角加速度
        angular_acceleration = jnp.linalg.solve(inertia_matrix, total_torque)

        return angular_acceleration

    def step(self, state: SimpleQuadrotorState, f_d: jnp.ndarray, omega_d: jnp.ndarray, dt: jnp.ndarray, params: SimpleQuadrotorParams = None) -> SimpleQuadrotorState:
        """
        四旋翼动力学前向一步
        
        Args:
            state: 当前状态
            f_d: 期望总推力 [N]
            omega_d: 期望机体角速度 [rad/s]
            dt: 时间步长 [s]
            params: 四旋翼参数
        
        Returns:
            新状态
        """
        if params is None:
            params = self.default_params()

        # 计算加速度
        linear_acceleration = self.linear_dynamics(f_d, state.R, params)
        angular_acceleration = self.angular_dynamics(omega_d, state.omega, params)

        # 积分动力学
        # 位置更新（使用Verlet积分以获得更好的稳定性）
        p_new = state.p + dt * state.v + 0.5 * dt**2 * linear_acceleration
        v_new = state.v + dt * linear_acceleration

        # 姿态更新（使用旋转矩阵）
        R_delta = rotation_matrix_from_vector(dt * state.omega)
        R_new = state.R @ R_delta

        # 角速度更新
        omega_new = state.omega + dt * angular_acceleration

        # 更新随机键
        key_next = jax.random.split(state.dr_key)[0]

        return SimpleQuadrotorState(
            p=p_new,
            R=R_new,
            v=v_new,
            omega=omega_new,
            acc=linear_acceleration,
            dr_key=key_next
        )

    def get_hovering_thrust(self, params: SimpleQuadrotorParams = None) -> jnp.ndarray:
        """获取悬停所需的推力"""
        if params is None:
            params = self.default_params()
        return params.mass * params.gravity

    def get_thrust_to_weight_ratio(self, params: SimpleQuadrotorParams = None) -> jnp.ndarray:
        """获取最大推重比"""
        if params is None:
            params = self.default_params()
        return 4 * params.thrust_max / (params.mass * params.gravity)

    @staticmethod
    def randomize_params(base_params: SimpleQuadrotorParams, key: chex.PRNGKey) -> SimpleQuadrotorParams:
        """
        生成随机化的四旋翼参数（JIT兼容）
        
        Args:
            base_params: 基础参数
            key: JAX随机键
        
        Returns:
            随机化的参数
        """
        key_mass, key_inertia, key_thrust, key_omega = jax.random.split(key, 4)

        # 随机化质量：±10%
        mass_multiplier = jax.random.uniform(key_mass, minval=0.9, maxval=1.1)
        mass = base_params.mass * mass_multiplier

        # 随机化惯性：±10%
        inertia_multiplier = jax.random.uniform(key_inertia, shape=(3,), minval=0.9, maxval=1.1)
        inertia = base_params.inertia * inertia_multiplier

        # 随机化推力：推重比在1.5到4之间
        thrust_to_weight_ratio = jax.random.uniform(key_thrust, minval=1.5, maxval=3.0)
        thrust_max = (thrust_to_weight_ratio * mass * base_params.gravity) / 4.0

        # 随机化角速度：±40%
        omega_multiplier = jax.random.uniform(key_omega, shape=(3,), minval=0.6, maxval=1.4)
        omega_max = base_params.omega_max * omega_multiplier

        return SimpleQuadrotorParams(
            mass=mass,
            inertia=inertia,
            arm_length=base_params.arm_length,
            motor_tau=base_params.motor_tau,
            thrust_max=thrust_max,
            omega_max=omega_max,
            kappa=base_params.kappa,
            gravity=base_params.gravity
        )


# 示例用法和测试
if __name__ == "__main__":
    # 测试四旋翼实现
    print("Testing QuadrotorSimple implementation...")

    # 创建四旋翼
    quadrotor = QuadrotorSimple()

    # 创建初始状态（悬停）
    state = quadrotor.default_state()
    print(f"Initial state position: {state.p}")

    # 创建测试动作（轻微向上推力，无旋转）
    f_d = jnp.array(1.2 * 9.81)  # 120%悬停推力
    omega_d = jnp.array([0.0, 0.0, 0.0])  # 无角速度命令
    print(f"Thrust: {f_d}")
    print(f"Angular velocity: {omega_d}")

    # 前向一步
    dt = jnp.array(0.02)
    new_state = quadrotor.step(state, f_d, omega_d, dt)
    print(f"New state position: {new_state.p}")
    print(f"New state velocity: {new_state.v}")

    # 测试物理属性
    print(f"Hovering thrust: {float(quadrotor.get_hovering_thrust()):.2f} N")
    print(f"Thrust-to-weight ratio: {float(quadrotor.get_thrust_to_weight_ratio()):.2f}")

    # 测试参数随机化
    key = jax.random.key(42)
    params = quadrotor.default_params()
    random_params = QuadrotorSimple.randomize_params(params, key)
    print(f"Randomized mass: {random_params.mass:.3f} kg")
    print(f"Randomized thrust_max: {random_params.thrust_max:.2f} N")

    print("QuadrotorSimple implementation completed successfully!")