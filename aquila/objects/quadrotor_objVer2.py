import os
from functools import partial

import jax
import jax_dataclasses as jdc
import numpy as np
import yaml
from aquila.utils.pytrees import field_jnp, CustomPyTree
from aquila.utils.math import lead_lag_discrete
from jax import numpy as jnp
from aquila.simulation.model_body_drag import (
    BodyDragParams,
    compute_drag_force,
)
from aquila.utils.math import rotation_matrix_from_vector

import chex


@jdc.pytree_dataclass
class QuadrotorParams:
    """可变的quadrotor参数Ver2，用于参数随机化（与JIT兼容）"""
    thrust_max: float = 9.0  # [N] per motor
    omega_max: jax.Array = field_jnp([5.0, 5.0, 5.0])  # [rad/s]
    motor_tau: float = 0.04  # [s]
    Kp: jax.Array = field_jnp([20.0, 20.0, 10.0])  # PID proportional gains Ver2


@jdc.pytree_dataclass
class QuadrotorState(CustomPyTree):
    p: jax.Array = field_jnp([0.0, 0.0, 0.0])
    R: jax.Array = field_jnp(jnp.eye(3))
    v: jax.Array = field_jnp([0.0, 0.0, 0.0])
    omega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    domega: jax.Array = field_jnp([0.0, 0.0, 0.0])
    motor_omega: jax.Array = field_jnp([0.0, 0.0, 0.0, 0.0])
    acc: jax.Array = field_jnp([0.0, 0.0, 0.0])
    dr_key: chex.PRNGKey = field_jnp(jax.random.key(0))
    fix_key: chex.PRNGKey = field_jnp(jax.random.key(0))
    # PID控制状态
    omega_err_integral: jax.Array = field_jnp([0.0, 0.0, 0.0])  # 积分项
    last_omega_err: jax.Array = field_jnp([0.0, 0.0, 0.0])  # 上次误差（用于微分项）

    def detached(self):
        return QuadrotorState(
            p=jax.lax.stop_gradient(self.p),
            R=jax.lax.stop_gradient(self.R),
            v=jax.lax.stop_gradient(self.v),
            omega=jax.lax.stop_gradient(self.omega),
            domega=jax.lax.stop_gradient(self.domega),
            motor_omega=jax.lax.stop_gradient(self.motor_omega),
            acc=jax.lax.stop_gradient(self.acc),
            dr_key=jax.lax.stop_gradient(self.dr_key),
            fix_key=jax.lax.stop_gradient(self.fix_key),
            omega_err_integral=jax.lax.stop_gradient(self.omega_err_integral),
            last_omega_err=jax.lax.stop_gradient(self.last_omega_err)
        )

    def as_vector(self):
        return jnp.concatenate(
            [self.p, self.R.flatten(), self.v, self.omega, self.domega,
             self.motor_omega]
        )

    @classmethod
    def from_vector(cls, vector):
        p = vector[:3]
        R = vector[3:12].reshape(3, 3)
        v = vector[12:15]
        omega = vector[15:18]
        domega = vector[18:21]
        motor_omega = vector[21:]
        return cls(p, R, v, omega, domega, motor_omega)


class QuadrotorVer2:
    """
    Full quadrotor model Ver2 based on agilicious framework.
    Recommendation: Use from_yaml or from_name to create a quadrotor object.
    Note, the thrust map and drag augmentation is only valid for Kolibri
    Ver2: Added Kp randomization support in QuadrotorParams

    >>> quad = Quadrotor.from_name("kolibri")
    >>> state = quad.default_state()
    >>> state_new = quad.step(state, 9.81 * quad._mass,
    >>>                       jnp.array([0.0, 0.0, 0.0]), 0.01)
    """

    def __init__(
            self,
            *,
            mass=0.248,  # [kg]
            tbm_fr=jnp.array([0.0825, -0.0825, 0.0]),  # [m]
            tbm_bl=jnp.array([-0.0825, 0.0825, 0.0]),  # [m]
            tbm_br=jnp.array([-0.0825, -0.0825, 0.0]),  # [m]
            tbm_fl=jnp.array([0.0825, 0.0825, 0.0]),  # [m]
            inertia=jnp.array([0.00026174, 0.00027494, 0.00037511]),  # [kgm^2]
            motor_omega_min=100.0,  # [rad/s]
            motor_omega_max=5400.0,  # [rad/s]
            motor_tau=0.015,  # [s]
            motor_inertia=2.6e-7,  # [kgm^2]
            omega_max=jnp.array([1.0, 1.0, 1.0]),  # [rad/s]
            thrust_map=jnp.array([2.0e-7, 0.0, 0.0]),
            kappa=0.008,  # [Nm/N]
            thrust_min=0.0,  # [N]
            thrust_max=2.5,  # [N] per motor
            rotors_config="cross",
            dt_low_level=0.001,
            disturbance_mag=0.0,  # [N] 常值扰动力的大小（训练时可设置>0，测试时设置为0）
    ):
        assert (
                rotors_config == "cross"
        ), "Only cross rotors configuration is supported"
        self._mass = mass
        self._tbm_fr = tbm_fr
        self._tbm_bl = tbm_bl
        self._tbm_br = tbm_br
        self._tbm_fl = tbm_fl
        self._inertia = inertia
        self._motor_omega_min = motor_omega_min
        self._motor_omega_max = motor_omega_max
        self._motor_tau = motor_tau
        self._motor_inertia = motor_inertia
        self._omega_max = omega_max
        self._thrust_map = thrust_map
        self._kappa = kappa
        self._thrust_min = thrust_min
        if thrust_min <= 0.0:
            self._thrust_min += thrust_map[0] * motor_omega_min ** 2
        self._thrust_max = thrust_max
        self._rotors_config = rotors_config
        self._dt_low_level = dt_low_level
        self._gravity = jnp.array([0, 0, 9.81]) # 向下为正
        self._disturbance_mag = disturbance_mag  # 保存扰动力大小参数

        # drag parameters
        self._drag_params = BodyDragParams(
            horizontal_drag_coefficient=1.04,
            vertical_drag_coefficient=1.04,
            frontarea_x=1.0e-3,
            frontarea_y=1.0e-3,
            frontarea_z=1.0e-2,
            air_density=1.2,
            disturbance_mag=disturbance_mag,  # 使用参数设置扰动力大小
        )


    @property
    def hovering_motor_speed(self) -> float:
        return jnp.sqrt(self._mass * 9.81 / (4 * self._thrust_map[0]))

    def default_state(self):
        hovering_motor_speeds = jnp.ones(4) * self.hovering_motor_speed
        return QuadrotorState(motor_omega=hovering_motor_speeds)

    def create_state(self, p, R, v, **kwargs):
        hovering_motor_speed = jnp.ones(4) * self.hovering_motor_speed
        if "motor_omega" not in kwargs.keys():
            kwargs["motor_omega"] = hovering_motor_speed

        return QuadrotorState(p, R, v, **kwargs)

    @property
    def allocation_matrix(self):
        """
        maps [f1, f2, f3, f4] to [f_T, tau_x, tau_y, tau_z]
        """

        rotor_coordinates = np.stack(
            [self._tbm_fr, self._tbm_bl, self._tbm_br, self._tbm_fl]
        )
        x = rotor_coordinates[:, 0]
        y = rotor_coordinates[:, 1]

        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                y,
                -x,
                self._kappa * np.array([-1.0, -1.0, 1.0, 1.0]),
            ],
            dtype=np.float32,
        )

    def inertial_matrix(self):
        return np.diag(self._inertia)

    @property
    def allocation_matrix_inv(self):
        return np.linalg.inv(self.allocation_matrix)

    def step(
        self,
        state: QuadrotorState,
        f_d: jax.Array,
        omega_d: jax.Array,
        dt: jax.Array,
        drag_params=None,
        quad_params: QuadrotorParams = None
    ) -> QuadrotorState:
        """Step quadrotor dynamics forward.

        Args:
            state: Current state
            f_d: Desired collective thrust [N]
            omega_d: Desired body rates [rad/s]
            dt: Time step [s]
            drag_params: Optional drag parameters to override default ones
            quad_params: Optional QuadrotorParams for randomized parameters

        Returns:
            New state
        """
        # 如果没有提供drag_params，使用默认的
        drag_params = drag_params or self._drag_params
        # 如果没有提供quad_params，使用默认的
        quad_params = quad_params or self.default_params()

        # @partial(jax.custom_jvp, nondiff_argnums=(3,))
        def _step(state, f_d, omega_d, dt, quad_params):
            """Forward pass of the quadrotor dynamics."""

            # round dt to 5 decimal places to avoid numerical issues
            dt = np.round(dt, 5)
            if dt <= 0.0:
                return state

            def control_fn(state, _unused):
                """
                Low-level controller and dynamics.
                Runs by default at 1 kHz.
                """

                motor_omega_d, omega_err_integral_new, omega_err_new = self._low_level_controller(
                    state, f_d, omega_d, quad_params
                )

                state = self._dynamics(
                    state, motor_omega_d, self._dt_low_level, drag_params, quad_params
                )
                
                # 更新PID状态
                state = QuadrotorState(
                    p=state.p,
                    R=state.R,
                    v=state.v,
                    omega=state.omega,
                    domega=state.domega,
                    motor_omega=state.motor_omega,
                    acc=state.acc,
                    dr_key=state.dr_key,
                    fix_key=state.fix_key,
                    omega_err_integral=omega_err_integral_new,
                    last_omega_err=omega_err_new
                )
                
                return state, None

            N = np.ceil(dt / self._dt_low_level).item()
            # check if dt is a multiple of dt_low_level
            assert np.isclose(
                N * self._dt_low_level, dt
            ), f"dt ({dt}) must be a multiple of dt_low_level ({self._dt_low_level})"

            state_new, _ = jax.lax.scan(control_fn, state, length=N)

            return state_new
        return _step(state, f_d, omega_d, dt, quad_params)

    def _dynamics(self, state: QuadrotorState, motor_omega_d, dt, drag_params=None, quad_params: QuadrotorParams = None):
        """Quadrotor dynamics."""
        # 如果没有提供quad_params，使用默认的
        quad_params = quad_params or self.default_params()
        # unpack state
        p = state.p
        R = state.R
        v = state.v
        omega = state.omega
        motor_omega = state.motor_omega

        # domain randomization keys
        key_thrust, key_acc, key_next = jax.random.split(state.dr_key, 3)
        key_drag = state.fix_key
        # position
        p_new = p + dt * v

        # orientation
        R_delta = rotation_matrix_from_vector(dt * omega)
        R_new = R @ R_delta

        # velocity
        # motor thrust
        thrust_map = self._thrust_map[0]
        thrust_map = jax.random.uniform(
            key_thrust,
            thrust_map.shape,
            minval=0.95 * thrust_map,
            maxval=1.05 * thrust_map,
        )

        f = thrust_map * motor_omega ** 2

        # Quadratic drag model
        f_drag = compute_drag_force(state, key_drag, drag_params or self._drag_params)

        f_vec = jnp.array([0, 0, -jnp.sum(f)]) + f_drag # 向下为正
        # Minimal per-step Gaussian acceleration disturbance (fixed std here)
        acc = self._gravity + R @ f_vec / self._mass + 0.0 * jax.random.normal(key_acc, (3,))
        v_new = v + dt * acc

        # angular acceleration - use motor_tau from quad_params
        dmotor_omega = 1 / quad_params.motor_tau * (motor_omega_d - motor_omega)
        motor_directions = jnp.array([-1, -1, 1, 1])
        motor_inertia = self._motor_inertia
        inertia_torque = jnp.array(
            [0, 0, (dmotor_omega * motor_directions).sum() * motor_inertia]
        )

        # body torques and collective thrust
        J = self.inertial_matrix()
        J_inv = np.linalg.inv(J)
        f_T_and_tau = self.allocation_matrix @ f
        f_T, tau = f_T_and_tau[0], f_T_and_tau[1:]
        domega_new = J_inv @ (
                tau - jnp.cross(omega, J @ omega) + inertia_torque
        )
        omega_new = omega + dt * domega_new

        # motor dynamics
        motor_omega_new = motor_omega + dt * dmotor_omega
        '''
        # 整合所有状态信息到一行
        has_nan = (jnp.any(jnp.isnan(p_new)) | 
                  jnp.any(jnp.isnan(v_new)) |
                  jnp.any(jnp.isnan(acc)) |
                  jnp.any(jnp.isnan(omega_new)) |
                  jnp.any(jnp.isnan(motor_omega_new)) |
                  jnp.any(jnp.isnan(f)) |
                  jnp.any(jnp.isnan(f_drag)))
        jax.debug.print(
            "State check [NaN={}, p={}, v={}, acc={}, omega={}, motor={}, f={}, drag={}]\n",
            has_nan, p_new, v_new, acc, omega_new, motor_omega_new, f, f_drag
        )
        '''

        return QuadrotorState(
            p=p_new,
            R=R_new,
            v=v_new,
            omega=omega_new,
            domega=domega_new,
            motor_omega=motor_omega_new,
            acc=acc,
            dr_key=key_next,
            fix_key=state.fix_key,
            omega_err_integral=state.omega_err_integral,  # 保留PID状态，将在control_fn中更新
            last_omega_err=state.last_omega_err
        )

    def motor_omega_to_thrust(self, motor_omega):
        return self._thrust_map[0] * motor_omega ** 2

    def default_params(self) -> QuadrotorParams:
        """
        Get default quadrotor parameters Ver2.
        
        Returns:
            QuadrotorParams with default values
        """
        return QuadrotorParams(
            thrust_max=self._thrust_max,
            omega_max=self._omega_max,
            motor_tau=self._motor_tau,
            Kp=jnp.array([60.0, 60.0, 30.0])  # PID proportional gains Ver2
        )
    
    @staticmethod
    def randomize_params(
        base_params: QuadrotorParams, 
        mass: float, 
        key: chex.PRNGKey,
        thrust_to_weight_min: float = 1.5,
        thrust_to_weight_max: float = 4.0
    ) -> QuadrotorParams:
        """
        Generate randomized quadrotor parameters Ver2 (JIT-compatible).
        
        Args:
            base_params: Base parameters to randomize from
            mass: Quadrotor mass [kg] (fixed, not randomized)
            key: JAX random key for generating random values
            thrust_to_weight_min: Minimum thrust-to-weight ratio for randomization
            thrust_to_weight_max: Maximum thrust-to-weight ratio for randomization
            
        Returns:
            QuadrotorParams with randomized values Ver2:
            - mass: fixed (not randomized)
            - thrust_max: randomized based on thrust-to-weight ratio range
            - omega_max: randomized with ±30% variation around 0.5 rad/s
            - motor_tau: randomized with ±30% variation around base value
            - Kp: randomized with multiplier in range [1.0, 3.0] applied to base [20, 20, 10]
        """
        key_thrust, key_omega, key_tau, key_kp = jax.random.split(key, 4)
        
        # Randomize maximum thrust based on thrust-to-weight ratio
        # Total thrust = 4 * thrust_max (4 motors)
        # Thrust-to-weight ratio = (4 * thrust_max) / (mass * g)
        thrust_to_weight_ratio = jax.random.uniform(
            key_thrust, 
            minval=thrust_to_weight_min, 
            maxval=thrust_to_weight_max
        )
        thrust_max = (thrust_to_weight_ratio * mass * 9.81) / 4.0
        
        # Randomize maximum angular velocity: ±30% variation around 0.5 rad/s
        # For each axis, omega_max in range [0.35, 0.65] rad/s
        omega_base = 1  # rad/s
        omega_max = jax.random.uniform(key_omega, shape=(3,), minval=omega_base * 0.7, maxval=omega_base * 1.3)
        # omega_max = jax.random.uniform(key_omega, shape=(3,), minval=0.49, maxval=0.51)
        
        # Randomize motor_tau: ±30% fluctuation around the base value
        tau_multiplier = jax.random.uniform(key_tau, minval=0.7, maxval=1.3)
        motor_tau = base_params.motor_tau * tau_multiplier
        
        # Randomize Kp Ver2: multiplier in range [1.0, 3.0] applied to base [20, 20, 10]
        # Kp_base = jnp.array([56.0, 56.0, 28.0])
        # Kp_multiplier = jax.random.uniform(key_kp, minval=1.0, maxval=1.1)
        # Kp = Kp_base * Kp_multiplier
        Kp = jnp.array([60.0, 60.0, 30.0])

        return QuadrotorParams(
            thrust_max=thrust_max,
            omega_max=omega_max,
            motor_tau=motor_tau,
            Kp=Kp
        )
    
    def get_thrust_to_weight_ratio_and_omega_max(self, params: QuadrotorParams = None):
        """
        Get thrust-to-weight ratio and maximum angular velocity.
        
        Args:
            params: Optional QuadrotorParams. If None, uses default parameters.
        
        Returns:
            tuple: (thrust_to_weight_ratio, omega_max)
                - thrust_to_weight_ratio: float, thrust-to-weight ratio
                - omega_max: jax.Array, maximum angular velocity [rad/s] for each axis
        """
        if params is None:
            params = self.default_params()
        
        # Total maximum thrust from 4 motors divided by weight
        thrust_to_weight_ratio = (4.0 * params.thrust_max) / (self._mass * 9.81)
        return thrust_to_weight_ratio, params.omega_max

    def _low_level_controller(self, state, f_T, omega_cmd, quad_params: QuadrotorParams = None):
        # 如果没有提供quad_params，使用默认的
        quad_params = quad_params or self.default_params()
        
        # PID控制参数Ver2（使用可随机化的Kp）
        Kp = jnp.diag(quad_params.Kp)  # 比例增益 Ver2 - 从quad_params获取
        Ki = jnp.diag(jnp.array([0, 0, 0]))     # 积分增益
        Kd = jnp.diag(jnp.array([0, 0, 0]))    # 微分增益
        
        omega = state.omega
        omega_err = omega_cmd - omega
        
        # 比例项
        P_term = Kp @ omega_err
        
        # 积分项（累积误差，带积分饱和限制）
        integral_max = jnp.array([2.0, 2.0, 1.0])  # 积分饱和限制
        omega_err_integral_new = state.omega_err_integral + omega_err * self._dt_low_level
        omega_err_integral_new = jnp.clip(omega_err_integral_new, -integral_max, integral_max)
        I_term = Ki @ omega_err_integral_new
        
        # 微分项（误差变化率）
        omega_err_derivative = (omega_err - state.last_omega_err) / self._dt_low_level
        D_term = Kd @ omega_err_derivative
        
        # PID控制输出
        pid_output = P_term + I_term + D_term
        
        # 计算期望的力矩指令（包含角速度交叉项）
        body_torques_cmd = self.inertial_matrix() @ pid_output + jnp.cross(
            omega, self.inertial_matrix() @ omega
        )
        
        alpha = jnp.concatenate([f_T[None], body_torques_cmd])
        f_cmd = self.allocation_matrix_inv @ alpha
        # Use thrust_max from quad_params
        f_cmd = jnp.clip(f_cmd, self._thrust_min, quad_params.thrust_max)
        motor_omega_d = jnp.sqrt(f_cmd / self._thrust_map[0])
        motor_omega_d = jnp.clip(
            motor_omega_d, self._motor_omega_min, self._motor_omega_max
        )
        
        # 返回电机指令和更新后的PID状态
        return motor_omega_d, omega_err_integral_new, omega_err

if __name__ == "__main__":

    quad = Quadrotor(mass=1.0)
    state = quad.default_state()
    f_d = jnp.array(1 * 9.81)
    omega_d = jnp.array([0.0, 0.0, 0.01])
    dt = 1.0
    state_new = quad.step(state, f_d, omega_d, dt)
    print(state_new.p)