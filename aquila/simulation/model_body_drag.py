from typing import NamedTuple

import jax
import jax.numpy as jnp


class BodyDragParams(NamedTuple):
    horizontal_drag_coefficient: float
    vertical_drag_coefficient: float
    frontarea_x: float
    frontarea_y: float
    frontarea_z: float
    air_density: float
    wind_disturbance: jax.Array = jnp.zeros(3)  # 添加风扰动向量
    use_fixed_disturbances: bool = False
    disturbance_mag: float = 0.0


def compute_drag_force(state, key, params: BodyDragParams) -> jnp.ndarray:
    # 分割随机数key
    #key = jax.random.key(0) # 固定随机数种子
    key_drag, key_disturb, key_choice = jax.random.split(key, 3)
    
    # unpack state
    v = state.v
    R = state.R

    # compute drag force
    v_body = R.T @ v
    rho = params.air_density
    area = jnp.array(
        [params.frontarea_x, params.frontarea_y, params.frontarea_z]
    )
    drag_coeff = jnp.array(
        [
            params.horizontal_drag_coefficient,
            params.horizontal_drag_coefficient,
            params.vertical_drag_coefficient,
        ]
    )

    # domain randomization for drag coefficients
    drag_coeff = jax.random.uniform(
        key_drag, drag_coeff.shape, minval=0.5 * drag_coeff, maxval=1.5 * drag_coeff
    )
    
    # 预定义7个方向（世界坐标），单位化
    dirs = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, -1.0, 0.0],
        [-1.0, 0.0, -1.0],
        [0.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ], dtype=jnp.float32)
    norms = jnp.linalg.norm(dirs, axis=1, keepdims=True)
    fixed_set = dirs / jnp.maximum(norms, 1e-8)

    def sample_fixed():
        idx = jax.random.randint(key_choice, (), 0, fixed_set.shape[0])
        vec = fixed_set[idx] * jnp.asarray(params.disturbance_mag, dtype=jnp.float32)
        return vec

    def sample_random():
        rnd = jax.random.uniform(key_disturb, shape=(3,), minval=-1.0, maxval=1.0)
        nrm = jnp.maximum(jnp.linalg.norm(rnd), 1.0)
        # 修复：乘以 disturbance_mag，这样当其为0时扰动真正为0
        return (rnd / nrm) * jnp.asarray(params.disturbance_mag, dtype=jnp.float32)

    disturb_world = jax.lax.cond(
        params.use_fixed_disturbances,
        lambda _: sample_fixed(),
        lambda _: sample_random(),
        operand=None,
    )
    #disturb_world = jnp.array([0.0, 2.0, 0.0])
    # 计算总的阻力
    # 1. 二次阻力（体坐标）
    #quadratic_drag = -0.5 * rho * drag_coeff * area * v_body * jnp.abs(v_body)
    quadratic_drag_xy = -(0.195 + 0.0065 * jnp.linalg.norm(v_body[0:2])) * v_body[0:2]
    alpha = 0.1  # 过渡宽度，可调， 在 0 附近用 tanh 平滑从 0.75 过渡到 0.2
    coef_z = 0.5 * ((0.75 + 0.2) + (0.75 - 0.2) * jnp.tanh(-v_body[2] / alpha))
    quadratic_drag_z = -coef_z * v_body[2:]
    quadratic_drag = jnp.concatenate([quadratic_drag_xy, quadratic_drag_z], axis=0)
    # 2. 常值扰动（转到体坐标）
    constant_drag = R.T @ disturb_world
    #jax.debug.print("  disturb_world: {}", disturb_world)
    # 合并阻力（体坐标）
    f_drag = quadratic_drag + constant_drag
    #jax.debug.print("  quadratic_drag: {}", quadratic_drag)
    #jax.debug.print("  constant_drag: {}", constant_drag)
    return f_drag