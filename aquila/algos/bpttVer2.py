from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from flax.struct import PyTreeNode
from flax.training.train_state import TrainState

from aquila.envs.env_base import Env, EnvState
from aquila.envs.wrappers import LogWrapper, VecEnv


class TrajectoryState(PyTreeNode):
    reward: jnp.array


def progress_callback_host(episode_loss):
    episode, loss = episode_loss
    print(f"Episode: {episode}, Loss: {loss:.2f}")
    # 使用全局变量访问 writer（不作为参数传递，避免 JAX 类型错误）
    global _TENSORBOARD_WRITER
    if _TENSORBOARD_WRITER is not None:
        _TENSORBOARD_WRITER.add_scalar('Loss/train', float(loss), int(episode))


NUM_EPOCHS_PER_CALLBACK = 10

# Global variable to store tensorboard writer
_TENSORBOARD_WRITER = None


def set_tensorboard_writer(writer):
    """Set the global tensorboard writer"""
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = writer


def progress_callback(episode, loss):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            progress_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, loss),
    )


def grad_callback_host(episode_grad):
    episode, grad = episode_grad
    print(f"Episode: {episode}, Grad max: {grad:.4f}")
    # 使用全局变量访问 writer（不作为参数传递，避免 JAX 类型错误）
    global _TENSORBOARD_WRITER
    if _TENSORBOARD_WRITER is not None:
        _TENSORBOARD_WRITER.add_scalar('Gradient/max_norm', float(grad), int(episode))


def grad_callback(episode, grad_norm):
    jax.lax.cond(
        pred=episode % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda eps_lss: jax.debug.callback(
            grad_callback_host, eps_lss
        ),
        false_fun=lambda eps_lss: None,
        operand=(episode, grad_norm),
    )


class RunnerState(NamedTuple):
    train_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int
    # Ver3新增：存储当前动作、动作计数器和动作-状态缓冲区
    current_action: jax.Array  # 当前正在执行的动作
    action_counter: jax.Array  # 动作计数器，记录当前动作已执行了多少步
    action_obs_buffer: jax.Array  # 动作-状态缓冲区，形状为 (num_envs, buffer_size, obs_dim + action_dim)


def train(
    env: Env,
    train_state: TrainState,
    num_epochs: int,
    num_steps_per_epoch: int,
    num_envs: int,
    key: chex.PRNGKey,
    truncate_k: int = 0,
    action_repeat: int = 10,  # 动作重复次数，默认10步
    buffer_size: int = 10,  # Ver3新增：动作-状态缓冲区大小，默认10个观测
):
    """
    Ver2: 支持双输出网络（Ver14支持）
    - 网络输出必须是 tuple (action, aux_output)，其中 action 是 (4,)，aux_output 是 (3,)
    - 直接将 (action, aux_output) 传递给环境
    """

    env = LogWrapper(env)
    env = VecEnv(env)

    trunc_k = int(truncate_k)
    action_repeat_k = int(action_repeat)
    buffer_size_k = int(buffer_size)

    def _train(runner_state: RunnerState):
        def epoch_fn(epoch_state: RunnerState, _unused):
            # Reset env at the start of each epoch to match single-epoch training behavior
            key_epoch, key_reset_base = jax.random.split(epoch_state.key)
            key_reset = jax.random.split(key_reset_base, num_envs)
            env_state, obs = env.reset(key_reset, None)
            
            # Ver3修改：初始化动作、计数器和动作-状态缓冲区
            obs_dim = obs.shape[1]
            action_dim = 4
            
            # 获取原始环境并归一化悬停动作
            # ⚠️ 修复：使用每个环境实际的thrust_max和omega_max（参数随机化后的值），而不是默认值
            original_env = env.unwrapped
            hovering_action_raw = original_env.hovering_action
            
            # 从env_state中获取每个环境实际的quad_params（支持参数随机化）
            # env_state是LogEnvState，包含env_state字段，由于VecEnv，env_state是向量化的TrackStateVer5
            actual_states = env_state.env_state  # 形状: (num_envs, ...)
            actual_thrust_max = actual_states.quad_params.thrust_max  # 形状: (num_envs,)
            actual_omega_max = actual_states.quad_params.omega_max  # 形状: (num_envs, 3) 或 (num_envs,) 取决于定义
            
            # 确保维度正确
            if actual_omega_max.ndim == 2:
                # 如果omega_max是每个轴的，取第一个轴的值用于归一化
                actual_omega_max_scalar = actual_omega_max[:, 0]
            else:
                actual_omega_max_scalar = actual_omega_max
            
            # 为每个环境分别计算归一化的悬停动作
            # hovering_action_raw是标量（所有环境相同），但thrust_max和omega_max是向量化的
            action_low_thrust = original_env.thrust_min * 4  # 标量，所有环境相同
            action_high_thrust = actual_thrust_max * 4  # 形状: (num_envs,)
            action_low_omega = -actual_omega_max_scalar  # 形状: (num_envs,)
            action_high_omega = actual_omega_max_scalar  # 形状: (num_envs,)
            
            # 归一化悬停动作的推力分量
            hovering_thrust_raw = hovering_action_raw[0]  # 标量
            hovering_thrust_normalized = 2.0 * (hovering_thrust_raw - action_low_thrust) / (action_high_thrust - action_low_thrust) - 1.0
            # 归一化悬停动作的角速度分量（都是0）
            hovering_omega_normalized = jnp.zeros((num_envs, 3))
            
            # 组合归一化的悬停动作
            hovering_action_normalized = jnp.concatenate([
                hovering_thrust_normalized[:, None],  # (num_envs, 1)
                hovering_omega_normalized  # (num_envs, 3)
            ], axis=1)  # 形状: (num_envs, 4)
            
            # 初始化缓冲区，使用归一化的悬停动作和零状态向量
            # hovering_action_normalized已经是(num_envs, 4)形状，不需要tile
            zero_obs = jnp.zeros_like(obs)  # 使用零向量代替实际观测
            action_obs_combined = jnp.concatenate([hovering_action_normalized, zero_obs], axis=1)
            action_obs_buffer_init = jnp.tile(action_obs_combined[:, None, :], (1, buffer_size_k, 1))
            
            # 在epoch开始时获取第一个动作（使用填充的缓冲区）
            # 将动作-状态缓冲区展平作为输入
            action_obs_buffer_flat = action_obs_buffer_init.reshape(num_envs, -1)
            # 网络输出7维：前4维是action，后3维是aux_output
            initial_output_7d = epoch_state.train_state.apply_fn(epoch_state.train_state.params, action_obs_buffer_flat)
            initial_action = initial_output_7d[:, :4]  # 前4维：动作
            
            epoch_state = epoch_state._replace(
                env_state=env_state, 
                last_obs=obs, 
                key=key_epoch,
                current_action=initial_action,
                action_counter=jnp.zeros(num_envs, dtype=jnp.int32),  # 初始化为0
                action_obs_buffer=action_obs_buffer_init
            )

            @partial(jax.value_and_grad, has_aux=True)
            def loss_fn(params, runner_state: RunnerState):

                def rollout(runner_state: RunnerState):
                    def step_fn(old_runner_state: RunnerState, t):
                        # minimal truncated BPTT: stop gradient every K steps
                        def sg(x):
                            return jax.tree.map(jax.lax.stop_gradient, x)
                        old_runner_state = jax.lax.cond(
                            jnp.logical_and(trunc_k > 1, (t % trunc_k) == 0),
                            lambda _: sg(old_runner_state),
                            lambda _: old_runner_state,
                            operand=None,
                        )

                        # extract states and obs
                        train_state, env_state, last_obs, key, epoch_idx, current_action, action_counter, action_obs_buffer = (
                            old_runner_state
                        )

                        # Ver3核心修改：每action_repeat_k步才获取新动作，并更新动作-状态缓冲区
                        # 判断是否需要获取新动作（对每个环境独立判断）
                        need_new_action = (action_counter % action_repeat_k) == 0
                        
                        # 步骤1：先用空动作+当前观测更新缓冲区（为获取新动作做准备）
                        action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=1)
                        # 创建空动作（4维零向量）
                        empty_action = jnp.zeros((num_envs, 4))
                        # 将空动作和观测拼接：[empty_action, obs]
                        action_obs_combined_empty = jnp.concatenate([empty_action, last_obs], axis=1)
                        action_obs_buffer_for_input = action_obs_buffer_for_input.at[:, -1, :].set(action_obs_combined_empty)
                        
                        # 步骤2：使用更新后的缓冲区获取新动作
                        action_obs_buffer_flat = action_obs_buffer_for_input.reshape(num_envs, -1)
                        # 网络输出7维：前4维是action，后3维是aux_output
                        output_7d = train_state.apply_fn(params, action_obs_buffer_flat)
                        new_action = output_7d[:, :4]  # 前4维：动作
                        new_aux_output = output_7d[:, 4:]  # 后3维：辅助输出
                        
                        new_counter = jnp.ones(num_envs, dtype=jnp.int32)  # 重置为1而不是0
                        
                        # 步骤3：用获取到的新动作更新缓冲区（用于下次使用）
                        action_obs_buffer_updated = jnp.roll(action_obs_buffer, shift=-1, axis=1)
                        # 将新动作和观测拼接：[new_action, obs]
                        action_obs_combined_new = jnp.concatenate([new_action, last_obs], axis=1)
                        action_obs_buffer_updated = action_obs_buffer_updated.at[:, -1, :].set(action_obs_combined_new)
                        
                        # 使用where选择缓冲区：只有在需要新动作时才更新缓冲区
                        obs_buffer_to_use = jnp.where(
                            need_new_action[:, None, None],  # 扩展维度以匹配缓冲区维度
                            action_obs_buffer_updated,
                            action_obs_buffer
                        )
                        
                        # 使用where选择动作：需要新动作时使用new_action，否则保持current_action
                        action = jnp.where(
                            need_new_action[:, None],  # 扩展维度以匹配action维度
                            new_action,
                            current_action
                        )
                        
                        # Ver2: 处理辅助输出并传递给环境
                        # 如果需要新动作，则使用新的辅助输出，否则保持上一次的辅助输出
                        aux_output = jnp.where(
                            need_new_action[:, None],
                            new_aux_output,
                            jnp.zeros_like(new_aux_output)  # 如果不需要新动作，使用零向量
                        )
                        # 传递tuple格式给环境: (action, aux_output)
                        action_for_env = (action, aux_output)
                        
                        # 使用where选择计数器：需要新动作时重置为0，否则+1
                        new_action_counter = jnp.where(
                            need_new_action,
                            new_counter,
                            action_counter + 1
                        )

                        # env step
                        key, key_ = jax.random.split(key)
                        key_step = jax.random.split(key_, num_envs)
                        (
                            env_state,
                            obs,
                            reward,
                            _terminated,
                            _truncated,
                            info,
                        ) = env.step(env_state, action_for_env, key_step)
                        
                        runner_state = RunnerState(
                            train_state, env_state, obs, key, epoch_idx, action, new_action_counter, obs_buffer_to_use
                        )

                        return (
                            runner_state,
                            TrajectoryState(reward=reward),
                        )

                    ts = jnp.arange(num_steps_per_epoch)
                    runner_state, trajectory = jax.lax.scan(
                        step_fn, runner_state, ts
                    )
                    return runner_state, trajectory

                # collect data
                runner_state, trajectory = rollout(runner_state)
                loss = -trajectory.reward.sum() / (num_envs * num_steps_per_epoch)
                return loss, runner_state

            # compute reward
            train_state = epoch_state.train_state
            (loss, epoch_state), grad = loss_fn(
                train_state.params, epoch_state
            )
            # update params
            train_state = train_state.apply_gradients(grads=grad)

            # calc stats on grad
            leaves = jax.tree_util.tree_leaves(grad)
            flattened_leaves = [jnp.ravel(leaf) for leaf in leaves]
            grad_vec = jnp.concatenate(flattened_leaves)
            grad_max = jnp.max(jnp.abs(grad_vec))

            progress_callback(epoch_state.epoch_idx, loss)
            grad_callback(epoch_state.epoch_idx, grad_max)
            epoch_state = epoch_state._replace(
                train_state=train_state, epoch_idx=epoch_state.epoch_idx + 1
            )

            return epoch_state, loss

        # run epochs
        runner_state_final, losses = jax.lax.scan(
            epoch_fn, runner_state, None, num_epochs
        )

        return {"runner_state": runner_state_final, "metrics": losses}

    # intialize environments
    print(f"[bpttVer2.train] 正在初始化 {num_envs} 个并行环境...")
    print(f"[bpttVer2.train] 动作重复设置: 每 {action_repeat} 步获取一次新动作")
    print(f"[bpttVer2.train] 动作-状态缓冲区大小: {buffer_size}")
    print(f"[bpttVer2.train] 支持双输出网络（Ver14）：动作 + 辅助输出")
    import sys
    sys.stdout.flush()
    
    key, key_ = jax.random.split(key)
    key_reset = jax.random.split(key_, num_envs)
    env_state, obs = env.reset(key_reset, None)
    
    # Ver3修改：初始化RunnerState时需要包含动作-状态缓冲区
    obs_dim = obs.shape[1]
    action_dim = 4
    
    # 获取原始环境并归一化悬停动作
    # ⚠️ 修复：使用每个环境实际的thrust_max和omega_max（参数随机化后的值），而不是默认值
    original_env = env.unwrapped
    hovering_action_raw = original_env.hovering_action
    
    # 从env_state中获取每个环境实际的quad_params（支持参数随机化）
    actual_states = env_state.env_state  # 形状: (num_envs, ...)
    actual_thrust_max = actual_states.quad_params.thrust_max  # 形状: (num_envs,)
    actual_omega_max = actual_states.quad_params.omega_max  # 形状: (num_envs, 3) 或 (num_envs,)
    
    # 确保维度正确
    if actual_omega_max.ndim == 2:
        actual_omega_max_scalar = actual_omega_max[:, 0]
    else:
        actual_omega_max_scalar = actual_omega_max
    
    # 为每个环境分别计算归一化的悬停动作
    action_low_thrust = original_env.thrust_min * 4
    action_high_thrust = actual_thrust_max * 4  # 形状: (num_envs,)
    
    hovering_thrust_raw = hovering_action_raw[0]
    hovering_thrust_normalized = 2.0 * (hovering_thrust_raw - action_low_thrust) / (action_high_thrust - action_low_thrust) - 1.0
    hovering_omega_normalized = jnp.zeros((num_envs, 3))
    
    hovering_action_normalized = jnp.concatenate([
        hovering_thrust_normalized[:, None],
        hovering_omega_normalized
    ], axis=1)  # 形状: (num_envs, 4)
    
    # 创建初始的动作-状态组合：[action, zero_obs]
    zero_obs = jnp.zeros_like(obs)  # 使用零向量代替实际观测
    action_obs_combined = jnp.concatenate([hovering_action_normalized, zero_obs], axis=1)
    
    # 初始化缓冲区，用悬停动作和零状态向量填充
    action_obs_buffer_init = jnp.tile(action_obs_combined[:, None, :], (1, buffer_size, 1))
    
    # 获取初始动作（使用填充的缓冲区）
    action_obs_buffer_flat = action_obs_buffer_init.reshape(num_envs, -1)
    # 网络输出7维：前4维是action，后3维是aux_output
    initial_output_7d = train_state.apply_fn(train_state.params, action_obs_buffer_flat)
    initial_action = initial_output_7d[:, :4]  # 前4维：动作（只取action部分用于初始化）
    
    runner_state = RunnerState(
        train_state, env_state, obs, key, epoch_idx=0,
        current_action=initial_action,
        action_counter=jnp.zeros(num_envs, dtype=jnp.int32),
        action_obs_buffer=action_obs_buffer_init
    )
    
    print(f"[bpttVer2.train] 环境初始化完成，观测形状: {obs.shape}")
    print(f"[bpttVer2.train] 动作-状态缓冲区形状: {action_obs_buffer_init.shape}")
    print(f"[bpttVer2.train] 展平后输入形状: {action_obs_buffer_flat.shape}")
    print(f"[bpttVer2.train] 初始动作形状: {initial_action.shape}")
    print(f"[bpttVer2.train] 开始 JIT 编译训练函数...")
    sys.stdout.flush()

    result = jax.jit(_train)(runner_state)
    
    print(f"[bpttVer2.train] JIT 编译完成，训练已结束")
    sys.stdout.flush()
    
    return result

