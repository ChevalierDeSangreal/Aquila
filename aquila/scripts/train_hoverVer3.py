#!/usr/bin/env python
# coding: utf-8

import os
import sys

# ==================== GPU Configuration ====================
# 必须在导入JAX之前设置CUDA_VISIBLE_DEVICES
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用单张GPU
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import time
import math
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax.training.train_state import TrainState
import pickle
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
# aquila/scripts/train_hoverVer3.py -> ../../ -> Aquila project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.hoverVer3 import HoverEnvVer3  # 使用HoverEnvVer3（基于HoverVer2调整了loss权重，悬停任务，使用Quadrotor不支持PID Kp参数随机化）
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP
from aquila.algos import bptt


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
    else:
        # 兼容旧格式
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
    
    print("✅ Policy parameters loaded successfully!")
    print(f"   Final loss: {final_loss}")
    print(f"   Training epochs: {training_epochs}")
    
    return params, env_config


def main():
    # ==================== GPU Configuration Info ====================
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # ==================== Environment Setup ====================
    # Create env - 使用HoverEnvVer3环境（基于HoverVer2调整了loss权重，悬停任务，使用Quadrotor不支持PID Kp参数随机化）
    env = HoverEnvVer3(
        max_steps_in_episode=1000,  # 悬停任务的最大步数
        dt=0.01,  # 使用完整四旋翼的默认时间步长
        delay=0.03,  # 可选执行延迟
        omega_std=0.1,
        action_penalty_weight=0.1,
        # Hovering specific parameters
        hover_height=2.0,  # m (悬停高度2米)
        init_pos_range=0.5,  # m (初始位置随机化范围，实际在0~0.5m球体内随机)
        max_distance=10.0,  # m (距离原点最大距离阈值，超过则重置)
        max_speed=20.0,  # m/s
        # Parameter randomization (quadrotor)
        thrust_to_weight_min=1.2,  # 最小推重比
        thrust_to_weight_max=5.0,  # 最大推重比
        disturbance_mag=2.0,  # 训练时开启常值随机扰动（2N），提高鲁棒性
    )
    
    # Normalize obs to [-1,1] and actions to [-1,1]
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # ==================== Model Setup ====================
    # 输入维度变为缓冲区大小 * (观测维度 + 动作维度)
    buffer_size = 50  # 动作-状态缓冲区大小
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)  # 输入维度为缓冲区大小乘以(观测维度+动作维度)
    
    # 创建MLP网络，输入维度为buffer_size * (obs_dim + action_dim)
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension (buffer_size * (obs_dim + action_dim)): {input_dim}")
    
    # ==================== Training Parameters ====================
    num_epochs = 4000  # 训练轮数
    num_envs = 512 # 并行环境数（单卡）
    
    # 动作重复参数
    action_repeat = 2  # 每2个step才获取一次新动作（每秒50次动作，每次持续0.02秒）
    
    # Optimizer - 使用余弦衰减学习率
    initial_learning_rate = 5e-3
    end_learning_rate = 5e-4
    scheduler = optax.cosine_decay_schedule(
        init_value=initial_learning_rate,
        decay_steps=num_epochs,
        alpha=end_learning_rate/initial_learning_rate
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # 梯度裁剪
        optax.adam(scheduler)
    )
    
    # Init params
    key = jax.random.key(0)
    init_params = policy.initialize(key)
    
    # ==================== Choose Training Mode ====================
    choice = 1   # 1: 使用初始参数, 2: 使用加载的参数
    
    if choice == 1:
        # 使用初始参数
        train_state = TrainState.create(
            apply_fn=policy.apply,
            params=init_params,
            tx=tx
        )
        print("✅ 使用初始网络参数开始训练")
    else:
        # 使用加载的参数
        policy_file = 'aquila/param/hoverVer3_policy.pkl'  # 使用hoverVer3的模型文件
        loaded_params, env_config = load_trained_policy(policy_file)
        train_state = TrainState.create(
            apply_fn=policy.apply,
            params=loaded_params,
            tx=tx
        )
        print("✅ 使用加载的网络参数继续训练")
    
    # ==================== TensorBoard Setup ====================
    # 创建tensorboard日志目录
    log_dir = f'runs/hoverVer3_{time.strftime("%Y%m%d_%H%M%S")}'  # 使用hoverVer3的日志目录
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir=runs' to view training progress")
    
    # 设置 tensorboard writer 到 bptt 模块（用于实时记录）
    bptt.set_tensorboard_writer(writer)
    
    # ==================== Training ====================
    time_start = time.time()
    training_log = []
    
    print(f"\n{'='*60}")
    print(f"开始训练悬停任务 (HoverVer3 - 基于HoverVer2调整了loss权重，使用Quadrotor不支持PID Kp参数随机化)...")
    print(f"Total environments: {num_envs}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Steps per epoch: {env.max_steps_in_episode}")
    print(f"Action repeat: {action_repeat} steps")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Quadrotor model: Full (Quadrotor - based on agilicious framework, without PID Kp parameter randomization)")
    print(f"{'='*60}\n")
    
    # 使用单卡训练函数
    print("正在调用 bptt.train()...")
    print("注意：首次调用会进行 JIT 编译，可能需要1-2分钟...")
    import sys
    sys.stdout.flush()
    
    res_dict = bptt.train(
        env=env,
        train_state=train_state,
        num_epochs=num_epochs,
        num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=num_envs,
        key=key,
        truncate_k=500,  # 0表示完整BPTT
        action_repeat=action_repeat,  # 传递动作重复参数
        buffer_size=buffer_size,  # 传递动作-状态缓冲区大小参数
    )
    
    print("\n✅ 训练完成！开始处理结果...")
    sys.stdout.flush()
    
    time_end = time.time()
    training_time = time_end - time_start
    
    # ==================== Record Training Results ====================
    # 获取所有epoch的损失
    losses = res_dict["metrics"]  # shape: (num_epochs,)
    losses_np = np.array(losses)
    
    # 补充记录每个epoch的损失到tensorboard（填补实时记录的间隙）
    print("\n补充记录训练数据到 TensorBoard...")
    for epoch_idx in range(num_epochs):
        loss_value = float(losses_np[epoch_idx])
        # 这里会覆盖之前实时记录的值，但没关系，数据是一致的
        writer.add_scalar('Loss/train_complete', loss_value, epoch_idx)
        training_log.append(loss_value)
        
        # 每100个epoch打印一次
        if (epoch_idx + 1) % 100 == 0:
            print(f"Epoch {epoch_idx + 1}/{num_epochs}, Loss: {loss_value:.6f}")
    
    
    # 记录损失统计信息
    writer.add_scalar('Loss/initial', float(losses_np[0]), 0)
    writer.add_scalar('Loss/final', float(losses_np[-1]), 0)
    writer.add_scalar('Loss/min', float(np.min(losses_np)), 0)
    writer.add_scalar('Loss/max', float(np.max(losses_np)), 0)
    writer.add_scalar('Loss/mean', float(np.mean(losses_np)), 0)
    writer.add_scalar('Loss/std', float(np.std(losses_np)), 0)
    
    # 记录动作-状态缓冲区相关的统计信息
    writer.add_scalar('Config/action_repeat', action_repeat, 0)
    writer.add_scalar('Config/action_obs_buffer_size', buffer_size, 0)
    writer.add_scalar('Config/input_dimension', input_dim, 0)
    writer.add_scalar('Config/effective_actions_per_epoch', env.max_steps_in_episode / action_repeat, 0)
    
    # 获取更新后的训练状态
    train_state = res_dict["runner_state"].train_state
    final_loss = float(losses_np[-1])
    
    # ==================== Print Summary ====================
    print(f"\n{'='*60}")
    print(f"悬停任务训练完成！(HoverVer3 - 基于HoverVer2调整了loss权重，使用Quadrotor不支持PID Kp参数随机化)")
    print(f"{'='*60}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Initial Loss: {float(losses_np[0]):.6f}")
    print(f"Loss improvement: {float(losses_np[0] - losses_np[-1]):.6f}")
    print(f"Action repeat: {action_repeat} steps")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Effective actions per epoch: {env.max_steps_in_episode / action_repeat:.1f}")
    print(f"Quadrotor model: Full (Quadrotor - based on agilicious framework, without PID Kp parameter randomization)")
    
    
    
    # ==================== Save Model ====================
    # 保存网络参数和训练信息
    checkpoint_data = {
        'params': train_state.params,
        'training_log': training_log,
        'final_loss': final_loss,
        'num_epochs': num_epochs,
        'training_time': training_time,
        'action_repeat': action_repeat,  # 保存动作重复参数
        'action_obs_buffer_size': buffer_size,  # 保存动作-状态缓冲区大小参数
        'input_dimension': input_dim,  # 保存输入维度参数
        'env_config': {
            'max_steps_in_episode': env.max_steps_in_episode,
            'dt': env.dt,
            'delay': env.delay,
            'action_penalty_weight': env.action_penalty_weight,
            # HoverVer3 基于HoverVer2调整了loss权重，使用Quadrotor：悬停任务，初始位置在0~0.5m球体内随机，姿态和速度完全随机，不支持PID Kp参数随机化
            'hover_height': env.hover_height,
            'init_pos_range': env.init_pos_range,
            'max_distance': env.max_distance,
            'max_speed': env.max_speed,
        }
    }
    
    # 确保目录存在
    os.makedirs('aquila/param', exist_ok=True)
    
    # 保存为pickle文件
    checkpoint_path = 'aquila/param/hoverVer3_policy.pkl'  # 使用hoverVer3的文件名
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"\n✅ Trained hovering policy saved as: {checkpoint_path}")
    
    # 额外保存一个带时间戳的备份
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f'aquila/param/hoverVer3_policy_{timestamp}.pkl'  # 使用hoverVer3的文件名
    with open(backup_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"✅ Backup saved as: {backup_path}")
    
    # 关闭tensorboard writer
    writer.close()
    print(f"\n✅ TensorBoard logs saved to: {log_dir}")
    print(f"   Run 'tensorboard --logdir=runs' to view the results")


if __name__ == "__main__":
    main()

