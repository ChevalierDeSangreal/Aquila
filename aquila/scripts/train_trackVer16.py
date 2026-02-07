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
from flax import linen as nn

# Add parent directory to path for imports
# aquila/scripts/train_trackVer14.py -> ../../ -> Aquila project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer16 import TrackEnvVer16  # 使用TrackEnvVer16（基于Ver15，添加圆形轨迹支持）
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP
from aquila.algos import bpttVer3

"""
TrackVer16: 基于Ver15，添加圆形轨迹支持和验证机制
- action_repeat = 2 (每2个step才获取一次新动作，每秒50次动作，每次持续0.02秒)
- buffer_size = 50 (动作-状态缓冲区大小)
- 网络输出：动作 (4,) + 辅助输出 (3,) = 目标速度预测（机体系）
- 辅助损失：预测值与真实值的L2距离，权重为1.0
- 验证：每10个epoch使用圆形轨迹验证一次
- 保存：训练最终权重 + 验证集上最佳权重
"""


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
    # Create env - 使用TrackEnvVer16环境（基于Ver15，添加圆形轨迹支持），训练50Hz网络
    env = TrackEnvVer16(
        max_steps_in_episode=600,  # 追踪任务的最大步数
        dt=0.01,  # 使用完整四旋翼的默认时间步长
        delay=0.03,  # 可选执行延迟
        omega_std=0.1,
        action_penalty_weight=0.5,
        # Tracking specific parameters
        target_height=2.0,  # m (高度2米，实际会在(-2.2, -1.8)范围内随机)
        target_init_distance_min=0.5,  # m (x轴上的初始距离最小值)
        target_init_distance_max=1.5,  # m (x轴上的初始距离最大值)
        target_speed_max=3.0,  # m/s (目标最大速度上限，实际每episode会在0-1之间随机)
        target_acceleration_max=5,  # m/s² (目标最大加速度，每次reset时在0到此值之间随机生成实际加速度)
        reset_distance=100.0,  # m (重置距离阈值)
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
    
    # 创建策略网络，输出7维：action (4,) + aux_output (3,)
    # 使用MLP直接输出7维，然后在bpttVer3中分割成 (action, aux_output)
    output_dim = action_dim + 3  # 4个动作 + 3个辅助输出
    policy = MLP([input_dim, 128, 128, output_dim], initial_scale=0.2)
    
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension (buffer_size * (obs_dim + action_dim)): {input_dim}")
    print(f"Network output: {output_dim}维 (action {action_dim} + aux_output 3)")
    
    # ==================== Training Parameters ====================
    num_epochs = 300  # 训练轮数
    num_envs = 512 # 并行环境数（单卡）
    
    # 动作重复参数
    action_repeat = 2  # 每2个step才获取一次新动作（每秒50次动作，每次持续0.02秒）
    
    # Ver16: 验证参数
    validation_interval = 10  # 每10个epoch验证一次
    num_val_episodes = 3  # 验证时运行的episode数量（从10改到3，加快验证速度）
    val_max_steps = 600  # 验证时每个episode的最大步数（从600改到300）
    
    # Optimizer - 使用余弦衰减学习率
    initial_learning_rate = 5e-3
    end_learning_rate = 5e-3
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
        policy_file = 'aquila/param_saved/trackVer16_policy.pkl'  # 使用Ver16的模型文件
        loaded_params, env_config = load_trained_policy(policy_file)
        train_state = TrainState.create(
            apply_fn=policy.apply,
            params=loaded_params,
            tx=tx
        )
        print("✅ 使用加载的网络参数继续训练")
    
    # ==================== TensorBoard Setup ====================
    # 创建tensorboard日志目录
    log_dir = f'runs/trackVer16_{time.strftime("%Y%m%d_%H%M%S")}'  # 使用Ver16的日志目录
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir=runs' to view training progress")
    
    # 设置 tensorboard writer 到 bpttVer3 模块（用于实时记录）
    bpttVer3.set_tensorboard_writer(writer)
    
    # ==================== Validation Setup ====================
    # Ver16: 创建单独的验证环境（使用圆形轨迹，参考RealFlight的target_publisher_node）
    # 圆形轨迹参数（参考target_publisher_node.cpp的默认参数）
    # ENU坐标系转NED：circle_center_z=1.2 -> -1.2，但实际使用-2更合适
    circle_radius = 2.0  # 半径2米
    circle_duration = 20.0  # 圆形运动周期20秒
    circle_angular_vel = 2.0 * jnp.pi / circle_duration  # 角速度
    circle_init_phase = 0.0  # 初始相位
    circle_center = jnp.array([0.0, 0.0, -2.0])  # 圆心位置（NED坐标系）
    
    # 创建验证环境（与训练环境相同的配置，但用于验证）
    val_env_raw = TrackEnvVer16(
        max_steps_in_episode=600,
        dt=0.01,
        delay=0.03,
        omega_std=0.1,
        action_penalty_weight=0.5,
        target_height=2.0,
        target_init_distance_min=0.5,
        target_init_distance_max=1.5,
        target_speed_max=3.0,
        target_acceleration_max=5,
        reset_distance=100.0,
        max_speed=20.0,
        thrust_to_weight_min=1.2,
        thrust_to_weight_max=5.0,
        disturbance_mag=0.0,  # 验证时不添加扰动
    )
    # 对验证环境应用相同的包装
    val_env = MinMaxObservationWrapper(val_env_raw)
    val_env = NormalizeActionWrapper(val_env)
    
    # 获取观测空间的边界用于手动归一化
    val_obs_min = jnp.array(val_env_raw.observation_space.low)
    val_obs_max = jnp.array(val_env_raw.observation_space.high)
    
    def normalize_obs(obs):
        """手动归一化观测到 [-1, 1]"""
        return 2.0 * (obs - val_obs_min) / (val_obs_max - val_obs_min) - 1.0
    
    # 记录验证最佳性能
    best_val_loss = float('inf')
    best_val_params = None
    best_val_epoch = 0
    
    def validation_fn(current_train_state, epoch_idx):
        """验证函数：使用圆形轨迹测试策略性能"""
        nonlocal best_val_loss, best_val_params, best_val_epoch
        
        val_losses = []
        val_distances = []
        val_velocities = []
        
        # 创建验证用的随机key
        val_key = jax.random.key(epoch_idx + 10000)
        
        # 运行多个验证episode
        for ep_idx in range(num_val_episodes):
            val_key, episode_key = jax.random.split(val_key)
            
            # 重置验证环境（使用圆形轨迹）
            # 直接访问原始环境来使用新的reset参数
            val_state, val_obs_raw = val_env_raw.reset(
                episode_key, 
                None, 
                None,
                use_circular_trajectory=True,
                circular_center=circle_center,
                circular_radius=circle_radius,
                circular_angular_vel=circle_angular_vel,
                circular_init_phase=circle_init_phase
            )
            
            # 手动归一化观测
            val_obs = normalize_obs(val_obs_raw)
            
            episode_reward = 0.0
            episode_distances = []
            episode_velocities = []
            
            # 初始化动作缓冲区（与训练时相同）
            action_obs_buffer = jnp.zeros((1, buffer_size, obs_dim + action_dim))
            
            # 运行一个episode（限制最大步数）
            for step_idx in range(val_max_steps):
                # 更新缓冲区（添加空动作 + 当前观测）
                action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=1)
                empty_action = jnp.zeros((1, 4))
                action_obs_combined = jnp.concatenate([empty_action, val_obs[None, :]], axis=1)
                action_obs_buffer = action_obs_buffer.at[:, -1, :].set(action_obs_combined)
                
                # 获取动作（每action_repeat步）
                if step_idx % action_repeat == 0:
                    action_obs_buffer_flat = action_obs_buffer.reshape(1, -1)
                    output_7d = current_train_state.apply_fn(current_train_state.params, action_obs_buffer_flat)
                    action = output_7d[:, :4]  # 前4维：动作
                    aux_output = output_7d[:, 4:]  # 后3维：辅助输出
                    
                    # 更新缓冲区（用真实动作替换空动作）
                    action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=1)
                    action_obs_combined = jnp.concatenate([action, val_obs[None, :]], axis=1)
                    action_obs_buffer = action_obs_buffer.at[:, -1, :].set(action_obs_combined)
                
                # 执行动作（直接在原始环境上）
                val_key, step_key = jax.random.split(val_key)
                transition = val_env_raw.step(val_state, (action[0], aux_output[0]), step_key)
                val_state, val_obs_raw, reward, terminated, truncated, info = transition
                
                # 手动归一化观测
                val_obs = normalize_obs(val_obs_raw)
                
                episode_reward += reward
                episode_distances.append(float(info['distance_to_target']))
                episode_velocities.append(float(jnp.linalg.norm(info['quad_v'])))
                
                if terminated or truncated:
                    break
            
            val_losses.append(-float(episode_reward) / (step_idx + 1))
            val_distances.append(np.mean(episode_distances))
            val_velocities.append(np.mean(episode_velocities))
        
        # 计算平均指标
        avg_val_loss = np.mean(val_losses)
        avg_val_distance = np.mean(val_distances)
        avg_val_velocity = np.mean(val_velocities)
        std_val_loss = np.std(val_losses)
        
        # 记录到tensorboard
        writer.add_scalar('Validation/loss', avg_val_loss, epoch_idx)
        writer.add_scalar('Validation/distance', avg_val_distance, epoch_idx)
        writer.add_scalar('Validation/velocity', avg_val_velocity, epoch_idx)
        writer.add_scalar('Validation/loss_std', std_val_loss, epoch_idx)
        
        # 检查是否为最佳验证性能
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_params = current_train_state.params
            best_val_epoch = epoch_idx
            writer.add_scalar('Validation/best_loss', best_val_loss, epoch_idx)
            print(f"  🎯 新的最佳验证性能！Loss: {best_val_loss:.6f}")
        
        return {
            'val_loss': avg_val_loss,
            'val_distance': avg_val_distance,
            'val_velocity': avg_val_velocity,
            'val_loss_std': std_val_loss,
        }
    
    # ==================== Training ====================
    time_start = time.time()
    training_log = []
    
    print(f"\n{'='*60}")
    print(f"开始训练追踪任务 (TrackVer16 - 基于Ver15，添加圆形轨迹支持和验证机制)...")
    print(f"Total environments: {num_envs}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Steps per epoch: {env.max_steps_in_episode}")
    print(f"Action repeat: {action_repeat} steps")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Network output: action + aux_target_vel (辅助损失)")
    print(f"Validation interval: {validation_interval} epochs")
    print(f"Validation episodes: {num_val_episodes}")
    print(f"Quadrotor model: Full (Quadrotor - based on agilicious framework)")
    print(f"{'='*60}\n")
    
    res_dict = bpttVer3.train(
        env=env,
        train_state=train_state,
        num_epochs=num_epochs,
        num_steps_per_epoch=env.max_steps_in_episode,
        num_envs=num_envs,
        key=key,
        truncate_k=500,  # 500表示完整BPTT
        action_repeat=action_repeat,  # 传递动作重复参数
        buffer_size=buffer_size,  # 传递动作-状态缓冲区大小参数
        validation_fn=validation_fn,  # Ver16: 传递验证函数
        validation_interval=validation_interval,  # Ver16: 传递验证间隔
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
    writer.add_scalar('Config/auxiliary_loss', 1.0, 0)  # Ver16继承：辅助损失权重
    writer.add_scalar('Config/validation_interval', validation_interval, 0)  # Ver16新增：验证间隔
    
    # 获取更新后的训练状态
    train_state = res_dict["runner_state"].train_state
    final_loss = float(losses_np[-1])
    
    # ==================== Print Summary ====================
    print(f"\n{'='*60}")
    print(f"追踪任务训练完成！(TrackVer16 - 基于Ver15，添加圆形轨迹支持和验证机制)")
    print(f"{'='*60}")
    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Final Loss: {final_loss:.6f}")
    print(f"Initial Loss: {float(losses_np[0]):.6f}")
    print(f"Loss improvement: {float(losses_np[0] - losses_np[-1]):.6f}")
    print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_val_epoch})")
    print(f"Action repeat: {action_repeat} steps")
    print(f"Action-obs buffer size: {buffer_size}")
    print(f"Input dimension: {input_dim}")
    print(f"Effective actions per epoch: {env.max_steps_in_episode / action_repeat:.1f}")
    print(f"Network output: action + aux_target_vel (辅助损失)")
    print(f"Quadrotor model: Full (Quadrotor - based on agilicious framework)")
    
    
    # ==================== Save Model ====================
    # 保存最终训练权重
    checkpoint_data = {
        'params': train_state.params,
        'training_log': training_log,
        'final_loss': final_loss,
        'num_epochs': num_epochs,
        'training_time': training_time,
        'action_repeat': action_repeat,  # 保存动作重复参数
        'action_obs_buffer_size': buffer_size,  # 保存动作-状态缓冲区大小参数
        'input_dimension': input_dim,  # 保存输入维度参数
        'best_val_loss': best_val_loss,  # Ver16: 保存最佳验证损失
        'best_val_epoch': best_val_epoch,  # Ver16: 保存最佳验证epoch
        'env_config': {
            'max_steps_in_episode': env.max_steps_in_episode,
            'dt': env.dt,
            'delay': env.delay,
            'action_penalty_weight': env.action_penalty_weight,
            # Ver16 基于Ver15，添加圆形轨迹支持和验证机制
            'target_height': env.target_height,
            'target_init_distance_min': env.target_init_distance_min,
            'target_init_distance_max': env.target_init_distance_max,
            'target_speed_max': env.target_speed_max,
            'target_acceleration_max': env.target_acceleration_max,
            'reset_distance': env.reset_distance,
            'max_speed': env.max_speed,
        }
    }
    
    # 确保目录存在
    os.makedirs('aquila/param_saved', exist_ok=True)
    
    # 保存最终权重为pickle文件
    checkpoint_path = 'aquila/param_saved/trackVer16_policy_final.pkl'  # 使用Ver16的文件名
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"\n✅ Final training policy saved as: {checkpoint_path}")
    
    # Ver16: 保存最佳验证权重
    if best_val_params is not None:
        best_checkpoint_data = {
            'params': best_val_params,
            'training_log': training_log,
            'final_loss': final_loss,
            'best_val_loss': best_val_loss,
            'best_val_epoch': best_val_epoch,
            'num_epochs': num_epochs,
            'training_time': training_time,
            'action_repeat': action_repeat,
            'action_obs_buffer_size': buffer_size,
            'input_dimension': input_dim,
            'env_config': checkpoint_data['env_config']
        }
        
        best_checkpoint_path = 'aquila/param_saved/trackVer16_policy_best_val.pkl'
        with open(best_checkpoint_path, 'wb') as f:
            pickle.dump(best_checkpoint_data, f)
        print(f"✅ Best validation policy saved as: {best_checkpoint_path}")
    
    # 额外保存一个带时间戳的备份
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = f'aquila/param_saved/trackVer16_policy_{timestamp}.pkl'  # 使用Ver16的文件名
    with open(backup_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    print(f"✅ Backup saved as: {backup_path}")
    
    # 关闭tensorboard writer
    writer.close()
    print(f"\n✅ TensorBoard logs saved to: {log_dir}")
    print(f"   Run 'tensorboard --logdir=runs' to view the results")


if __name__ == "__main__":
    main()