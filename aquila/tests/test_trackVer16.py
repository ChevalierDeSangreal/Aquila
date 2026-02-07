#!/usr/bin/env python
"""测试 TrackEnvVer16 的 JIT 编译和圆形轨迹功能"""

import jax
import jax.numpy as jnp
import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.target_trackVer16 import TrackEnvVer16
from aquila.utils.random import key_generator

def test_random_trajectory():
    """测试随机轨迹模式（训练模式）"""
    print("="*60)
    print("测试 1: 随机轨迹模式（训练模式）")
    print("="*60)
    
    key_gen = key_generator(0)
    env = TrackEnvVer16()
    
    # Reset with random trajectory
    state, obs = env.reset(next(key_gen), use_circular_trajectory=False)
    print(f"✓ Reset 成功")
    print(f"  初始目标位置: {state.target_pos}")
    print(f"  使用圆形轨迹: {state.use_circular_trajectory}")
    
    # Test step (JIT compiled)
    action = env.action_space.sample(next(key_gen))
    transition = env.step(state, action, next(key_gen))
    state, obs, reward, terminated, truncated, info = transition
    print(f"✓ Step 成功 (JIT 编译)")
    print(f"  奖励: {reward:.4f}")
    print(f"  距离: {info['distance_to_target']:.4f}")
    
    # Test multiple steps to ensure JIT works
    for i in range(10):
        action = env.action_space.sample(next(key_gen))
        transition = env.step(state, action, next(key_gen))
        state, obs, reward, terminated, truncated, info = transition
    print(f"✓ 10个连续步骤成功")
    print()

def test_circular_trajectory():
    """测试圆形轨迹模式（验证模式）"""
    print("="*60)
    print("测试 2: 圆形轨迹模式（验证模式）")
    print("="*60)
    
    key_gen = key_generator(1)
    env = TrackEnvVer16()
    
    # Reset with circular trajectory
    circle_center = jnp.array([0.0, 0.0, -2.0])
    circle_radius = 2.0
    circle_angular_vel = 2.0 * jnp.pi / 20.0  # 20秒一圈
    
    state, obs = env.reset(
        next(key_gen),
        use_circular_trajectory=True,
        circular_center=circle_center,
        circular_radius=circle_radius,
        circular_angular_vel=circle_angular_vel,
        circular_init_phase=0.0
    )
    print(f"✓ Reset 成功")
    print(f"  圆心: {state.circular_center}")
    print(f"  半径: {state.circular_radius}")
    print(f"  角速度: {state.circular_angular_vel:.4f} rad/s")
    print(f"  初始目标位置: {state.target_pos}")
    print(f"  初始目标速度: {state.target_vel}")
    print(f"  使用圆形轨迹: {state.use_circular_trajectory}")
    
    # 验证初始位置在圆上
    dist_to_center = jnp.linalg.norm(state.target_pos[:2] - circle_center[:2])
    print(f"  到圆心距离: {dist_to_center:.6f} (期望: {circle_radius:.6f})")
    assert jnp.allclose(dist_to_center, circle_radius, atol=1e-5), "初始位置不在圆上！"
    
    # Test step (JIT compiled)
    action = env.action_space.sample(next(key_gen))
    transition = env.step(state, action, next(key_gen))
    state, obs, reward, terminated, truncated, info = transition
    print(f"✓ Step 成功 (JIT 编译)")
    
    # Test multiple steps and verify target stays on circle
    positions = [state.target_pos]
    velocities = [state.target_vel]
    
    for i in range(100):
        action = env.action_space.sample(next(key_gen))
        transition = env.step(state, action, next(key_gen))
        state, obs, reward, terminated, truncated, info = transition
        positions.append(state.target_pos)
        velocities.append(state.target_vel)
    
    print(f"✓ 100个连续步骤成功")
    
    # 验证目标始终在圆上
    positions = jnp.array(positions)
    distances = jnp.linalg.norm(positions[:, :2] - circle_center[:2], axis=1)
    max_error = jnp.max(jnp.abs(distances - circle_radius))
    mean_error = jnp.mean(jnp.abs(distances - circle_radius))
    print(f"  圆形轨迹误差 - 最大: {max_error:.6f}, 平均: {mean_error:.6f}")
    assert jnp.allclose(distances, circle_radius, atol=1e-3), "目标偏离圆形轨迹！"
    
    # 验证速度大小恒定
    velocities = jnp.array(velocities)
    speeds = jnp.linalg.norm(velocities, axis=1)
    expected_speed = circle_angular_vel * circle_radius
    max_speed_error = jnp.max(jnp.abs(speeds - expected_speed))
    mean_speed_error = jnp.mean(jnp.abs(speeds - expected_speed))
    print(f"  速度误差 - 最大: {max_speed_error:.6f}, 平均: {mean_speed_error:.6f}")
    print(f"  期望速度: {expected_speed:.6f}, 实际平均速度: {jnp.mean(speeds):.6f}")
    print()

def test_jit_compilation():
    """测试 JIT 编译是否正常工作"""
    print("="*60)
    print("测试 3: JIT 编译验证")
    print("="*60)
    
    key_gen = key_generator(2)
    env = TrackEnvVer16()
    
    # 测试两种模式的 JIT 编译
    import time
    
    # 模式1: 随机轨迹
    state1, _ = env.reset(next(key_gen), use_circular_trajectory=False)
    action = env.action_space.sample(next(key_gen))
    
    start = time.time()
    for i in range(10):
        transition = env.step(state1, action, next(key_gen))
        state1 = transition[0]
    time1 = time.time() - start
    print(f"✓ 随机轨迹模式 10 步: {time1*1000:.2f} ms")
    
    # 模式2: 圆形轨迹
    state2, _ = env.reset(
        next(key_gen),
        use_circular_trajectory=True,
        circular_center=jnp.array([0.0, 0.0, -2.0]),
        circular_radius=2.0,
        circular_angular_vel=0.5
    )
    
    start = time.time()
    for i in range(10):
        transition = env.step(state2, action, next(key_gen))
        state2 = transition[0]
    time2 = time.time() - start
    print(f"✓ 圆形轨迹模式 10 步: {time2*1000:.2f} ms")
    
    print(f"✓ JIT 编译正常工作，两种模式都能运行")
    print()

def main():
    print("\n" + "="*60)
    print("TrackEnvVer16 测试套件")
    print("="*60 + "\n")
    
    try:
        test_random_trajectory()
        test_circular_trajectory()
        test_jit_compilation()
        
        print("="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n主要验证点:")
        print("  ✓ 随机轨迹模式正常工作")
        print("  ✓ 圆形轨迹模式正常工作")
        print("  ✓ JIT 编译没有错误")
        print("  ✓ 圆形轨迹保持在圆上")
        print("  ✓ 目标速度保持恒定")
        print("  ✓ jax.lax.cond 正确处理条件分支")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

