#!/usr/bin/env python
# coding: utf-8

import os
import time
import asyncio
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pytz
from datetime import datetime
import sys

# MAVLink imports
from mavsdk import System
from mavsdk.offboard import (AttitudeRate, PositionNedYaw)

"""
测试Iris四旋翼的最大角速度和推重比
参考test_trackVer5_iris_onboard.py，但不使用策略，而是发送测试命令
"""


# ==================== Utility Functions ====================
def get_time():
    """Get current timestamp in Shanghai timezone"""
    timestamp = time.time()
    dt_object_utc = datetime.utcfromtimestamp(timestamp)
    target_timezone = pytz.timezone("Asia/Shanghai")
    dt_object_local = dt_object_utc.replace(tzinfo=pytz.utc).astimezone(target_timezone)
    formatted_time_local = dt_object_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    return formatted_time_local


def safe_norm(x, eps=1e-8):
    """Safe norm computation"""
    return np.sqrt(np.sum(x * x) + eps)


# ==================== Telemetry Subscribers ====================
latest_state = {
    "odometry": None,
    "attitude": None
}

async def subscribe_telemetry(drone):
    """Subscribe to odometry updates"""
    async for odometry in drone.telemetry.odometry():
        latest_state["odometry"] = odometry

async def subscribe_attitude(drone):
    """Subscribe to attitude updates"""
    async for attitude in drone.telemetry.attitude_euler():
        latest_state["attitude"] = attitude


# ==================== Test Functions ====================
async def test_max_angular_velocity(drone, writer, test_axis='roll', max_cmd_rad_s=10.0, test_duration=1.0, target_height=-2.0):
    """
    测试最大角速度（带高度控制和安全机制）
    
    Args:
        drone: MAVLink无人机对象
        writer: TensorBoard writer
        test_axis: 测试轴 ('roll', 'pitch', 'yaw')
        max_cmd_rad_s: 最大命令角速度 (rad/s)
        test_duration: 每个测试命令的持续时间 (秒)
        target_height: 目标高度 (NED坐标系，向下为正，通常为-2.0表示2m高度)
    
    Returns:
        max_achieved_omega: 实际达到的最大角速度 (rad/s)
    """
    print(f"\n{'='*60}")
    print(f"测试最大{test_axis}角速度")
    print(f"{'='*60}\n")
    
    # 角速度命令范围 (rad/s) - 从较小的值开始，逐步增加
    test_values = np.linspace(0.5, max_cmd_rad_s, num=20)
    
    max_achieved_omega = 0.0
    results = []
    
    axis_idx_map = {'roll': 0, 'pitch': 1, 'yaw': 2}
    axis_idx = axis_idx_map[test_axis]
    
    # 高度PID控制器参数
    height_kp = 0.15  # 比例增益
    height_kd = 0.05  # 微分增益
    hover_thrust = 0.71  # 基准悬停推力
    
    # 安全参数
    max_height_error = 1.0  # 最大高度误差 (m)，超过此值停止测试
    min_safe_height = -0.5  # 最低安全高度 (m)，NED坐标系
    
    # 获取初始高度
    odometry = latest_state["odometry"]
    if odometry is None:
        print("❌ 无法获取初始状态，等待中...")
        await asyncio.sleep(1.0)
        odometry = latest_state["odometry"]
    
    initial_height = odometry.position_body.z_m if odometry else target_height
    print(f"初始高度: {initial_height:.3f}m (目标: {target_height:.3f}m)")
    
    # 记录是否因安全原因提前停止
    safety_stop = False
    
    for i, cmd_omega_rad_s in enumerate(test_values):
        # 如果之前因安全原因停止，不再继续测试更大的值
        if safety_stop:
            print(f"⚠️ 已因安全原因停止，跳过后续测试")
            break
        
        print(f"测试 {i+1}/{len(test_values)}: 命令角速度 = {cmd_omega_rad_s:.2f} rad/s ({cmd_omega_rad_s*180/np.pi:.1f} deg/s)")
        
        # 每次测试前先恢复到稳定状态
        if i > 0:
            print(f"  恢复到稳定状态...")
            for _ in range(30):
                odometry = latest_state["odometry"]
                if odometry is not None:
                    current_height = odometry.position_body.z_m
                    height_error = current_height - target_height
                    height_error_dot = 0.0  # 简化，不计算速度
                    
                    # PID控制推力
                    thrust_adjustment = height_kp * height_error + height_kd * height_error_dot
                    current_thrust = np.clip(hover_thrust + thrust_adjustment, 0.3, 0.95)
                    
                    await drone.offboard.set_attitude_rate(
                        AttitudeRate(
                            roll_deg_s=0.0,
                            pitch_deg_s=0.0,
                            yaw_deg_s=0.0,
                            thrust_value=float(current_thrust)
                        )
                    )
                await asyncio.sleep(0.02)
            
            # 等待稳定
            await asyncio.sleep(1.0)
        
        # 设置角速度命令
        roll_rate_deg_s = 0.0
        pitch_rate_deg_s = 0.0
        yaw_rate_deg_s = 0.0
        
        if test_axis == 'roll':
            roll_rate_deg_s = cmd_omega_rad_s * 180.0 / np.pi
        elif test_axis == 'pitch':
            pitch_rate_deg_s = cmd_omega_rad_s * 180.0 / np.pi
        elif test_axis == 'yaw':
            yaw_rate_deg_s = cmd_omega_rad_s * 180.0 / np.pi
        
        # 发送命令并记录数据（带高度控制）
        step = 0
        achieved_omegas = []
        heights = []
        prev_height = None
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < test_duration:
            # 获取当前状态
            odometry = latest_state["odometry"]
            if odometry is not None:
                current_height = odometry.position_body.z_m
                heights.append(current_height)
                
                # 高度PID控制
                height_error = current_height - target_height
                
                # 计算高度变化率（简化，使用最近的高度差）
                if prev_height is not None:
                    height_error_dot = (current_height - prev_height) / 0.01  # dt = 0.01s
                else:
                    height_error_dot = 0.0
                
                # PID控制推力调整
                thrust_adjustment = height_kp * height_error + height_kd * height_error_dot
                current_thrust = np.clip(hover_thrust + thrust_adjustment, 0.3, 0.95)
                
                # 安全检查
                if current_height > min_safe_height:
                    print(f"  ⚠️ 高度过低 ({current_height:.2f}m)，紧急恢复并停止当前测试!")
                    # 使用较大推力恢复
                    current_thrust = 0.9
                    safety_stop = True  # 标记为安全停止，不再测试更大的值
                elif abs(height_error) > max_height_error:
                    print(f"  ⚠️ 高度误差过大 ({height_error:.2f}m)，停止当前测试")
                    safety_stop = True  # 标记为安全停止，不再测试更大的值
                    break
                
                # 获取实际角速度
                omega_actual = np.array([
                    odometry.angular_velocity_body.roll_rad_s,
                    odometry.angular_velocity_body.pitch_rad_s,
                    odometry.angular_velocity_body.yaw_rad_s
                ])
                achieved_omegas.append(abs(omega_actual[axis_idx]))
                
                # 发送命令（带高度控制）
                await drone.offboard.set_attitude_rate(
                    AttitudeRate(
                        roll_deg_s=float(roll_rate_deg_s),
                        pitch_deg_s=float(pitch_rate_deg_s),
                        yaw_deg_s=float(yaw_rate_deg_s),
                        thrust_value=float(current_thrust)
                    )
                )
                
                # 记录到TensorBoard
                writer.add_scalar(f'AngularVelocity/{test_axis}/cmd_rad_s', cmd_omega_rad_s, step)
                writer.add_scalar(f'AngularVelocity/{test_axis}/actual_rad_s', omega_actual[axis_idx], step)
                writer.add_scalar(f'AngularVelocity/{test_axis}/actual_abs_rad_s', abs(omega_actual[axis_idx]), step)
                writer.add_scalar(f'AngularVelocity/{test_axis}/height_m', current_height, step)
                writer.add_scalar(f'AngularVelocity/{test_axis}/height_error_m', height_error, step)
                writer.add_scalar(f'AngularVelocity/{test_axis}/thrust', current_thrust, step)
                
                prev_height = current_height
            else:
                # 如果没有数据，使用基准推力
                await drone.offboard.set_attitude_rate(
                    AttitudeRate(
                        roll_deg_s=float(roll_rate_deg_s),
                        pitch_deg_s=float(pitch_rate_deg_s),
                        yaw_deg_s=float(yaw_rate_deg_s),
                        thrust_value=float(hover_thrust)
                    )
                )
            
            step += 1
            await asyncio.sleep(0.01)  # 10ms control loop
        
        # 计算统计信息
        if achieved_omegas:
            mean_omega = np.mean(achieved_omegas)
            max_omega = np.max(achieved_omegas)
            std_omega = np.std(achieved_omegas)
            
            # 高度统计
            mean_height = None
            height_std = None
            if heights:
                mean_height = np.mean(heights)
                height_std = np.std(heights)
                print(f"  实际角速度: 均值={mean_omega:.2f} rad/s, 最大值={max_omega:.2f} rad/s, 标准差={std_omega:.2f} rad/s")
                print(f"  高度: 均值={mean_height:.3f}m, 标准差={height_std:.3f}m")
            else:
                print(f"  实际角速度: 均值={mean_omega:.2f} rad/s, 最大值={max_omega:.2f} rad/s, 标准差={std_omega:.2f} rad/s")
            
            results.append({
                'cmd_omega': cmd_omega_rad_s,
                'mean_omega': mean_omega,
                'max_omega': max_omega,
                'std_omega': std_omega,
                'mean_height': mean_height,
                'height_std': height_std
            })
            
            if max_omega > max_achieved_omega:
                max_achieved_omega = max_omega
        else:
            print(f"  ⚠️ 未获取到角速度数据")
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"{test_axis.upper()}角速度测试结果:")
    print(f"  最大实际角速度: {max_achieved_omega:.2f} rad/s ({max_achieved_omega*180/np.pi:.1f} deg/s)")
    print(f"{'='*60}\n")
    
    return max_achieved_omega, results


async def test_thrust_to_weight_ratio(drone, writer, max_thrust_ratio=4.0, test_duration=1.0):
    """
    测试推重比 (推力/重力比)
    
    Args:
        drone: MAVLink无人机对象
        writer: TensorBoard writer
        max_thrust_ratio: 最大推重比（推力/重力）
        test_duration: 每个测试命令的持续时间 (秒)
    
    Returns:
        max_achieved_thrust_ratio: 实际达到的最大推重比
    """
    print(f"\n{'='*60}")
    print(f"测试推重比")
    print(f"{'='*60}\n")
    
    # 物理参数
    mass = 1.0  # kg (Iris quadrotor)
    gravity = 9.81  # m/s^2
    weight = mass * gravity  # N
    
    # 推力命令范围 (推重比)
    # Gazebo的推力范围是[0, 1]，1.0对应最大推力
    # 最大推力 = 4个电机 * 9N/电机 = 36N
    max_thrust_total = 36.0  # N
    max_thrust_normalized = 1.0  # Gazebo normalized
    
    # 推重比测试范围
    test_ratios = np.linspace(0.5, max_thrust_ratio, num=30)
    test_ratios = np.clip(test_ratios, 0.5, max_thrust_total / weight)  # 限制在物理范围内
    
    max_achieved_ratio = 0.0
    results = []
    
    for i, target_ratio in enumerate(test_ratios):
        # 计算目标推力
        target_thrust_N = target_ratio * weight
        # 转换为Gazebo的归一化推力 [0, 1]
        target_thrust_normalized = np.clip(target_thrust_N / max_thrust_total, 0.0, 1.0)
        
        print(f"测试 {i+1}/{len(test_ratios)}: 目标推重比 = {target_ratio:.2f}, "
              f"目标推力 = {target_thrust_N:.2f} N, "
              f"归一化推力 = {target_thrust_normalized:.3f}")
        
        # 发送推力命令（零角速度）
        step = 0
        velocities = []
        timestamps = []
        dt = 0.01  # 控制循环时间步长 (秒)
        start_time = time.perf_counter()
        
        while time.perf_counter() - start_time < test_duration:
            await drone.offboard.set_attitude_rate(
                AttitudeRate(
                    roll_deg_s=0.0,
                    pitch_deg_s=0.0,
                    yaw_deg_s=0.0,
                    thrust_value=float(target_thrust_normalized)
                )
            )
            
            # 获取实际速度
            odometry = latest_state["odometry"]
            if odometry is not None:
                vel = np.array([
                    odometry.velocity_body.x_m_s,
                    odometry.velocity_body.y_m_s,
                    odometry.velocity_body.z_m_s
                ])
                velocities.append(vel.copy())
                timestamps.append(time.perf_counter() - start_time)
                
                # 记录到TensorBoard
                writer.add_scalar(f'ThrustRatio/target_ratio', target_ratio, step)
                writer.add_scalar(f'ThrustRatio/target_thrust_N', target_thrust_N, step)
                writer.add_scalar(f'ThrustRatio/normalized_thrust', target_thrust_normalized, step)
                writer.add_scalar(f'ThrustRatio/velocity_z', vel[2], step)
            
            step += 1
            await asyncio.sleep(dt)
        
        # 估算实际推重比（基于垂直加速度）
        # 实际加速度 = (推力 - 重力) / 质量
        # 推重比 = 推力 / 重力 = (实际加速度 * 质量 + 重力) / 重力
        if len(velocities) > 20:  # 需要足够的数据点
            velocities = np.array(velocities)
            # 跳过初始瞬态（前20%的数据）
            skip = len(velocities) // 5
            vel_z = velocities[skip:, 2]  # 垂直速度（NED frame，向下为正）
            
            # 使用速度差分估算加速度：a = dv/dt
            if len(vel_z) > 1:
                # 计算速度差分
                dv = np.diff(vel_z)
                # 加速度 = 速度变化 / 时间步长
                accelerations = dv / dt
                # 使用平均值来估算稳态加速度
                mean_accel = np.mean(accelerations)
                
                # 在NED坐标系中，向下为正，所以加速度向下为正
                # 实际推力 = m * (a + g)，其中a是向下的加速度
                # 推重比 = 推力 / 重力 = (m * (a + g)) / (m * g) = (a + g) / g
                estimated_ratio = (mean_accel + gravity) / gravity
                estimated_thrust = estimated_ratio * weight
                
                results.append({
                    'target_ratio': target_ratio,
                    'target_thrust_N': target_thrust_N,
                    'normalized_thrust': target_thrust_normalized,
                    'estimated_ratio': estimated_ratio,
                    'estimated_thrust_N': estimated_thrust,
                    'mean_accel_m_s2': mean_accel
                })
                
                if estimated_ratio > max_achieved_ratio:
                    max_achieved_ratio = estimated_ratio
                
                print(f"  估算加速度: {mean_accel:.2f} m/s², "
                      f"推重比: {estimated_ratio:.2f}, "
                      f"推力: {estimated_thrust:.2f} N")
            else:
                print(f"  数据不足，跳过")
        else:
            print(f"  数据不足，跳过")
        
        # 短暂暂停，让系统稳定
        await asyncio.sleep(0.5)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"推重比测试结果:")
    print(f"  最大推重比: {max_achieved_ratio:.2f}")
    print(f"  最大推力: {max_achieved_ratio * weight:.2f} N")
    print(f"{'='*60}\n")
    
    return max_achieved_ratio, results


# ==================== Main Test Function ====================
async def run():
    # ==================== Setup ====================
    print(f"\n{'='*60}")
    print(f"Iris四旋翼最大角速度和推重比测试")
    print(f"{'='*60}\n")
    
    # ==================== TensorBoard Setup ====================
    run_name = f"max_omega_thrust_ratio_test__{get_time()}"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    log_path = os.path.join(project_root, "aquila", "test_runs", run_name)
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_path)
    writer.add_text(
        "test_info",
        f"|parameter|value|\n|-|-|\n"
        f"|test_type|max_angular_velocity_and_thrust_ratio|\n"
        f"|mass_kg|1.0|\n"
        f"|gravity_m_s2|9.81|\n"
    )
    
    print(f"TensorBoard logs: {log_path}\n")
    
    # ==================== MAVLink Connection ====================
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"✅ Connected to drone!")
            break
    
    print("-- Arming")
    await drone.action.arm()
    
    print("-- Taking off")
    await drone.action.takeoff()
    await asyncio.sleep(10)
    
    # Get initial position
    async for odometry in drone.telemetry.odometry():
        desired_z = odometry.position_body.z_m
        print(f"Initial position: N={odometry.position_body.x_m:.3f}, "
              f"E={odometry.position_body.y_m:.3f}, D={desired_z:.3f}")
        break
    
    # Start offboard mode
    target_yaw_deg = 0.0
    await drone.offboard.set_position_ned(
        PositionNedYaw(north_m=0, east_m=0, down_m=-2, yaw_deg=target_yaw_deg)
    )
    await drone.offboard.start()
    
    print("-- Initializing position")
    for _ in range(40):
        await drone.offboard.set_position_ned(
            PositionNedYaw(north_m=0, east_m=0, down_m=-2, yaw_deg=target_yaw_deg)
        )
        await asyncio.sleep(0.05)
    
    print("-- Preheating offboard setpoints...")
    for _ in range(20):
        await drone.offboard.set_attitude_rate(AttitudeRate(0, 0.0, 0.0, 0.71))
        await asyncio.sleep(0.02)
    
    # ==================== Start Telemetry Subscribers ====================
    telemetry_task = asyncio.create_task(subscribe_telemetry(drone))
    attitude_task = asyncio.create_task(subscribe_attitude(drone))
    
    # Wait for telemetry data
    while (latest_state["odometry"] is None) or (latest_state["attitude"] is None):
        await asyncio.sleep(0.1)
    
    print("✅ Telemetry data received\n")
    
    # ==================== Run Tests ====================
    test_results = {}
    
    # 测试最大角速度（每个轴）
    print("\n" + "="*60)
    print("开始角速度测试")
    print("="*60)
    
    # 获取目标高度（与初始化时的高度一致）
    target_height = -2.0  # NED坐标系，-2.0表示2m高度
    
    for axis in ['roll', 'pitch', 'yaw']:
        max_omega, results = await test_max_angular_velocity(
            drone, writer, 
            test_axis=axis, 
            max_cmd_rad_s=10.0,  # 测试到10 rad/s
            test_duration=1.0,  # 缩短测试时间，避免过度偏离
            target_height=target_height
        )
        test_results[f'max_omega_{axis}'] = max_omega
        test_results[f'max_omega_{axis}_results'] = results
        
        # 记录总结到TensorBoard
        writer.add_scalar(f'Summary/max_omega_{axis}_rad_s', max_omega, 0)
        writer.add_scalar(f'Summary/max_omega_{axis}_deg_s', max_omega * 180 / np.pi, 0)
        
        # 回到悬停状态（使用PID控制恢复高度）
        print(f"恢复到悬停状态...")
        hover_thrust = 0.71
        height_kp = 0.15
        height_kd = 0.05
        prev_height = None
        
        for recovery_step in range(100):  # 更长的恢复时间
            odometry = latest_state["odometry"]
            if odometry is not None:
                current_height = odometry.position_body.z_m
                
                # PID控制
                height_error = current_height - target_height
                if prev_height is not None:
                    height_error_dot = (current_height - prev_height) / 0.02
                else:
                    height_error_dot = 0.0
                
                thrust_adjustment = height_kp * height_error + height_kd * height_error_dot
                current_thrust = np.clip(hover_thrust + thrust_adjustment, 0.3, 0.95)
                
                await drone.offboard.set_attitude_rate(
                    AttitudeRate(0, 0.0, 0.0, float(current_thrust))
                )
                prev_height = current_height
            else:
                await drone.offboard.set_attitude_rate(
                    AttitudeRate(0, 0.0, 0.0, float(hover_thrust))
                )
            await asyncio.sleep(0.02)
        
        # 额外稳定时间
        await asyncio.sleep(2.0)
        print(f"恢复完成\n")
    
    # 测试推重比
    print("\n" + "="*60)
    print("开始推重比测试")
    print("="*60)
    
    max_thrust_ratio, thrust_results = await test_thrust_to_weight_ratio(
        drone, writer,
        max_thrust_ratio=4.0,  # 测试到4倍重力
        test_duration=1.0
    )
    test_results['max_thrust_ratio'] = max_thrust_ratio
    test_results['thrust_ratio_results'] = thrust_results
    
    # 记录总结到TensorBoard
    writer.add_scalar(f'Summary/max_thrust_ratio', max_thrust_ratio, 0)
    
    # ==================== Print Final Summary ====================
    print(f"\n{'='*60}")
    print(f"测试总结")
    print(f"{'='*60}")
    print(f"最大Roll角速度: {test_results['max_omega_roll']:.2f} rad/s ({test_results['max_omega_roll']*180/np.pi:.1f} deg/s)")
    print(f"最大Pitch角速度: {test_results['max_omega_pitch']:.2f} rad/s ({test_results['max_omega_pitch']*180/np.pi:.1f} deg/s)")
    print(f"最大Yaw角速度: {test_results['max_omega_yaw']:.2f} rad/s ({test_results['max_omega_yaw']*180/np.pi:.1f} deg/s)")
    print(f"最大推重比: {test_results['max_thrust_ratio']:.2f}")
    print(f"最大推力: {test_results['max_thrust_ratio'] * 1.0 * 9.81:.2f} N")
    print(f"{'='*60}\n")
    
    # ==================== Cleanup ====================
    print("返回到悬停状态...")
    for _ in range(100):
        await drone.offboard.set_attitude_rate(
            AttitudeRate(0, 0.0, 0.0, 0.71)
        )
        await asyncio.sleep(0.02)
    
    telemetry_task.cancel()
    attitude_task.cancel()
    
    writer.close()
    
    print(f"✅ TensorBoard logs saved to: {log_path}")
    print(f"   View with: tensorboard --logdir={log_path}\n")


if __name__ == "__main__":
    asyncio.run(run())

