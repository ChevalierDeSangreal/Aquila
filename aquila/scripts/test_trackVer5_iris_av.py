#!/usr/bin/env python
# coding: utf-8

import os
import random
import time
import asyncio

import numpy as np
import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter

import pytz
from datetime import datetime
import sys

# MAVLink imports
from mavsdk import System
from mavsdk.offboard import (AttitudeRate, OffboardError, PositionNedYaw)

"""
Test roll rate tracking with Gazebo SITL using MAVLink communication.
Sends constant roll rate commands and visualizes tracking performance.
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


# ==================== Main Test Function ====================
async def run():
    # ==================== Setup ====================
    print(f"\n{'='*60}")
    print(f"Roll Rate Tracking Test")
    print(f"{'='*60}\n")
    
    # JAX setup
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")
    
    # Random seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    
    # ==================== Test Configuration ====================
    dt = 0.01  # Time step in seconds
    max_steps = 1000  # Maximum number of steps per episode
    target_height = 2.0  # Target height in meters
    
    # Constant roll rate command (in rad/s)
    # You can modify this value to test different roll rates
    roll_rate_cmd_rad_s = 0.2  # 0.2 rad/s ≈ 11.5 deg/s
    roll_rate_cmd_deg_s = roll_rate_cmd_rad_s * 180.0 / np.pi
    
    # Hovering thrust for stability
    hovering_thrust = 0.71  # Gazebo range [0, 1]
    
    print(f"\n{'='*60}")
    print(f"Roll Rate Tracking Test Configuration:")
    print(f"  Time step: {dt}s")
    print(f"  Max steps: {max_steps}")
    print(f"  Target height: {target_height}m")
    print(f"  Roll rate command: {roll_rate_cmd_rad_s:.3f} rad/s ({roll_rate_cmd_deg_s:.2f} deg/s)")
    print(f"  Hovering thrust: {hovering_thrust:.3f}")
    print(f"{'='*60}\n")
    
    # ==================== TensorBoard Setup ====================
    run_name = f"roll_rate_tracking__{seed}__{get_time()}"
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    log_path = os.path.join(project_root, "aquila", "test_runs", run_name)
    os.makedirs(log_path, exist_ok=True)
    
    writer = SummaryWriter(log_path)
    writer.add_text(
        "hyperparameters",
        f"|param|value|\n|-|-|\n"
        f"|dt|{dt}|\n"
        f"|max_steps|{max_steps}|\n"
        f"|target_height|{target_height}|\n"
        f"|roll_rate_cmd_rad_s|{roll_rate_cmd_rad_s}|\n"
        f"|roll_rate_cmd_deg_s|{roll_rate_cmd_deg_s}|\n"
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
        PositionNedYaw(north_m=0, east_m=0, down_m=-target_height, yaw_deg=target_yaw_deg)
    )
    await drone.offboard.start()
    
    print("-- Initializing position")
    for _ in range(100):
        await drone.offboard.set_position_ned(
            PositionNedYaw(north_m=0, east_m=0, down_m=-target_height, yaw_deg=target_yaw_deg)
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
    
    # ==================== Get Initial State ====================
    odometry = latest_state["odometry"]
    attitude = latest_state["attitude"]
    
    initial_roll_deg = attitude.roll_deg
    initial_roll_rad = initial_roll_deg * np.pi / 180.0
    
    print(f"{'='*60}")
    print(f"Starting roll rate tracking test...")
    print(f"Initial roll angle: {initial_roll_deg:.2f}° ({initial_roll_rad:.3f} rad)")
    print(f"Commanded roll rate: {roll_rate_cmd_deg_s:.2f} deg/s ({roll_rate_cmd_rad_s:.3f} rad/s)")
    print(f"{'='*60}\n")
    
    # ==================== Main Test Loop ====================
    for step in range(max_steps):
        step_start_time = time.perf_counter()
        
        # ==================== Send Constant Roll Rate Command ====================
        # Send constant roll rate command with hovering thrust
        # Pitch and yaw rates are zero to maintain forward flight
        await drone.offboard.set_attitude_rate(
            AttitudeRate(
                roll_deg_s=float(roll_rate_cmd_deg_s),
                pitch_deg_s=0.0,
                yaw_deg_s=0.0,
                thrust_value=float(hovering_thrust)
            )
        )
            
        # ==================== Update State from Telemetry ====================
        odometry = latest_state["odometry"]
        attitude = latest_state["attitude"]
        
        # Get actual roll rate (in rad/s)
        roll_rate_actual_rad_s = odometry.angular_velocity_body.roll_rad_s
        roll_rate_actual_deg_s = roll_rate_actual_rad_s * 180.0 / np.pi
        
        # Get current roll angle
        roll_deg = attitude.roll_deg
        roll_rad = roll_deg * np.pi / 180.0
        
        # Get position and height
        quad_pos = jnp.array([
            odometry.position_body.x_m,
            odometry.position_body.y_m,
            odometry.position_body.z_m
        ])
        height = float(-quad_pos[2])  # NED: negative z is up
        
        # Compute tracking error
        roll_rate_error_rad_s = roll_rate_cmd_rad_s - roll_rate_actual_rad_s
        roll_rate_error_deg_s = roll_rate_error_rad_s * 180.0 / np.pi
        
        # Compute cumulative roll angle change
        cumulative_roll_change_deg = (roll_deg - initial_roll_deg) % 360.0
        if cumulative_roll_change_deg > 180.0:
            cumulative_roll_change_deg -= 360.0
        
        # ==================== Log to TensorBoard ====================
        # Roll rate tracking (main focus)
        writer.add_scalar('Roll_Rate/Command_rad_s', roll_rate_cmd_rad_s, step)
        writer.add_scalar('Roll_Rate/Actual_rad_s', roll_rate_actual_rad_s, step)
        writer.add_scalar('Roll_Rate/Error_rad_s', roll_rate_error_rad_s, step)
        writer.add_scalar('Roll_Rate/Command_deg_s', roll_rate_cmd_deg_s, step)
        writer.add_scalar('Roll_Rate/Actual_deg_s', roll_rate_actual_deg_s, step)
        writer.add_scalar('Roll_Rate/Error_deg_s', roll_rate_error_deg_s, step)
        
        # Roll angle
        writer.add_scalar('Roll_Angle/Roll_deg', roll_deg, step)
        writer.add_scalar('Roll_Angle/Cumulative_Change_deg', cumulative_roll_change_deg, step)
        
        # Position and height
        writer.add_scalar('Position/Height', height, step)
        writer.add_scalar('Position/North', float(quad_pos[0]), step)
        writer.add_scalar('Position/East', float(quad_pos[1]), step)
        writer.add_scalar('Position/Down', float(quad_pos[2]), step)
        
        # Attitude
        writer.add_scalar('Attitude/Roll_deg', attitude.roll_deg, step)
        writer.add_scalar('Attitude/Pitch_deg', attitude.pitch_deg, step)
        writer.add_scalar('Attitude/Yaw_deg', attitude.yaw_deg, step)
        
        # Angular velocities (all axes)
        writer.add_scalar('Angular_Velocity/Roll_rad_s', roll_rate_actual_rad_s, step)
        writer.add_scalar('Angular_Velocity/Pitch_rad_s', odometry.angular_velocity_body.pitch_rad_s, step)
        writer.add_scalar('Angular_Velocity/Yaw_rad_s', odometry.angular_velocity_body.yaw_rad_s, step)
        
        # Command sent
        writer.add_scalar('Command/Roll_rate_deg_s', roll_rate_cmd_deg_s, step)
        writer.add_scalar('Command/Thrust', hovering_thrust, step)
        
        # ==================== Print Progress ====================
        if step % 50 == 0:
            print(f"Step {step:4d} | "
                  f"Roll Rate Cmd: {roll_rate_cmd_deg_s:6.2f} deg/s | "
                  f"Roll Rate Actual: {roll_rate_actual_deg_s:6.2f} deg/s | "
                  f"Error: {roll_rate_error_deg_s:6.2f} deg/s | "
                  f"Roll Angle: {roll_deg:6.2f}° | "
                  f"Height: {height:5.2f}m")
        
        # ==================== Timing ====================
        step_end_time = time.perf_counter()
        elapsed_time = step_end_time - step_start_time
        sleep_time = dt - elapsed_time
        
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        
        # ==================== Check Termination ====================
        if height < 0.1 or height > 20.0:
            print(f"\n❌ Episode terminated: height out of bounds ({height:.1f}m)")
            break
    
    # ==================== Cleanup ====================
    print(f"\n{'='*60}")
    print(f"Roll rate tracking test completed!")
    print(f"Total steps: {step + 1}")
    print(f"Final roll angle: {roll_deg:.2f}°")
    print(f"Final roll rate (actual): {roll_rate_actual_deg_s:.2f} deg/s")
    print(f"Final roll rate (command): {roll_rate_cmd_deg_s:.2f} deg/s")
    print(f"Final roll rate error: {roll_rate_error_deg_s:.2f} deg/s")
    print(f"Final height: {height:.3f}m")
    print(f"Cumulative roll change: {cumulative_roll_change_deg:.2f}°")
    print(f"{'='*60}\n")
    
    telemetry_task.cancel()
    attitude_task.cancel()
    
    writer.close()
    
    print(f"✅ TensorBoard logs saved to: {log_path}")
    print(f"   View with: tensorboard --logdir={log_path}\n")


if __name__ == "__main__":
    asyncio.run(run())
