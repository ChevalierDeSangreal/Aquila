#!/usr/bin/env python
# coding: utf-8
"""
Trajectory generation utilities for testing and simulation.

This module provides various trajectory generators that can be used to create
reference trajectories for quadrotor tracking tasks.
"""

import jax
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators."""
    
    @abstractmethod
    def get_position(self, t: float) -> jnp.ndarray:
        """
        Get position at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            Position vector [x, y, z] in NED frame
        """
        pass
    
    @abstractmethod
    def get_velocity(self, t: float) -> jnp.ndarray:
        """
        Get velocity at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            Velocity vector [vx, vy, vz] in NED frame
        """
        pass
    
    @abstractmethod
    def get_total_duration(self) -> float:
        """Get total duration of the trajectory in seconds."""
        pass
    
    def get_state(self, t: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get both position and velocity at time t.
        
        Args:
            t: Time in seconds
            
        Returns:
            Tuple of (position, velocity)
        """
        return self.get_position(t), self.get_velocity(t)


class CircularTrajectory(TrajectoryGenerator):
    """
    Circular trajectory generator with smooth ramp-up and ramp-down phases.
    
    This implementation follows the same logic as the C++ version in tmp.cpp,
    with three phases:
    1. Ramp-up: Angular velocity increases from 0 to max
    2. Constant: Angular velocity remains at max
    3. Ramp-down: Angular velocity decreases from max to 0
    
    The trajectory is parameterized in NED (North-East-Down) coordinates.
    """
    
    def __init__(self,
                 center: Tuple[float, float, float] = (0.0, 0.0, -1.2),
                 radius: float = 2.0,
                 num_circles: int = 1,
                 ramp_up_time: float = 3.0,
                 ramp_down_time: float = 3.0,
                 circle_duration: float = 20.0,
                 init_phase: float = 0.0,
                 max_speed: float = -1.0):
        """
        Initialize circular trajectory generator.
        
        Args:
            center: Circle center position [x, y, z] in NED frame (meters)
            radius: Circle radius (meters)
            num_circles: Number of complete circles to perform
            ramp_up_time: Time to accelerate from 0 to max angular velocity (seconds)
            ramp_down_time: Time to decelerate from max to 0 angular velocity (seconds)
            circle_duration: Nominal duration for one circle at constant velocity (seconds)
            init_phase: Initial phase angle (radians)
            max_speed: Maximum linear speed constraint (m/s), -1 to disable
        """
        self.center_x, self.center_y, self.center_z = center
        self.radius = radius
        self.num_circles = max(1, num_circles)
        self.ramp_up_time = ramp_up_time
        self.ramp_down_time = ramp_down_time
        self.circle_duration = circle_duration
        self.init_phase = init_phase
        self.max_speed = max_speed
        
        # Calculate effective duration considering max speed constraint
        self.effective_duration = self._calculate_effective_duration()
        
        # Calculate angular velocities and accelerations
        self.max_angular_vel = 2.0 * np.pi / self.effective_duration
        self.angular_acceleration = self.max_angular_vel / self.ramp_up_time
        
        # Calculate angular displacements for each phase
        theta_ramp_up = 0.5 * self.max_angular_vel * self.ramp_up_time
        theta_ramp_down = 0.5 * self.max_angular_vel * self.ramp_down_time
        theta_ramps_total = theta_ramp_up + theta_ramp_down
        
        # Total required angular displacement for N complete circles
        theta_required = self.num_circles * 2.0 * np.pi
        
        # Angular displacement needed during constant velocity phase
        theta_constant = theta_required - theta_ramps_total
        
        # Time needed at constant velocity
        self.total_constant_duration = theta_constant / self.max_angular_vel
        
        # Total trajectory duration
        self._total_duration = (self.ramp_up_time + 
                               self.total_constant_duration + 
                               self.ramp_down_time)
    
    def _calculate_effective_duration(self) -> float:
        """Calculate effective duration considering max speed constraint."""
        if self.max_speed <= 0:
            return self.circle_duration
        
        circumference = 2.0 * np.pi * self.radius
        min_duration = circumference / self.max_speed
        
        return max(min_duration, self.circle_duration)
    
    def _calculate_theta_at_time(self, t: float) -> float:
        """Calculate angular position at time t."""
        theta = 0.0
        omega_max = self.max_angular_vel
        alpha = self.angular_acceleration
        alpha_down = omega_max / self.ramp_down_time
        t_up = self.ramp_up_time
        t_const = self.total_constant_duration
        t_down = self.ramp_down_time
        
        # Phase 1: Ramp-up
        if t <= t_up:
            theta = 0.5 * alpha * t * t
        # Phase 2: Constant velocity
        elif t <= t_up + t_const:
            theta_at_t_up = 0.5 * alpha * t_up * t_up
            dt = t - t_up
            theta = theta_at_t_up + omega_max * dt
        # Phase 3: Ramp-down
        elif t <= t_up + t_const + t_down:
            theta_at_t_up = 0.5 * alpha * t_up * t_up
            theta_at_start_down = theta_at_t_up + omega_max * t_const
            
            t_start_down = t_up + t_const
            dt = t - t_start_down
            theta = theta_at_start_down + omega_max * dt - 0.5 * alpha_down * dt * dt
        # Phase 4: Hold final position
        else:
            theta_at_t_up = 0.5 * alpha * t_up * t_up
            theta_at_start_down = theta_at_t_up + omega_max * t_const
            theta = theta_at_start_down + omega_max * t_down - 0.5 * alpha_down * t_down * t_down
        
        return theta
    
    def _calculate_angular_velocity_at_time(self, t: float) -> float:
        """Calculate angular velocity at time t."""
        omega_max = self.max_angular_vel
        alpha = self.angular_acceleration
        alpha_down = omega_max / self.ramp_down_time
        t_up = self.ramp_up_time
        t_const = self.total_constant_duration
        t_down = self.ramp_down_time
        t_start_down = t_up + t_const
        
        current_omega = 0.0
        
        if t <= t_up:
            current_omega = alpha * t
        elif t <= t_start_down:
            current_omega = omega_max
        elif t <= t_start_down + t_down:
            dt_down = t - t_start_down
            current_omega = omega_max - alpha_down * dt_down
            current_omega = max(0.0, current_omega)
        else:
            current_omega = 0.0
        
        return current_omega
    
    def get_position(self, t: float) -> jnp.ndarray:
        """Get position at time t."""
        # Calculate angular position
        theta = self._calculate_theta_at_time(t)
        theta_with_phase = theta + self.init_phase
        
        # Position on circle
        x = self.center_x + self.radius * np.cos(theta_with_phase)
        y = self.center_y + self.radius * np.sin(theta_with_phase)
        z = self.center_z
        
        return jnp.array([x, y, z])
    
    def get_velocity(self, t: float) -> jnp.ndarray:
        """Get velocity at time t."""
        # Calculate angular velocity at current time
        current_omega = self._calculate_angular_velocity_at_time(t)
        
        # Calculate angular position for direction
        theta = self._calculate_theta_at_time(t)
        theta_with_phase = theta + self.init_phase
        
        # Linear velocity tangent to circle
        v_linear = current_omega * self.radius
        vx = -v_linear * np.sin(theta_with_phase)
        vy = v_linear * np.cos(theta_with_phase)
        vz = 0.0
        
        return jnp.array([vx, vy, vz])
    
    def get_total_duration(self) -> float:
        """Get total duration of the trajectory."""
        return self._total_duration
    
    def get_info(self) -> Dict[str, Any]:
        """Get trajectory information for logging."""
        return {
            'type': 'circular',
            'center': (self.center_x, self.center_y, self.center_z),
            'radius': self.radius,
            'num_circles': self.num_circles,
            'ramp_up_time': self.ramp_up_time,
            'ramp_down_time': self.ramp_down_time,
            'total_constant_duration': self.total_constant_duration,
            'total_duration': self._total_duration,
            'max_angular_vel': self.max_angular_vel,
            'max_linear_speed': self.max_angular_vel * self.radius,
        }


class StaticTrajectory(TrajectoryGenerator):
    """Static trajectory that stays at a fixed position."""
    
    def __init__(self, position: Tuple[float, float, float] = (0.0, 0.0, -1.2)):
        """
        Initialize static trajectory.
        
        Args:
            position: Fixed position [x, y, z] in NED frame (meters)
        """
        self.position = jnp.array(position)
    
    def get_position(self, t: float) -> jnp.ndarray:
        """Get position at time t (always the same)."""
        return self.position
    
    def get_velocity(self, t: float) -> jnp.ndarray:
        """Get velocity at time t (always zero)."""
        return jnp.zeros(3)
    
    def get_total_duration(self) -> float:
        """Get total duration (infinite for static trajectory)."""
        return float('inf')
    
    def get_info(self) -> Dict[str, Any]:
        """Get trajectory information for logging."""
        return {
            'type': 'static',
            'position': tuple(self.position),
        }


class LinearTrajectory(TrajectoryGenerator):
    """Linear trajectory with constant velocity."""
    
    def __init__(self,
                 start_position: Tuple[float, float, float] = (0.0, 0.0, -1.2),
                 velocity: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 duration: float = 10.0):
        """
        Initialize linear trajectory.
        
        Args:
            start_position: Starting position [x, y, z] in NED frame (meters)
            velocity: Constant velocity [vx, vy, vz] in NED frame (m/s)
            duration: Total duration of the trajectory (seconds)
        """
        self.start_position = jnp.array(start_position)
        self.velocity = jnp.array(velocity)
        self.duration = duration
    
    def get_position(self, t: float) -> jnp.ndarray:
        """Get position at time t."""
        t_clamped = min(t, self.duration)
        return self.start_position + self.velocity * t_clamped
    
    def get_velocity(self, t: float) -> jnp.ndarray:
        """Get velocity at time t."""
        if t <= self.duration:
            return self.velocity
        return jnp.zeros(3)
    
    def get_total_duration(self) -> float:
        """Get total duration of the trajectory."""
        return self.duration
    
    def get_info(self) -> Dict[str, Any]:
        """Get trajectory information for logging."""
        return {
            'type': 'linear',
            'start_position': tuple(self.start_position),
            'velocity': tuple(self.velocity),
            'duration': self.duration,
        }


class AcceleratedLinearTrajectory(TrajectoryGenerator):
    """
    Linear trajectory with constant acceleration to max speed, then constant velocity.
    
    Two phases:
    1. Acceleration: Constant acceleration from 0 to max_speed
    2. Constant velocity: Maintain max_speed
    """
    
    def __init__(self,
                 start_position: Tuple[float, float, float] = (0.0, 0.0, -1.2),
                 direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 max_speed: float = 1.0,
                 acceleration: float = 0.5,
                 constant_velocity_duration: float = 20.0):
        """
        Initialize accelerated linear trajectory.
        
        Args:
            start_position: Starting position [x, y, z] in NED frame (meters)
            direction: Direction vector (will be normalized) [dx, dy, dz] in NED frame
            max_speed: Maximum speed to reach (m/s)
            acceleration: Constant acceleration magnitude (m/s²)
            constant_velocity_duration: Duration to maintain max_speed after acceleration (seconds)
        """
        self.start_position = jnp.array(start_position)
        # Normalize direction vector
        direction_np = np.array(direction)
        direction_norm = np.linalg.norm(direction_np)
        if direction_norm < 1e-6:
            raise ValueError("Direction vector cannot be zero")
        self.direction = jnp.array(direction_np / direction_norm)
        
        self.max_speed = max_speed
        self.acceleration = acceleration
        self.constant_velocity_duration = constant_velocity_duration
        
        # Calculate time to reach max_speed
        self.acceleration_time = max_speed / acceleration if acceleration > 0 else 0.0
        
        # Calculate distance traveled during acceleration
        self.acceleration_distance = 0.5 * acceleration * self.acceleration_time ** 2
        
        # Total duration
        self._total_duration = self.acceleration_time + constant_velocity_duration
    
    def get_position(self, t: float) -> jnp.ndarray:
        """Get position at time t."""
        if t <= self.acceleration_time:
            # Phase 1: Acceleration
            # s = 0.5 * a * t^2
            distance = 0.5 * self.acceleration * t * t
        elif t <= self._total_duration:
            # Phase 2: Constant velocity
            # s = s_acc + v_max * (t - t_acc)
            distance = self.acceleration_distance + self.max_speed * (t - self.acceleration_time)
        else:
            # Beyond total duration, hold final position
            distance = self.acceleration_distance + self.max_speed * self.constant_velocity_duration
        
        return self.start_position + self.direction * distance
    
    def get_velocity(self, t: float) -> jnp.ndarray:
        """Get velocity at time t."""
        if t <= self.acceleration_time:
            # Phase 1: Acceleration
            # v = a * t
            speed = self.acceleration * t
        elif t <= self._total_duration:
            # Phase 2: Constant velocity
            speed = self.max_speed
        else:
            # Beyond total duration, velocity is zero
            speed = 0.0
        
        return self.direction * speed
    
    def get_total_duration(self) -> float:
        """Get total duration of the trajectory."""
        return self._total_duration
    
    def get_info(self) -> Dict[str, Any]:
        """Get trajectory information for logging."""
        return {
            'type': 'accelerated_linear',
            'start_position': tuple(self.start_position),
            'direction': tuple(self.direction),
            'max_speed': self.max_speed,
            'acceleration': self.acceleration,
            'acceleration_time': self.acceleration_time,
            'acceleration_distance': self.acceleration_distance,
            'constant_velocity_duration': self.constant_velocity_duration,
            'total_duration': self._total_duration,
        }


def create_trajectory(traj_type: str, **kwargs) -> TrajectoryGenerator:
    """
    Factory function to create trajectory generators.
    
    Args:
        traj_type: Type of trajectory ('circular', 'static', 'linear', 'accelerated_linear')
        **kwargs: Parameters specific to the trajectory type
        
    Returns:
        TrajectoryGenerator instance
        
    Examples:
        >>> # Create circular trajectory
        >>> traj = create_trajectory('circular', 
        ...                          center=(0, 0, -1.2), 
        ...                          radius=2.0, 
        ...                          num_circles=2)
        
        >>> # Create static trajectory
        >>> traj = create_trajectory('static', position=(0, 0, -1.5))
        
        >>> # Create linear trajectory
        >>> traj = create_trajectory('linear',
        ...                          start_position=(0, 0, -1.2),
        ...                          velocity=(1.0, 0.5, 0.0),
        ...                          duration=10.0)
        
        >>> # Create accelerated linear trajectory
        >>> traj = create_trajectory('accelerated_linear',
        ...                          start_position=(0, 0, -1.2),
        ...                          direction=(1.0, 0.0, 0.0),
        ...                          max_speed=1.0,
        ...                          acceleration=0.5,
        ...                          constant_velocity_duration=20.0)
    """
    if traj_type == 'circular':
        return CircularTrajectory(**kwargs)
    elif traj_type == 'static':
        return StaticTrajectory(**kwargs)
    elif traj_type == 'linear':
        return LinearTrajectory(**kwargs)
    elif traj_type == 'accelerated_linear':
        return AcceleratedLinearTrajectory(**kwargs)
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")

