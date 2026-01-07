#!/usr/bin/env python
# coding: utf-8

"""
Crazyflie实机测试脚本 - HoverVer2策略
使用OptiTrack提供位置，神经网络控制悬停

流程：
1. 连接Crazyflie
2. 使用位置控制模式起飞到0.5m高度
3. 切换到神经网络控制，悬停在当前点
4. 结束控制并降落

坐标系说明：
- 神经网络：NED (North-East-Down)
- Crazyflie/OptiTrack：ENU (East-North-Up)
- 转换关系：ENU_x=NED_y, ENU_y=NED_x, ENU_z=-NED_z
"""

import os
import sys
import time
import pickle
import numpy as np
from collections import deque
import threading
import logging

# ==================== 导入Crazyflie库 ====================
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.positioning.motion_commander import MotionCommander
    from cflib.utils import uri_helper
except ImportError:
    print("❌ 错误：未找到cflib库，请安装：pip install cflib")
    sys.exit(1)

# ==================== 导入OptiTrack库 ====================
try:
    from natnetclient import NatClient
except ImportError:
    print("❌ 错误：未找到natnetclient库，请安装：pip install natnetclient")
    sys.exit(1)

# ==================== 导入JAX和模型 ====================
import jax
import jax.numpy as jnp

# 添加父目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from aquila.envs.hoverVer2 import HoverEnvVer2, ExtendedQuadrotorParams
from aquila.envs.wrappers import MinMaxObservationWrapper, NormalizeActionWrapper
from aquila.modules.mlp import MLP

# ==================== 配置 ====================
# Crazyflie URI
CRAZYFLIE_URI = 'radio://0/80/2M/E7E7E7E701'  # 请根据实际情况修改

# 控制参数
TAKEOFF_HEIGHT = 0.5  # 起飞高度 (m, ENU坐标系)
HOVER_DURATION = 10.0  # 神经网络悬停时长 (s)
CONTROL_FREQ = 100  # 控制频率 (Hz)
dt = 1.0 / CONTROL_FREQ  # 控制周期 (s)

# 起飞降落参数
TAKEOFF_VELOCITY = 0.3  # 起飞速度 (m/s)
LAND_VELOCITY = 0.3  # 降落速度 (m/s)

# 神经网络参数（从训练加载）
POLICY_FILE = 'aquila/param/hoverVer2_policy.pkl'

# OptiTrack配置
OPTITRACK_SERVER_IP = "192.168.1.1"  # OptiTrack服务器IP，请根据实际情况修改
OPTITRACK_LOCAL_IP = "192.168.1.100"  # 本地IP，请根据实际情况修改
OPTITRACK_RIGID_BODY_ID = 1  # 刚体ID，请根据实际情况修改

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ==================== OptiTrack客户端 ====================
class OptiTrackReceiver:
    """OptiTrack位置数据接收器（仅接收位置，姿态从Crazyflie获取）"""
    
    def __init__(self, state_manager, crazyflie=None):
        """
        Args:
            state_manager: CrazyflieStateManager实例
            crazyflie: Crazyflie实例（用于发送位置数据）
        """
        self.state_manager = state_manager
        self.cf = crazyflie
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # 最新位置数据
        self.latest_position = None
        self.latest_timestamp = None
        
        # 初始化NatNet客户端
        try:
            self.natnet_client = NatClient(OPTITRACK_SERVER_IP)
        except Exception as e:
            self.natnet_client = None
            logger.error(f"❌ 无法初始化OptiTrack客户端: {e}")
    
    def _receive_loop(self):
        """接收循环：从OptiTrack获取位置数据"""
        try:
            while self.running:
                # 获取刚体数据
                rigid_body = self.natnet_client.rigid_bodies.get(OPTITRACK_RIGID_BODY_ID)
                if rigid_body is not None:
                    # 只获取位置（ENU坐标系），不获取姿态
                    pos = rigid_body.position
                    x, y, z = pos[0], pos[1], pos[2]
                    
                    with self.lock:
                        self.latest_position = np.array([x, y, z])
                        self.latest_timestamp = time.time()
                    
                    # 更新状态管理器（只更新位置）
                    self.state_manager.update_position(x, y, z)
                    
                    # 发送位置给Crazyflie（通过外部定位系统）
                    if self.cf is not None:
                        self._send_position_to_crazyflie(x, y, z)
                
                time.sleep(0.01)  # 100Hz
        except Exception as e:
            logger.error(f"OptiTrack接收循环错误: {e}")
    
    def _send_position_to_crazyflie(self, x, y, z):
        """将位置数据发送给Crazyflie外部定位系统（仅位置，不含姿态）"""
        try:
            # 使用Crazyflie的外部定位系统接口
            # 注意：这需要Crazyflie固件支持外部定位（如lighthouse或mocap）
            
            # 通过extpos接口发送位置（仅位置，姿态由Crazyflie自己估计）
            if hasattr(self.cf, 'extpos'):
                # 发送位置（ENU坐标系，单位：米）
                # 注意：某些extpos接口可能需要姿态，但我们可以只发送位置
                try:
                    self.cf.extpos.send_extpos(x, y, z)
                except TypeError:
                    # 如果接口需要更多参数，尝试发送零姿态
                    self.cf.extpos.send_extpos(x, y, z, 0.0, 0.0, 0.0, 1.0)
            else:
                logger.debug(f"发送位置到Crazyflie: ({x:.3f}, {y:.3f}, {z:.3f})")
        except Exception as e:
            # 只在调试模式下显示警告，避免日志过多
            if logger.level <= logging.DEBUG:
                logger.warning(f"发送位置到Crazyflie失败: {e}")
    
    def start(self):
        """启动OptiTrack接收线程"""
        if self.natnet_client is None:
            logger.error("❌ OptiTrack客户端不可用，无法启动")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        logger.info("✅ OptiTrack接收线程已启动")
        return True
    
    def stop(self):
        """停止OptiTrack接收线程"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        logger.info("✅ OptiTrack接收线程已停止")
    
    def get_latest_position(self):
        """获取最新位置"""
        with self.lock:
            return self.latest_position.copy() if self.latest_position is not None else None


# ==================== 坐标系转换 ====================
def enu_to_ned(pos_enu):
    """
    ENU坐标转NED坐标
    ENU: x=East, y=North, z=Up
    NED: x=North, y=East, z=Down
    转换：NED_x = ENU_y, NED_y = ENU_x, NED_z = -ENU_z
    """
    if isinstance(pos_enu, np.ndarray):
        return np.array([pos_enu[1], pos_enu[0], -pos_enu[2]])
    else:
        return jnp.array([pos_enu[1], pos_enu[0], -pos_enu[2]])


def ned_to_enu(pos_ned):
    """
    NED坐标转ENU坐标
    转换：ENU_x = NED_y, ENU_y = NED_x, ENU_z = -NED_z
    """
    if isinstance(pos_ned, np.ndarray):
        return np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])
    else:
        return jnp.array([pos_ned[1], pos_ned[0], -pos_ned[2]])


# ==================== Crazyflie状态管理器 ====================
class CrazyflieStateManager:
    """管理Crazyflie的状态信息（位置、速度、姿态等）"""
    
    def __init__(self):
        self.lock = threading.Lock()
        
        # 位置和速度（ENU坐标系）
        self.position = np.array([0.0, 0.0, 0.0])  # [x, y, z] in ENU
        self.velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz] in ENU
        
        # 姿态（四元数：[x, y, z, w]）
        self.quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        
        # 角速度（弧度/秒）
        self.gyro = np.array([0.0, 0.0, 0.0])  # [wx, wy, wz]
        
        # 加速度（m/s²）
        self.acceleration = np.array([0.0, 0.0, 0.0])  # [ax, ay, az]
        
        # 上一时刻的位置（用于计算速度）
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.last_time = time.time()
        
        # 数据是否已更新
        self.data_ready = False
    
    def update_position(self, x, y, z):
        """更新位置（ENU坐标系）"""
        with self.lock:
            current_time = time.time()
            dt = current_time - self.last_time
            
            new_position = np.array([x, y, z])
            
            # 计算速度（数值微分）
            if dt > 0 and self.data_ready:
                self.velocity = (new_position - self.position) / dt
            
            self.last_position = self.position.copy()
            self.position = new_position
            self.last_time = current_time
            self.data_ready = True
    
    def update_attitude(self, qx, qy, qz, qw):
        """更新姿态（四元数）"""
        with self.lock:
            self.quaternion = np.array([qx, qy, qz, qw])
    
    def update_gyro(self, wx, wy, wz):
        """更新角速度（弧度/秒）"""
        with self.lock:
            # Crazyflie gyro单位是deg/s，转换为rad/s
            self.gyro = np.deg2rad(np.array([wx, wy, wz]))
    
    def update_acceleration(self, ax, ay, az):
        """更新加速度（m/s²）"""
        with self.lock:
            self.acceleration = np.array([ax, ay, az])
    
    def get_state_enu(self):
        """获取当前状态（ENU坐标系）"""
        with self.lock:
            return {
                'position': self.position.copy(),
                'velocity': self.velocity.copy(),
                'quaternion': self.quaternion.copy(),
                'gyro': self.gyro.copy(),
                'acceleration': self.acceleration.copy(),
                'data_ready': self.data_ready
            }
    
    def get_rotation_matrix_enu(self):
        """从四元数计算旋转矩阵（ENU坐标系）"""
        with self.lock:
            qx, qy, qz, qw = self.quaternion
            
            # 四元数转旋转矩阵
            R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])
            
            return R


# ==================== 加载训练好的策略 ====================
def load_trained_policy(checkpoint_path):
    """加载训练好的策略参数"""
    logger.info(f"加载策略：{checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        params = data['params']
        env_config = data.get('env_config', {})
        final_loss = data.get('final_loss', 'Unknown')
        training_epochs = data.get('training_epochs', 'Unknown')
        action_repeat = data.get('action_repeat', 2)
        buffer_size = data.get('action_obs_buffer_size', 50)
    else:
        params = data
        env_config = {}
        final_loss = 'Unknown'
        training_epochs = 'Unknown'
        action_repeat = 2
        buffer_size = 50
    
    logger.info(f"✅ 策略加载成功！")
    logger.info(f"   Final loss: {final_loss}")
    logger.info(f"   Training epochs: {training_epochs}")
    logger.info(f"   Action repeat: {action_repeat}")
    logger.info(f"   Buffer size: {buffer_size}")
    
    return params, env_config, action_repeat, buffer_size


# ==================== 观测构建 ====================
def build_observation(state_manager, hover_target_ned, env):
    """
    构建神经网络观测
    
    Args:
        state_manager: CrazyflieStateManager实例
        hover_target_ned: 悬停目标位置（NED坐标系）
        env: 环境实例（用于获取观测归一化参数）
    
    Returns:
        obs: 归一化后的观测向量
    """
    # 获取当前状态（ENU坐标系）
    state_enu = state_manager.get_state_enu()
    
    # 转换到NED坐标系
    pos_ned = enu_to_ned(state_enu['position'])
    vel_ned = enu_to_ned(state_enu['velocity'])
    
    # 获取旋转矩阵（ENU坐标系）
    R_enu = state_manager.get_rotation_matrix_enu()
    
    # 转换旋转矩阵到NED坐标系
    # ENU->NED转换矩阵
    T_enu_to_ned = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, -1]
    ])
    R_ned = T_enu_to_ned @ R_enu @ T_enu_to_ned.T
    
    # 角速度转换到NED
    gyro_ned = enu_to_ned(state_enu['gyro'])
    
    # 构建观测（与HoverEnvVer2.get_obs()一致）
    # 位置误差
    pos_error = pos_ned - hover_target_ned
    
    # 姿态（旋转矩阵的前6个元素）
    R_flat = R_ned[:2, :].flatten()  # 取前两行，展平为6个元素
    
    # 组合观测：[pos_error(3), vel(3), R_flat(6), omega(3)] = 15维
    obs_raw = np.concatenate([
        pos_error,
        vel_ned,
        R_flat,
        gyro_ned
    ])
    
    # 转换为JAX数组
    obs_raw = jnp.array(obs_raw, dtype=jnp.float32)
    
    # 应用MinMaxObservationWrapper的归一化
    # 注意：这里需要使用环境的归一化参数
    original_env = env.unwrapped
    obs_min = original_env._obs_min
    obs_max = original_env._obs_max
    
    # 归一化到[-1, 1]
    obs = 2.0 * (obs_raw - obs_min) / (obs_max - obs_min + 1e-8) - 1.0
    obs = jnp.clip(obs, -1.0, 1.0)
    
    return obs


def action_normalized_to_crazyflie(action_normalized, omega_max):
    """
    将归一化的网络输出直接转换为Crazyflie控制指令
    
    Args:
        action_normalized: 归一化的动作 [thrust, wx, wy, wz]，范围[-1, 1]
                          - thrust: -1~1，表示推力百分比（-1对应0%，1对应100%）
                          - wx, wy, wz: -1~1，表示角速度（弧度/秒），范围[-omega_max, omega_max]
        omega_max: 最大角速度 (rad/s)
    
    Returns:
        thrust_percentage: 推力百分比，范围[0, 100]
        roll_rate: roll角速度 (deg/s) in ENU body frame
        pitch_rate: pitch角速度 (deg/s) in ENU body frame
        yaw_rate: yaw角速度 (deg/s) in ENU body frame
    """
    # 推力：从-1~1映射到0~100%
    # -1 -> 0%, 0 -> 50%, 1 -> 100%
    thrust_percentage = (action_normalized[0] + 1.0) / 2.0 * 100.0
    thrust_percentage = np.clip(thrust_percentage, 0.0, 100.0)
    
    # 角速度：网络输出是-1~1（弧度/秒），需要转换为度/秒
    # 并转换坐标系：NED -> ENU body frame
    wx_ned = action_normalized[1] * omega_max  # 弧度/秒
    wy_ned = action_normalized[2] * omega_max  # 弧度/秒
    wz_ned = action_normalized[3] * omega_max  # 弧度/秒
    
    # 坐标系转换：NED -> ENU body frame
    # NED: x=North, y=East, z=Down
    # ENU body frame: x=East, y=North, z=Up
    # roll_rate (绕x轴) = wy_ned
    # pitch_rate (绕y轴) = wx_ned
    # yaw_rate (绕z轴) = -wz_ned
    roll_rate = np.rad2deg(wy_ned)   # ENU x对应NED y
    pitch_rate = np.rad2deg(wx_ned)  # ENU y对应NED x
    yaw_rate = np.rad2deg(-wz_ned)   # ENU z对应NED -z
    
    return thrust_percentage, roll_rate, pitch_rate, yaw_rate


# ==================== 主控制流程 ====================
def setup_logging(scf, state_manager):
    """设置Crazyflie日志回调
    
    注意：
    - 位置数据从OptiTrack直接获取（通过OptiTrackReceiver）
    - 姿态数据从Crazyflie日志获取（stateEstimate.qx/qy/qz/qw）
    - 角速度数据从Crazyflie日志获取（gyro.x/y/z）
    """
    
    # 姿态日志（从Crazyflie获取）
    log_att = LogConfig(name='Attitude', period_in_ms=10)
    log_att.add_variable('stateEstimate.qx', 'float')
    log_att.add_variable('stateEstimate.qy', 'float')
    log_att.add_variable('stateEstimate.qz', 'float')
    log_att.add_variable('stateEstimate.qw', 'float')
    
    def att_callback(timestamp, data, logconf):
        qx = data['stateEstimate.qx']
        qy = data['stateEstimate.qy']
        qz = data['stateEstimate.qz']
        qw = data['stateEstimate.qw']
        state_manager.update_attitude(qx, qy, qz, qw)
    
    scf.cf.log.add_config(log_att)
    log_att.data_received_cb.add_callback(att_callback)
    log_att.start()
    
    # 陀螺仪日志
    log_gyro = LogConfig(name='Gyro', period_in_ms=10)
    log_gyro.add_variable('gyro.x', 'float')
    log_gyro.add_variable('gyro.y', 'float')
    log_gyro.add_variable('gyro.z', 'float')
    
    def gyro_callback(timestamp, data, logconf):
        wx = data['gyro.x']
        wy = data['gyro.y']
        wz = data['gyro.z']
        state_manager.update_gyro(wx, wy, wz)
    
    scf.cf.log.add_config(log_gyro)
    log_gyro.data_received_cb.add_callback(gyro_callback)
    log_gyro.start()
    
    logger.info("✅ 日志配置完成（位置数据从OptiTrack获取）")
    
    return [log_att, log_gyro]  # 不再返回log_pos


def takeoff_to_height(scf, state_manager, target_height, timeout=10.0):
    """
    使用位置控制起飞到指定高度
    
    Args:
        scf: SyncCrazyflie实例
        state_manager: 状态管理器
        target_height: 目标高度 (m, ENU坐标系)
        timeout: 超时时间 (s)
    """
    logger.info(f"起飞到高度 {target_height:.2f}m...")
    
    # 等待状态数据准备好
    start_time = time.time()
    while not state_manager.data_ready:
        time.sleep(0.01)
        if time.time() - start_time > 5.0:
            raise RuntimeError("等待状态数据超时")
    
    # 记录起飞点
    state = state_manager.get_state_enu()
    takeoff_x = state['position'][0]
    takeoff_y = state['position'][1]
    
    logger.info(f"起飞点：({takeoff_x:.3f}, {takeoff_y:.3f}, 0.0)")
    
    # 使用高级指挥官发送位置指令
    cf = scf.cf
    
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > timeout:
            logger.warning("起飞超时")
            break
        
        # 获取当前高度
        state = state_manager.get_state_enu()
        current_height = state['position'][2]
        
        # 检查是否到达目标高度
        if abs(current_height - target_height) < 0.05:  # 5cm误差
            logger.info(f"✅ 到达目标高度：{current_height:.3f}m")
            break
        
        # 发送位置指令（使用高级指挥官）
        cf.commander.send_position_setpoint(takeoff_x, takeoff_y, target_height, 0)
        
        time.sleep(dt)
    
    # 稳定一段时间
    logger.info("稳定中...")
    for _ in range(int(1.0 / dt)):  # 稳定1秒
        cf.commander.send_position_setpoint(takeoff_x, takeoff_y, target_height, 0)
        time.sleep(dt)
    
    return takeoff_x, takeoff_y


def neural_network_hover(scf, state_manager, policy, params, env, 
                         hover_target_enu, duration, action_repeat, buffer_size):
    """
    使用神经网络控制悬停
    
    Args:
        scf: SyncCrazyflie实例
        state_manager: 状态管理器
        policy: 神经网络策略
        params: 策略参数
        env: 环境实例
        hover_target_enu: 悬停目标位置（ENU坐标系）
        duration: 悬停时长 (s)
        action_repeat: 动作重复次数
        buffer_size: 动作-观测缓冲区大小
    
    Returns:
        data: 记录的数据字典
    """
    logger.info(f"开始神经网络控制悬停（目标：{hover_target_enu}，时长：{duration:.1f}s）...")
    
    cf = scf.cf
    
    # 转换目标位置到NED坐标系
    hover_target_ned = enu_to_ned(hover_target_enu)
    
    # 无人机参数（假设值，可以根据实际情况调整）
    omega_max = np.array([1.0, 1.0, 1.0])  # 最大角速度 (rad/s)
    
    # 数据记录
    positions_enu = []
    velocities_enu = []
    actions_ned = []
    times = []
    
    # 获取初始观测
    obs = build_observation(state_manager, hover_target_ned, env)
    
    # 初始化动作-观测缓冲区（与训练一致）
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 设置悬停动作的原始值（0到1范围）
    hovering_thrust_raw = 0.1  # 0到1之间的值
    # 归一化到-1到1范围：normalized = 2.0 * value - 1.0
    hovering_thrust_normalized = 2.0 * hovering_thrust_raw - 1.0
    hovering_action_normalized = jnp.array([hovering_thrust_normalized, 0.0, 0.0, 0.0])
    
    # 初始化缓冲区（所有位置：零观测）
    zero_obs = jnp.zeros_like(obs)
    action_obs_combined_zero = jnp.concatenate([hovering_action_normalized, zero_obs])
    action_obs_buffer = jnp.tile(action_obs_combined_zero[None, :], (buffer_size, 1))
    
    # 获取初始动作
    action_obs_buffer_flat = action_obs_buffer.reshape(-1)
    current_action_normalized = policy.apply(params, action_obs_buffer_flat)
    current_action_normalized = np.array(current_action_normalized)
    
    # 初始化动作计数器
    action_counter = 0
    
    # 主控制循环
    start_time = time.time()
    step = 0
    
    try:
        while True:
            loop_start = time.time()
            current_time = loop_start - start_time
            
            # 检查是否超时
            if current_time > duration:
                logger.info(f"✅ 悬停完成（{duration:.1f}s）")
                break
            
            # 获取当前状态
            state = state_manager.get_state_enu()
            
            # 记录数据
            positions_enu.append(state['position'].copy())
            velocities_enu.append(state['velocity'].copy())
            times.append(current_time)
            
            # 每action_repeat步获取新动作
            need_new_action = (action_counter % action_repeat) == 0
            
            if need_new_action:
                # 构建观测
                obs = build_observation(state_manager, hover_target_ned, env)
                
                # 步骤1：用空动作+当前观测更新缓冲区（准备获取新动作）
                action_obs_buffer_for_input = jnp.roll(action_obs_buffer, shift=-1, axis=0)
                empty_action = jnp.zeros(action_dim)
                action_obs_combined_empty = jnp.concatenate([empty_action, obs])
                action_obs_buffer_for_input = action_obs_buffer_for_input.at[-1, :].set(action_obs_combined_empty)
                
                # 步骤2：使用更新后的缓冲区获取新动作
                action_obs_buffer_flat = action_obs_buffer_for_input.reshape(-1)
                current_action_normalized = policy.apply(params, action_obs_buffer_flat)
                current_action_normalized = np.array(current_action_normalized)
                
                # 步骤3：用新动作+当前观测更新缓冲区
                action_obs_buffer = jnp.roll(action_obs_buffer, shift=-1, axis=0)
                action_obs_combined_new = jnp.concatenate([jnp.array(current_action_normalized), obs])
                action_obs_buffer = action_obs_buffer.at[-1, :].set(action_obs_combined_new)
                
                # 重置计数器
                action_counter = 1
            else:
                action_counter += 1
            
            # 将归一化的网络输出转换为Crazyflie控制指令
            # 网络输出：[-1, 1]的推力，[-1, 1]的角速度（弧度/秒）
            thrust_percentage, roll_rate, pitch_rate, yaw_rate = action_normalized_to_crazyflie(
                current_action_normalized, omega_max
            )
            
            # 记录动作（保存归一化后的动作，用于后续分析）
            actions_ned.append(current_action_normalized.copy())
            
            # 发送控制指令（使用send_setpoint_manual，rate=True表示roll/pitch是角速度）
            # 参数：roll(deg/s), pitch(deg/s), yawrate(deg/s), thrust_percentage(0-100), rate=True
            cf.commander.send_setpoint_manual(roll_rate, pitch_rate, yaw_rate, thrust_percentage, rate=True)
            
            # 每100步打印一次状态
            if step % 100 == 0:
                pos = state['position']
                vel = state['velocity']
                distance = np.linalg.norm(pos - hover_target_enu)
                logger.info(f"[{current_time:.2f}s] 位置(ENU): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                           f"速度: {np.linalg.norm(vel):.3f}m/s, 距离: {distance:.3f}m")
            
            step += 1
            
            # 控制频率
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif step % 100 == 0:
                logger.warning(f"控制循环超时：{elapsed*1000:.1f}ms (目标：{dt*1000:.1f}ms)")
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    
    # 返回记录的数据
    return {
        'positions_enu': np.array(positions_enu),
        'velocities_enu': np.array(velocities_enu),
        'actions_ned': np.array(actions_ned),
        'times': np.array(times),
        'hover_target_enu': hover_target_enu,
        'hover_target_ned': hover_target_ned,
    }


def land_safely(scf, state_manager, land_height=0.0, timeout=10.0):
    """
    安全降落
    
    Args:
        scf: SyncCrazyflie实例
        state_manager: 状态管理器
        land_height: 降落目标高度 (m, ENU坐标系)
        timeout: 超时时间 (s)
    """
    logger.info("开始降落...")
    
    cf = scf.cf
    
    # 获取当前位置
    state = state_manager.get_state_enu()
    land_x = state['position'][0]
    land_y = state['position'][1]
    
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        if elapsed > timeout:
            logger.warning("降落超时")
            break
        
        # 获取当前高度
        state = state_manager.get_state_enu()
        current_height = state['position'][2]
        
        # 检查是否到达地面
        if current_height < land_height + 0.05:  # 5cm误差
            logger.info(f"✅ 到达地面：{current_height:.3f}m")
            break
        
        # 发送位置指令
        cf.commander.send_position_setpoint(land_x, land_y, land_height, 0)
        
        time.sleep(dt)
    
    # 停止电机
    cf.commander.send_stop_setpoint()
    logger.info("电机已停止")


def save_results(data, output_dir='aquila/output'):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    data_path = os.path.join(output_dir, 'test_hoverVer2_crazy_data.pkl')
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"✅ 数据已保存到：{data_path}")
    
    # 打印统计信息
    positions = data['positions_enu']
    velocities = data['velocities_enu']
    times = data['times']
    target = data['hover_target_enu']
    
    distances = np.linalg.norm(positions - target, axis=1)
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    logger.info("\n" + "="*60)
    logger.info("测试统计：")
    logger.info(f"  总时长: {times[-1]:.2f}s")
    logger.info(f"  总步数: {len(times)}")
    logger.info(f"  距离目标:")
    logger.info(f"    平均: {np.mean(distances):.4f}m")
    logger.info(f"    最小: {np.min(distances):.4f}m")
    logger.info(f"    最大: {np.max(distances):.4f}m")
    logger.info(f"    最终: {distances[-1]:.4f}m")
    logger.info(f"  速度:")
    logger.info(f"    平均: {np.mean(velocity_magnitudes):.4f}m/s")
    logger.info(f"    最大: {np.max(velocity_magnitudes):.4f}m/s")
    logger.info(f"    最终: {velocity_magnitudes[-1]:.4f}m/s")
    logger.info("="*60)


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("Crazyflie HoverVer2 神经网络控制测试")
    logger.info("="*60)
    
    # 初始化Crazyflie驱动
    cflib.crtp.init_drivers()
    logger.info("✅ Crazyflie驱动初始化完成")
    
    # 加载神经网络策略
    params, env_config, action_repeat, buffer_size = load_trained_policy(POLICY_FILE)
    
    # 创建环境（用于获取归一化参数）
    env = HoverEnvVer2(
        max_steps_in_episode=2000,
        dt=0.01,
        delay=0.01,
        omega_std=0.1,
        action_penalty_weight=0.1,
        hover_height=2.0,
        init_pos_range=0.5,
        max_distance=10.0,
        max_speed=20.0,
        thrust_to_weight_min=3.8,
        thrust_to_weight_max=4.1,
        disturbance_mag=0.0,
    )
    env = MinMaxObservationWrapper(env)
    env = NormalizeActionWrapper(env)
    
    # 创建神经网络模型
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    input_dim = buffer_size * (obs_dim + action_dim)
    policy = MLP([input_dim, 128, 128, action_dim], initial_scale=0.2)
    
    logger.info(f"观测维度: {obs_dim}")
    logger.info(f"动作维度: {action_dim}")
    logger.info(f"缓冲区大小: {buffer_size}")
    logger.info(f"动作重复: {action_repeat}")
    logger.info(f"输入维度: {input_dim}")
    
    # 连接Crazyflie
    logger.info(f"\n尝试连接到Crazyflie: {CRAZYFLIE_URI}")
    
    try:
        with SyncCrazyflie(CRAZYFLIE_URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            logger.info("✅ Crazyflie连接成功")
            
            # 创建状态管理器
            state_manager = CrazyflieStateManager()
            
            # 创建OptiTrack接收器
            logger.info(f"\n连接OptiTrack服务器: {OPTITRACK_SERVER_IP}")
            optitrack_receiver = OptiTrackReceiver(state_manager, crazyflie=scf.cf)
            
            # 启动OptiTrack接收线程
            if not optitrack_receiver.start():
                logger.error("❌ 无法启动OptiTrack接收器，请检查配置")
                return
            
            # 设置日志回调（姿态和陀螺仪数据仍从日志获取）
            log_configs = setup_logging(scf, state_manager)
            
            # 等待OptiTrack数据准备好
            logger.info("等待OptiTrack数据...")
            time.sleep(2.0)
            
            if not state_manager.data_ready:
                logger.error("❌ 未收到OptiTrack数据，请检查配置")
                logger.error(f"   服务器IP: {OPTITRACK_SERVER_IP}")
                logger.error(f"   本地IP: {OPTITRACK_LOCAL_IP}")
                logger.error(f"   刚体ID: {OPTITRACK_RIGID_BODY_ID}")
                optitrack_receiver.stop()
                return
            
            state = state_manager.get_state_enu()
            logger.info(f"✅ 当前位置(ENU): {state['position']}")
            
            # ==================== 阶段1: 起飞 ====================
            takeoff_x, takeoff_y = takeoff_to_height(scf, state_manager, TAKEOFF_HEIGHT)
            
            # 设置悬停目标（当前位置）
            hover_target_enu = np.array([takeoff_x, takeoff_y, TAKEOFF_HEIGHT])
            logger.info(f"悬停目标(ENU): {hover_target_enu}")
            
            # ==================== 阶段2: 神经网络控制 ====================
            data = neural_network_hover(
                scf, state_manager, policy, params, env,
                hover_target_enu, HOVER_DURATION, action_repeat, buffer_size
            )
            
            # ==================== 阶段3: 降落 ====================
            land_safely(scf, state_manager)
            
            # 停止日志
            for log_config in log_configs:
                log_config.stop()
            
            # 停止OptiTrack接收器
            optitrack_receiver.stop()
            
            # 保存结果
            save_results(data)
            
            logger.info("\n✅ 测试完成！")
    
    except Exception as e:
        logger.error(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

