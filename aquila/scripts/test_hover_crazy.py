#!/usr/bin/env python
# coding: utf-8

"""
Crazyflie实机测试脚本 - 位置控制悬停
使用Crazyswarm提供的位置估计作为网络输入

流程：
1. 初始化Crazyswarm（自动处理OptiTrack/外部定位系统）
2. 起飞到0.5m高度
3. 使用位置控制API悬停在当前点
4. 记录位置、速度等状态数据
5. 降落

坐标系说明：
- Crazyswarm使用标准ENU坐标系 (East-North-Up)
"""

import os
import sys
import time
import pickle
import numpy as np
import logging

# ==================== 导入Crazyswarm库 ====================
try:
    from pycrazyswarm import Crazyswarm
except ImportError:
    print("❌ 错误：未找到pycrazyswarm库，请安装：pip install pycrazyswarm")
    print("   或按照官方文档编译安装：https://crazyswarm.readthedocs.io/")
    sys.exit(1)

# ==================== 配置 ====================
# 控制参数
TAKEOFF_HEIGHT = 0.5  # 起飞高度 (m, ENU坐标系)
HOVER_DURATION = 10.0  # 悬停时长 (s)
CONTROL_FREQ = 30  # 控制频率 (Hz) - Crazyswarm推荐30Hz
dt = 1.0 / CONTROL_FREQ  # 控制周期 (s)

# 起飞降落参数
TAKEOFF_DURATION = 2.0  # 起飞时长 (s)
LAND_DURATION = 2.0  # 降落时长 (s)

# 日志配置
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ==================== Crazyflie状态管理器 ====================
class CrazyflieStateManager:
    """管理Crazyflie的状态信息（从Crazyswarm获取位置估计）"""
    
    def __init__(self, cf):
        """
        Args:
            cf: Crazyswarm的Crazyflie对象
        """
        self.cf = cf
        
        # 上一时刻的位置（用于计算速度）
        self.last_position = None
        self.last_time = time.time()
    
    def get_state_enu(self):
        """获取当前状态（ENU坐标系）- 从Crazyswarm获取"""
        # 从Crazyswarm获取位置估计
        position = self.cf.position()  # 返回 [x, y, z] in ENU
        
        # 计算速度（数值微分）
        current_time = time.time()
        if self.last_position is not None:
            dt = current_time - self.last_time
            if dt > 0:
                velocity = (position - self.last_position) / dt
            else:
                velocity = np.zeros(3)
        else:
            velocity = np.zeros(3)
        
        # 更新历史数据
        self.last_position = position.copy()
        self.last_time = current_time
        
        return {
            'position': position,
            'velocity': velocity,
            'data_ready': True  # Crazyswarm始终提供位置数据
        }


# ==================== 主控制流程 ====================
def takeoff_to_height(cf, state_manager, target_height, duration=TAKEOFF_DURATION):
    """
    使用Crazyswarm API起飞到指定高度
    
    Args:
        cf: Crazyswarm的Crazyflie对象
        state_manager: 状态管理器
        target_height: 目标高度 (m, ENU坐标系)
        duration: 起飞时长 (s)
    """
    logger.info(f"起飞到高度 {target_height:.2f}m (时长 {duration:.1f}s)...")
    
    # 使用Crazyswarm的takeoff命令
    cf.takeoff(targetHeight=target_height, duration=duration)
    time.sleep(duration + 0.5)  # 等待起飞完成并稳定
    
    # 获取起飞后的位置
    state = state_manager.get_state_enu()
    takeoff_x = state['position'][0]
    takeoff_y = state['position'][1]
    current_z = state['position'][2]
    
    logger.info(f"✅ 起飞完成，当前位置: ({takeoff_x:.3f}, {takeoff_y:.3f}, {current_z:.3f})")
    
    return takeoff_x, takeoff_y


def position_control_hover(cf, timeHelper, state_manager, hover_target_enu, duration):
    """
    使用Crazyswarm位置控制API实现悬停
    
    Args:
        cf: Crazyswarm的Crazyflie对象
        timeHelper: Crazyswarm的TimeHelper对象
        state_manager: 状态管理器
        hover_target_enu: 悬停目标位置（ENU坐标系）[x, y, z]
        duration: 悬停时长 (s)
    
    Returns:
        data: 记录的数据字典
    """
    logger.info(f"开始位置控制悬停（目标：{hover_target_enu}，时长：{duration:.1f}s）...")
    
    # 数据记录
    positions_enu = []
    velocities_enu = []
    times = []
    
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
            
            # 获取当前状态（从Crazyswarm）
            state = state_manager.get_state_enu()
            
            # 记录数据
            positions_enu.append(state['position'].copy())
            velocities_enu.append(state['velocity'].copy())
            times.append(current_time)
            
            # 使用Crazyswarm的goTo命令发送目标位置
            # goTo(goal, yaw, duration, relative=False)
            cf.goTo(hover_target_enu, yaw=0.0, duration=dt, relative=False)
            
            # 每30步打印一次状态（约1秒）
            if step % 30 == 0:
                pos = state['position']
                vel = state['velocity']
                distance = np.linalg.norm(pos - hover_target_enu)
                logger.info(f"[{current_time:.2f}s] 位置(ENU): ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), "
                           f"速度: {np.linalg.norm(vel):.3f}m/s, 距离: {distance:.3f}m")
            
            step += 1
            
            # 控制频率（使用timeHelper保持同步）
            timeHelper.sleepForRate(CONTROL_FREQ)
    
    except KeyboardInterrupt:
        logger.info("用户中断")
    
    # 返回记录的数据
    return {
        'positions_enu': np.array(positions_enu),
        'velocities_enu': np.array(velocities_enu),
        'times': np.array(times),
        'hover_target_enu': hover_target_enu,
    }


def land_safely(cf, state_manager, duration=LAND_DURATION):
    """
    使用Crazyswarm API安全降落
    
    Args:
        cf: Crazyswarm的Crazyflie对象
        state_manager: 状态管理器
        duration: 降落时长 (s)
    """
    logger.info(f"开始降落 (时长 {duration:.1f}s)...")
    
    # 获取降落前位置
    state = state_manager.get_state_enu()
    logger.info(f"降落前位置: ({state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f})")
    
    # 使用Crazyswarm的land命令
    cf.land(targetHeight=0.02, duration=duration)  # 降落到2cm高度
    time.sleep(duration + 0.5)  # 等待降落完成
    
    logger.info("✅ 降落完成")


def save_results(data, output_dir='aquila/output'):
    """保存测试结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存数据
    data_path = os.path.join(output_dir, 'test_hover_crazy_data.pkl')
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
    logger.info("Crazyflie 位置控制悬停测试 (使用Crazyswarm)")
    logger.info("="*60)
    
    try:
        # 初始化Crazyswarm
        logger.info("\n初始化Crazyswarm...")
        swarm = Crazyswarm()
        timeHelper = swarm.timeHelper
        allcfs = swarm.allcfs
        
        if len(allcfs.crazyflies) == 0:
            logger.error("❌ 未找到Crazyflie，请检查crazyswarm配置文件")
            return
        
        # 获取第一架无人机
        cf = allcfs.crazyflies[0]
        logger.info(f"✅ Crazyswarm初始化完成，连接到 Crazyflie: {cf.id}")
        
        # 创建状态管理器
        state_manager = CrazyflieStateManager(cf)
        
        # 等待位置估计稳定
        logger.info("\n等待位置估计稳定...")
        time.sleep(2.0)
        
        # 获取当前位置
        state = state_manager.get_state_enu()
        logger.info(f"✅ 当前位置(ENU): ({state['position'][0]:.3f}, {state['position'][1]:.3f}, {state['position'][2]:.3f})")
        
        # ==================== 阶段1: 起飞 ====================
        takeoff_x, takeoff_y = takeoff_to_height(cf, state_manager, TAKEOFF_HEIGHT)
        
        # 设置悬停目标（当前位置）
        hover_target_enu = np.array([takeoff_x, takeoff_y, TAKEOFF_HEIGHT])
        logger.info(f"\n悬停目标(ENU): ({hover_target_enu[0]:.3f}, {hover_target_enu[1]:.3f}, {hover_target_enu[2]:.3f})")
        
        # ==================== 阶段2: 位置控制悬停 ====================
        data = position_control_hover(
            cf, timeHelper, state_manager,
            hover_target_enu, HOVER_DURATION
        )
        
        # ==================== 阶段3: 降落 ====================
        land_safely(cf, state_manager)
        
        # 保存结果
        save_results(data)
        
        logger.info("\n✅ 测试完成！")
    
    except Exception as e:
        logger.error(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

