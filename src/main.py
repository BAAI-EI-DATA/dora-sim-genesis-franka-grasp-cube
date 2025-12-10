"""
主程序 - 使用配置文件外部化参数，增强错误处理和性能监控
"""
import numpy as np
import genesis as gs
import math
from keyboard_controller import KeyboardController
from config_loader import get_config_loader
from dora import Node
import cv2
import pyarrow as pa
import time
from collections import deque
import threading
from queue import Queue
import traceback


class FrankaSimulation:
    """Franka机械臂仿真主类"""
    
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """
        初始化仿真环境
        
        Args:
            config_path: 配置文件路径
        """
        self.config_loader = get_config_loader(config_path)
        self.config = self.config_loader.config
        
        # 初始化变量
        self.node = None
        self.scene = None
        self.franka = None
        self.cam = None
        self.controller = None
        self.end_effector = None
        
        # 线程控制
        self.should_exit = threading.Event()
        self.step_data_queue = Queue()
        self.control_thread = None
        
        # 性能监控
        self.step_times = deque(maxlen=100)
        self.step_intervals = deque(maxlen=100)
        self.render_times = deque(maxlen=100)
        self.tick_intervals = deque(maxlen=100)
        
        self.last_tick_time = time.perf_counter()
        self.print_counter = 0
        self.print_interval = self.config['performance']['print_interval']
        
        # 错误统计
        self.ik_failures = 0
        self.total_ik_calls = 0
        
    def initialize_simulation(self):
        """初始化仿真环境"""
        print("初始化仿真环境...")
        
        # 初始化Genesis引擎
        gs.init(backend=gs.gpu, logging_level="warn")
        
        # 创建Dora节点
        self.node = Node()
        
        # 获取仿真配置
        sim_config = self.config['simulation']
        robot_config = self.config['robot']
        
        # 创建场景
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(),
            viewer_options=gs.options.ViewerOptions(),
            show_viewer=True,
        )
        
        # 创建实体
        if sim_config['plane_enabled']:
            plane = self.scene.add_entity(gs.morphs.Plane())
            print("  平面实体已创建")
        
        cube_config = sim_config['cube']
        cube = self.scene.add_entity(
            gs.morphs.Box(
                size=tuple(cube_config['size']),
                pos=tuple(cube_config['position']),
            )
        )
        print(f"  立方体实体已创建: 大小={cube_config['size']}, 位置={cube_config['position']}")
        
        # 创建Franka机械臂
        self.franka = self.scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        print("  Franka机械臂已加载")
        
        # 创建相机
        cam_config = sim_config['camera']
        self.cam = self.scene.add_camera(
            res=tuple(cam_config['resolution']),
            pos=tuple(cam_config['position']),
            lookat=tuple(cam_config['lookat']),
            fov=cam_config['fov'],
            GUI=cam_config['gui_enabled']
        )
        print(f"  相机已创建: 分辨率={cam_config['resolution']}, 位置={cam_config['position']}")
        
        # 构建场景
        self.scene.build()
        print("  场景构建完成")
        
        # 配置机械臂控制参数
        self.franka.set_dofs_kp(np.array(robot_config['kp_gains']))
        self.franka.set_dofs_kv(np.array(robot_config['kv_gains']))
        self.franka.set_dofs_force_range(
            np.array(robot_config['force_min']),
            np.array(robot_config['force_max'])
        )
        print("  机械臂控制参数已配置")
        
        # 获取末端执行器
        self.end_effector = self.franka.get_link("hand")
        
        # 初始化键盘控制器
        self.controller = KeyboardController()
        
        # 设置关节索引
        self.motors_dof = np.array(robot_config['motor_dofs'])
        self.fingers_dof = np.array(robot_config['finger_dofs'])
        
        print("仿真环境初始化完成")
        
    def move_to_initial_position(self):
        """移动到初始位置"""
        print("\n移动到初始位置...")
        
        keyboard_config = self.config['keyboard']
        motion_config = self.config['motion']
        
        initial_pos = np.array(keyboard_config['initial_position'])
        initial_euler = np.array(keyboard_config['initial_euler'])
        initial_quat = self.controller.euler_to_quat(*initial_euler)
        
        self.controller.target_pos = initial_pos.copy()
        self.controller.target_euler = initial_euler.copy()
        
        try:
            # 逆运动学计算
            qpos = self.franka.inverse_kinematics(
                link=self.end_effector,
                pos=self.controller.target_pos,
                quat=initial_quat,
            )
            
            # 路径规划
            path = self.franka.plan_path(
                qpos_goal=qpos,
                num_waypoints=motion_config['initial_path_waypoints'],
            )
            
            print(f"  路径规划完成: {len(path)}个路径点")
            
            # 执行规划路径
            for i, waypoint in enumerate(path):
                self.franka.control_dofs_position(waypoint[:-2], self.motors_dof)
                self.scene.step()
                
                if i % 50 == 0:
                    print(f"    执行路径点 {i}/{len(path)}")
            
            # 等待机械臂到达最后路径点
            for i in range(motion_config['initial_wait_steps']):
                self.scene.step()
                
            print("  初始位置移动完成")
            
        except Exception as e:
            print(f"  移动到初始位置失败: {e}")
            print("  继续执行程序...")
    
    def step_thread_func(self):
        """控制线程函数"""
        last_step_time = time.perf_counter()
        render_counter = 0
        
        while not self.should_exit.is_set():
            try:
                # 更新目标位置和姿态
                self.controller.update_target_from_keyboard()
                
                # 检查是否打印信息
                if "p" in self.controller.keys_pressed:
                    quat = self.controller.get_target_quat()
                    print(f"\n位置: {self.controller.target_pos}")
                    print(f"欧拉角 (度): {np.degrees(self.controller.target_euler)}")
                    print(f"四元数: {quat}")
                    # 移除按键避免连续打印
                    self.controller.keys_pressed.discard("p")
                
                # 获取当前目标四元数
                target_quat = self.controller.get_target_quat()
                
                # 逆运动学计算
                self.total_ik_calls += 1
                try:
                    qpos = self.franka.inverse_kinematics(
                        link=self.end_effector,
                        pos=self.controller.target_pos,
                        quat=target_quat,
                    )
                    
                    # 控制机械臂
                    self.franka.control_dofs_position(qpos[:-2], self.motors_dof)
                    
                    # 控制夹爪
                    self.franka.control_dofs_force(self.controller.gripper_force, self.fingers_dof)
                    
                except Exception as e:
                    self.ik_failures += 1
                    if self.ik_failures % 10 == 0:  # 每10次失败打印一次
                        print(f"逆运动学求解失败 ({self.ik_failures}/{self.total_ik_calls}): {e}")
                        print(f"当前位置: {self.controller.target_pos}, 姿态: {self.controller.target_euler}")
                
                # 测量 scene.step 执行时间和间隔
                step_start = time.perf_counter()
                self.scene.step()
                step_end = time.perf_counter()
                
                # 计算 step 间隔时间
                step_interval = step_start - last_step_time
                last_step_time = step_end
                
                # 渲染相机图像（每2次循环渲染一次以平衡性能）
                render_data = None
                render_counter += 1
                if render_counter >= 4:  # 控制渲染频率
                    render_counter = 0
                    render_start = time.perf_counter()
                    rgb, _, _, _ = self.cam.render()
                    render_end = time.perf_counter()
                    
                    # 将RGB图像转换为BGR格式用于OpenCV
                    rgb_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                    render_data = {
                        'rgb_image': rgb_image,
                        'render_time': render_end - render_start
                    }
                
                # 将统计数据发送回主线程
                self.step_data_queue.put({
                    'step_time': step_end - step_start,
                    'step_interval': step_interval,
                    'render_data': render_data
                })
                    
            except Exception as e:
                print(f"控制线程错误: {e}")
                traceback.print_exc()
    
    def start_control_thread(self):
        """启动控制线程"""
        print("启动控制线程...")
        self.control_thread = threading.Thread(target=self.step_thread_func, daemon=True)
        self.control_thread.start()
        print("控制线程已启动")
    
    def print_performance_stats(self):
        """打印性能统计信息"""
        perf_config = self.config['performance']
        
        if perf_config['enable_tick_stats'] and self.tick_intervals:
            print(f"\n=== tick 调用间隔统计 ===")
            print(f"调用间隔 - 平均: {np.mean(self.tick_intervals)*1000:.2f}ms, "
                  f"最大: {np.max(self.tick_intervals)*1000:.2f}ms, "
                  f"最小: {np.min(self.tick_intervals)*1000:.2f}ms")
            print(f"估计频率: {1/np.mean(self.tick_intervals):.1f}Hz")

        if perf_config['enable_step_stats'] and self.step_intervals:
            print(f"\n=== step 调用间隔统计 ===")
            print(f"调用间隔 - 平均: {np.mean(self.step_intervals)*1000:.2f}ms, "
                  f"最大: {np.max(self.step_intervals)*1000:.2f}ms, "
                  f"最小: {np.min(self.step_intervals)*1000:.2f}ms")
            print(f"估计频率: {1/np.mean(self.step_intervals):.1f}Hz")
        
        if perf_config['enable_step_stats'] and self.step_times:
            print(f"\n=== scene.step 时间统计 ===")
            print(f"执行时间 - 平均: {np.mean(self.step_times)*1000:.2f}ms, "
                  f"最大: {np.max(self.step_times)*1000:.2f}ms, "
                  f"最小: {np.min(self.step_times)*1000:.2f}ms")
        
        if perf_config['enable_render_stats'] and self.render_times:
            print(f"\n=== cam.render 时间统计 ===")
            print(f"执行时间 - 平均: {np.mean(self.render_times)*1000:.2f}ms, "
                  f"最大: {np.max(self.render_times)*1000:.2f}ms, "
                  f"最小: {np.min(self.render_times)*1000:.2f}ms")
        
        if self.total_ik_calls > 0:
            success_rate = (self.total_ik_calls - self.ik_failures) / self.total_ik_calls * 100
            print(f"\n=== 逆运动学统计 ===")
            print(f"调用次数: {self.total_ik_calls}")
            print(f"失败次数: {self.ik_failures}")
            print(f"成功率: {success_rate:.1f}%")
        
        print("=" * 50)
    
    def process_tick_event(self):
        """处理tick事件"""
        # 计算 tick 间隔时间
        current_time = time.perf_counter()
        tick_interval = current_time - self.last_tick_time
        self.tick_intervals.append(tick_interval)
        self.last_tick_time = current_time
        
        # 从线程获取统计数据
        threading_config = self.config['threading']
        try:
            step_data = self.step_data_queue.get(timeout=threading_config['queue_timeout'])
            self.step_times.append(step_data['step_time'])
            self.step_intervals.append(step_data['step_interval'])
            
            # 处理渲染数据（如果存在）
            if step_data.get('render_data') is not None:
                render_data = step_data['render_data']
                self.render_times.append(render_data['render_time'])
                
                # 发送彩色图像
                rgb_image = render_data['rgb_image']
                ret, frame = cv2.imencode("." + "jpeg", rgb_image)
                if ret:
                    self.node.send_output(
                        "image",
                        pa.array(frame),
                        {"encoding": "jpeg", "width": int(640), "height": int(480)},
                    )
        except Exception:
            pass
    
    def run(self):
        """运行主循环"""
        try:
            # 初始化
            self.initialize_simulation()
            self.move_to_initial_position()
            
            # 打印控制说明
            self.controller.print_controls()
            print("开始键盘控制...")
            print("按 'p' 键打印当前位置和姿态")
            print(f"每{self.print_interval}次循环打印一次时间统计信息")
            
            # 启动控制线程
            self.start_control_thread()
            
            # 主事件循环
            print("\n进入主事件循环...")
            for event in self.node:
                if event["type"] == "INPUT":
                    if event["id"] == "tick":
                        self.process_tick_event()
                        
                        # 定期打印统计信息
                        self.print_counter += 1
                        if self.print_counter >= self.print_interval:
                            self.print_counter = 0
                            self.print_performance_stats()
                
                elif event["type"] == "STOP":
                    print("程序被中断")
                    break
            
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"程序运行错误: {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        
        # 停止控制线程
        if self.should_exit:
            self.should_exit.set()
        
        # 等待控制线程结束
        if self.control_thread:
            threading_config = self.config['threading']
            self.control_thread.join(timeout=threading_config['step_thread_timeout'])
            if self.control_thread.is_alive():
                print("警告: 控制线程未正常结束")
            else:
                print("控制线程已结束")
        
        # 清理键盘控制器
        if self.controller:
            self.controller.cleanup()
            print("键盘控制器已清理")
        
        print("程序结束")


def main():
    """主函数"""
    print("=" * 60)
    print("Franka机械臂抓取立方体仿真系统 - 改进版")
    print("=" * 60)
    
    # 创建并运行仿真
    simulation = FrankaSimulation()
    simulation.run()


if __name__ == "__main__":
    main()
