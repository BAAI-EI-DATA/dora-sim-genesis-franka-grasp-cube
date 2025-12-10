import numpy as np
import genesis as gs
from pynput import keyboard
import math
from keyboard_controller import KeyboardController
from dora import Node
import cv2
import pyarrow as pa

# init
gs.init(backend=gs.gpu, logging_level="warn")
node = Node()

# 创建场景
scene = gs.Scene(
    sim_options=gs.options.SimOptions(),
    viewer_options=gs.options.ViewerOptions(),
    show_viewer=True,
)

# 创建实体
plane = scene.add_entity(gs.morphs.Plane())
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.05, 0.05, 0.05),
        pos=(0.4, 0.0, 0.00),
    )
)
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
cam = scene.add_camera(
    res=(640, 480), pos=(1.0, 0.0, 1.0), lookat=(0, 0, 0.5), fov=60, GUI=False
)

# 构建场景
scene.build()

# 关节索引定义
motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# 设置控制增益
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

# 获取末端执行器
end_effector = franka.get_link("hand")

# 初始化键盘控制器
controller = KeyboardController()

# 初始位置
initial_pos = np.array([0.4, 0.0, 0.4])
initial_euler = np.array([math.pi, 0.0, 0.0])
initial_quat = controller.euler_to_quat(*initial_euler)

controller.target_pos = initial_pos.copy()
controller.target_euler = initial_euler.copy()

# 先移动到初始位置,这个IK感觉有问题
qpos = franka.inverse_kinematics(
    link=end_effector,
    pos=controller.target_pos,
    quat=initial_quat,
)
path = franka.plan_path(
    qpos_goal=qpos,
    num_waypoints=200,  # 2s duration
)
# execute the planned path
for waypoint in path:
    franka.control_dofs_position(waypoint[:-2], motors_dof)
    scene.step()

# allow robot to reach the last waypoint
for i in range(100):
    scene.step()

controller.print_controls()
print("开始键盘控制...")
print("按 'p' 键打印当前位置和姿态")
print("按 ESC 键退出程序")

########################## 主控制循环 ##########################
for event in node:
    if event["type"] == "INPUT":
        if event["id"] == "tick":
            # 更新目标位置和姿态
            controller.update_target_from_keyboard()

            # 检查是否退出
            if keyboard.Key.esc in controller.keys_pressed:
                print("退出程序")
                break

            # 检查是否打印信息
            if "p" in controller.keys_pressed:
                quat = controller.get_target_quat()
                print(f"\n位置: {controller.target_pos}")
                print(f"欧拉角 (度): {np.degrees(controller.target_euler)}")
                print(f"四元数: {quat}")
                # 移除按键避免连续打印
                controller.keys_pressed.discard("p")

            # 获取当前目标四元数
            target_quat = controller.get_target_quat()

            try:
                # 逆运动学计算
                qpos = franka.inverse_kinematics(
                    link=end_effector,
                    pos=controller.target_pos,
                    quat=target_quat,
                )

                # 控制机械臂
                franka.control_dofs_position(qpos[:-2], motors_dof)

                # 控制夹爪
                franka.control_dofs_force(controller.gripper_force, fingers_dof)

            except Exception as e:
                print(f"逆运动学求解失败: {e}")
                print(f"当前位置: {controller.target_pos}, 姿态: {controller.target_euler}")

            scene.step()

        if event["id"] == "tick_image":
            rgb, _, _, _ = cam.render()

            # Send Color Image
            rgb_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            ret, frame = cv2.imencode("." + "jpeg", rgb_image)
            if ret:
                node.send_output(
                    "image",
                    pa.array(frame),
                    {"encoding": "jpeg", "width": int(640), "height": int(480)},
                )

    elif event["type"] == "STOP":
        print("程序被中断")
        break


controller.listener.stop()
print("程序结束")
