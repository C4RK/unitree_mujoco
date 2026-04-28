import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize#宇树官方通讯库
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand#连接仿真和真机的桥梁

import config


locker = threading.Lock()#线程锁。因为两个线程都要访问mj_data（机器人状态）。防止边读边写

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)#机器人的静态信息。在这里更改不同的机器人
mj_data = mujoco.MjData(mj_model)#机器人的动态信息


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()#这个来自bridge.py
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id#带子连接的是躯干
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(#启动一个“被动渲染”窗口。
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )#这里的keycallback相当于监听器。按下键盘，触发功能
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT#物理计算步长，每秒计算多少次
num_motor_ = mj_model.nu#自动读取电机数量
dim_motor_sensor_ = 3 * num_motor_#预留位置。角度，转速，扭矩

time.sleep(0.2)#引入同步延迟


def SimulationThread():#物理线程。高频运行。计算受力、关节运动、DDS 通信
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)#启动了宇树的通讯中间件。
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)#建立桥梁

    if config.USE_JOYSTICK:#初始化手柄
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:#打印关节名称和ID信息
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()#记录开始时间

        locker.acquire()#加锁，开始写数据

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                #把计算出来的悬挂拉力直接施加到挂载点上
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(#挂载点，空间的三个维度，FxFyFz
                    mj_data.qpos[:3], mj_data.qvel[:3]#根据机器人的当前位置和速度计算力
                )
                #xfrc_applied：这是 MuJoCo 给物体施加“外力”的接口。
        #执行物理步进
        mujoco.mj_step(mj_model, mj_data)#根据当前的力、速度、重力，计算出极短时间后机器人应该在哪里

        locker.release()#释放锁
        #精准时间补偿。为了确保时间一致。
        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():#渲染线程。负责更新窗口里的画面。
    while viewer.is_running():
        locker.acquire()#加上锁
        viewer.sync()#读取当前 mj_data 中所有物体的位置、姿态、甚至是接触力，然后瞬间刷新屏幕上的 3D 模型。
        locker.release()#释放锁
        time.sleep(config.VIEWER_DT)#休息


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
