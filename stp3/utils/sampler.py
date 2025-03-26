# sampler.py
# trajectory sampler
# including a mix of clothoids, straight lines, and circles
import numpy as np
import torch
from scipy.special import fresnel

def sample(v0, Kappa, T0, N0, tt, M, possibility=None):
    '''
    :param v0: 初始速度
    :param Kappa: 曲率
    :param T0: 初始切向量
    :param N0: 初始法向量
    :param tt: 时间序列
    :param M: 采样数量
    :param possibility: 概率分布 [左转概率，直行概率，右转概率]
    :return: 轨迹数组
    '''
    if possibility is None:
        possibility = [0.4, 0.2, 0.4]

    # 计算各类型轨迹数量
    straight_num = int(M * possibility[1])
    left_num = int(M * possibility[0])
    right_num = M - straight_num - left_num

    # 采样初始速度（确保速度非负）
    v_options = np.stack((np.full(M, v0), 15*np.random.rand(M)))
    v_selections = (np.random.rand(M) >= 0.2).astype(int)
    velocities = np.clip(v_options[v_selections, np.arange(M)], 0.1, None)  # 确保最小速度0.1m/s

    # 安全加速度采样
    t_max = np.max(tt)
    a_min = -velocities / t_max  # 计算最小允许加速度
    a_max = 7  # 保持原有最大加速度
    accelerations = np.random.rand(M) * (a_max - a_min) + a_min  # 在安全范围内采样

    # 生成纵向位移
    L = velocities[:, None] * tt[None, :] + accelerations[:, None] * (tt[None, :]**2) / 2
    
    # 直行轨迹生成
    line_points = L[:straight_num, :, None] * T0[None, None, :]
    lines = np.concatenate((line_points, np.zeros_like(line_points[..., :1])), axis=-1)

    # Clothoid曲线参数
    alphas = (300 - 10) * np.random.rand(left_num + right_num) + 10
    Xi0 = np.abs(Kappa) / np.pi
    Xis = Xi0 + L[straight_num:]

    # 计算Fresnel积分
    Ss, Cs = fresnel(Xis / alphas[:, None])

    # 生成Clothoid曲线点
    clothoid_points = alphas[:, None, None] * (Cs[:, :, None]*T0 + Ss[:, :, None]*N0)
    Xs = clothoid_points[..., 0] - clothoid_points[:, 0:1, 0]
    Ys = clothoid_points[..., 1] - clothoid_points[:, 0:1, 1]

    # 旋转调整
    clothoid_theta0s = 0.5 * np.pi * ((Kappa / np.pi / alphas) ** 2)[:, None]
    signed_theta0s = clothoid_theta0s * np.sign(Kappa)
    clothoid_points[..., 0] = np.cos(signed_theta0s)*Xs + np.sin(signed_theta0s)*Ys
    clothoid_points[..., 1] = -np.sin(signed_theta0s)*Xs + np.cos(signed_theta0s)*Ys

    # 计算方向角
    clothoid_thetas = 0.5 * np.pi * ((Xis / alphas[:, None])**2) - clothoid_theta0s
    wrapped_thetas = (clothoid_thetas * np.sign(Kappa) + np.pi) % (2*np.pi) - np.pi
    clothoids = np.concatenate((clothoid_points, wrapped_thetas[..., None]), axis=-1)

    # 处理左右转向
    if Kappa > 0:
        left_traj = clothoids[:left_num]
        right_traj = clothoids[left_num:].copy()
        right_traj[..., [0, 2]] *= -1  # x坐标和方向角取反
    else:
        right_traj = clothoids[:left_num]
        left_traj = clothoids[left_num:].copy()
        left_traj[..., [0, 2]] *= -1

    # 合并轨迹并排序
    trajectories = np.concatenate([left_traj, lines, right_traj], axis=0)
    return trajectories[np.argsort(trajectories[:, -1, 0])]

def sample_cp(v0, Kappa, T0, N0, tt, M, possibility = None):
    '''
    :param v0: initial velocity
    :param Kappa: curvature
    :param T0: initial tangent vector
    :param N0: initial normal vector
    :param tt: time stamp
    :param M: the number of sample
    :param possibility: torch.Tensor [3]
    :param debug: whether in debug mode
    :return: the nparray of trajectory
    '''
    # sample accelerations
    if possibility is None:
        possibility = [0.4, 0.2, 0.4]

    straight_num = int(M * possibility[1])
    left_num = int(M * possibility[0])
    right_num = int(M * possibility[2])

    accelerations = 10*(np.random.rand(M)-0.5) + 2  # -3m/s^2 to 7m/s^2

    # sample velocities
    # randomly sample a velocity <=15m/s at 80% of time
    v_options = np.stack((np.full(M, v0), 15*np.random.rand(M)))
    v_selections = (np.random.rand(M) >= 0.2).astype(int)
    velocities = v_options[v_selections, np.arange(M)]

    # generate longitudinal distances
    L = velocities[:, None] * tt[None, :] + accelerations[:, None] * (tt[None, :]**2) / 2
    L_straight = L[:straight_num]
    L = L[straight_num:]
    # print("L:", L)

    # scaling factor which determine the Clothiod curve  6 ~ 80
    alphas = (500 - 10) * np.random.rand(left_num + right_num) + 10

    ############################################################################
    # sample M straight lines
    line_points = L_straight[:, :, None] * T0[None, None, :]
    line_thetas = np.zeros_like(L_straight)
    lines = np.concatenate((line_points, line_thetas[:, :, None]), axis=-1)

    ############################################################################
    # sample M circles
    Krappa = min(-0.01, Kappa) if Kappa <= 0 else max(0.01, Kappa)
    radius = np.abs(1 / Krappa)
    center = np.array([-1 / Krappa, 0])
    circle_phis = L / radius if Krappa >= 0 else np.pi - L/radius

    circle_points = np.dstack([
        center[0] + radius * np.cos(circle_phis),
        center[1] + radius * np.sin(circle_phis),
    ])

    # rotate thetas, wrap
    circle_thetas = L/radius if Krappa >= 0 else -L/radius
    circle_thetas = (circle_thetas + np.pi) % (2 * np.pi) - np.pi
    circles = np.concatenate((circle_points, circle_thetas[:, :, None]), axis=-1)

    ############################################################################
    # sample M clothoids
    # Xi0 = Kappa / np.pi
    # Xis = Xi0 + L
    Xi0 = np.abs(Kappa) / np.pi
    Xis = Xi0 + L

    #
    # Ss, Cs = fresnel((Xis - Xi0) / alphas[:, None])
    Ss, Cs = fresnel(Xis / alphas[:, None])

    clothoid_points = alphas[:, None, None] * (Cs[:, :, None]*T0[None, None, :] + Ss[:, :, None]*N0[None, None, :])

    #
    Xs = clothoid_points[:, :, 0] - clothoid_points[:, 0, 0, None]
    Ys = clothoid_points[:, :, 1] - clothoid_points[:, 0, 1, None]
    clothoid_theta0s = 0.5 * np.pi * ((Kappa / np.pi / alphas) ** 2)
    clothoid_theta0s = clothoid_theta0s[:, None]
    signed_clothoid_theta0s = clothoid_theta0s * np.sign(Kappa)
    # when kappa is positive, the clothoid curves left, theta is positive
    # we will rotate it clockwise by theta
    # when kappa is negative, the clothoid curves right, theta is negative
    # we will rotate it counterclockwise by theta
    clothoid_points[:, :, 0] = np.cos(signed_clothoid_theta0s) * Xs + np.sin(signed_clothoid_theta0s) * Ys
    clothoid_points[:, :, 1] = - np.sin(signed_clothoid_theta0s) * Xs + np.cos(signed_clothoid_theta0s) * Ys

    # tangent vector: http://mathworld.wolfram.com/CornuSpiral.html
    clothoid_thetas = 0.5 * np.pi * ((Xis / alphas[:, None])**2)
    clothoid_thetas = clothoid_thetas - clothoid_theta0s
    signed_clothoid_thetas = clothoid_thetas * np.sign(Kappa)
    # clothoid_thetas = clothoid_thetas if Krappa >= 0 else -clothoid_thetas
    # wrap
    # clothoid_thetas = (clothoid_thetas + np.pi) % (2 * np.pi) - np.pi
    wrapped_signed_clothoid_thetas = (signed_clothoid_thetas + np.pi) % (2 * np.pi) - np.pi
    # wrapped_signed_clothoid_thetas = (signed_clothoid_thetas) % (2 * np.pi)
    #
    clothoids = np.concatenate((clothoid_points, wrapped_signed_clothoid_thetas[:, :, None]), axis=-1)

    ############################################################################
    # pick M in total
    t_options = np.stack((circles, clothoids))
    t_selections = np.random.choice([0, 1], size=left_num + right_num, p=(0.2, 0.8))
    ############################################################################

    trajs = t_options[t_selections, np.arange(left_num + right_num)]

    # toss a coin for vertical flipping
    # left_possibility = possibility[1] / (possibility[1] + possibility[2])
    # if Kappa > 0:
    #     heads = (np.random.rand(M) <= left_possibility.item())
    # else:
    #     heads = (np.random.rand(M) <= (1- left_possibility).item())
    # tails = np.logical_not(heads)
    #
    # # NOTE theta means what here
    # conditions = [heads[:, None, None], tails[:, None, None]]
    # choices = [trajs, np.dstack((
    #     -trajs[:, :, 0], trajs[:, :, 1], -trajs[:, :, 2]
    # ))]
    #
    # trajectories = np.select(conditions, choices)
    if Kappa > 0:
        left_curve = trajs[: left_num]
        right_curve = trajs[left_num: left_num + right_num]
        right_curve = np.dstack((
            -right_curve[:, :, 0], right_curve[:, :, 1], -right_curve[:, :, 2]
        ))
    else:
        right_curve = trajs[: left_num]
        left_curve = trajs[left_num: left_num + right_num]
        left_curve = np.dstack((
            -left_curve[:, :, 0], left_curve[:, :, 1], -left_curve[:, :, 2]
        ))

    trajectories = np.concatenate([left_curve, lines, right_curve], axis=0)
    mask = np.argsort(trajectories[:, -1, 0])
    trajectories = trajectories[mask]

    return trajectories

if __name__ == "__main__":
    from nuscenes.nuscenes import NuScenes
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    import matplotlib.pyplot as plt
    nusc = NuScenes("v1.0-mini", '/home/hsc/data/Nuscenes')
    nusc_can = NuScenesCanBus(dataroot='/home/hsc/data/Nuscenes')

    for scene in nusc.scene:
        scene_name = scene["name"]
        scene_id = int(scene_name[-4:])
        if scene_id in nusc_can.can_blacklist:
            print(f"skipping {scene_name}")
            continue
        pose = nusc_can.get_messages(scene_name, "pose") # The current pose of the ego vehicle, sampled at 50 HZ
        saf = nusc_can.get_messages(scene_name, "steeranglefeedback") # Steering angle feedback in radians at 100 HZ
        vm = nusc_can.get_messages(scene_name, "vehicle_monitor") # information most, but sample at 2 HZ
        # NOTE: I tried to verify if the relevant measurements are consistent
        # across multiple tables that contain redundant information
        # NOTE: verified pose's velocity matches vehicle monitor's
        # but the pose table offers at a much higher frequency
        # NOTE: same that steeranglefeedback's steering angle matches vehicle monitor's
        # but the steeranglefeedback table offers at a much higher frequency
        print(pose[23])
        print(saf[45])
        print(vm[0])

        # initial velocity (m/s)
        v0 = pose[23]["vel"][0]
        # curvature
        #Kappa = 2 * saf[45]["value"] / 2.588  # 2 x \phi / distance between front and rear
        Kappa = 0
        # T0: longitudinal axis  Tangent vector
        T0 = np.array([0.0, 1.0])
        # N0: normal directional vector  Normal vector
        N0 = np.array([1.0, 0.0]) if Kappa <= 0 else np.array([-1.0, 0.0])
        # tt: time stamps
        tt = np.arange(0.0, 3.01, 0.01)
        # M: number of samples
        M = 1800
        #
        debug = False
        #
        trajectories = sample(v0, Kappa, T0, N0, tt, M)

        trajectories = trajectories[:,::100]

        #
        for i in range(len(trajectories)):
            trajectory = trajectories[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1])
        plt.grid(False)
        plt.axis("equal")

        plt.show()

        break
