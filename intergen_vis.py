import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegFileWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import shutil
import subprocess
import tempfile

t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
up_axis = 2

def plot_3d_motion_alternative(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4, up_axis=1):
    """
    通过先生成图像序列，再用 ffmpeg 合成视频的方式来可视化3D动作。
    """
    # 1. 创建一个唯一的临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时帧文件将保存在: {temp_dir}")

    # --- 数据预处理和设置代码 (基本不变) ---
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    fig = plt.figure(figsize=figsize)
    try:
        ax = fig.add_subplot(111, projection='3d')
    except AttributeError:
        ax = p3.Axes3D(fig)

    def init():
        """这个函数现在会在每一帧被调用，以重置绘图环境"""
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)
        ax.view_init(azim=-60, elev=30) 
        if hasattr(ax, 'dist'):
            ax.dist = 20
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # 数据预处理部分
    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])

    colors = ['red', 'green', 'blue', 'orange', 'purple',
              'darkblue', 'darkgreen', 'darkred', 'darkorange', 'darkviolet']

    # 确保颜色列表足够长
    mp_colors = [colors[i % len(colors)] for i in range(len(mp_joints))]

    for i, joints in enumerate(mp_joints):
        data = joints.copy().reshape(len(joints), -1, 3)
        height_offset = data.min(axis=0).min(axis=0)[up_axis]
        data[:, :, up_axis] -= height_offset
        mp_data.append({"joints": data})

    def plot_ground_plane(ax, radius):
        """绘制地平面"""
        verts = [[-radius, -radius, 0], [-radius, radius, 0], [radius, radius, 0], [radius, -radius, 0]]
        ground = Poly3DCollection([verts], facecolors='lightgrey', linewidths=0, alpha=0.2)
        ax.add_collection3d(ground)

    # --- 核心修改部分 ---
    try:
        # 2. 循环生成每一帧图像
        for i in range(frame_number):
            ax.cla()
            init()
            plot_ground_plane(ax, radius)

            # 绘制当前帧的骨架
            for pid, data in enumerate(mp_data):
                person_color = mp_colors[pid]
                # 获取当前人物对应的 mask
                
                for chain in kinematic_tree:
                    # 获取 chain 连接的两个关节点的索引
                    joint1_idx, joint2_idx = chain[0], chain[1]

                    # >>> 主要逻辑修改点 <<<
                    # 检查当前帧下，这两个关节点是否都被 mask 标记
                    # mask 的形状是 (num_joints, )

                    # 根据 mask 决定线条颜色
                    line_color = person_color
                    
                    linewidth = 2.0
                    ax.plot3D(data["joints"][i, chain, 0],   # X 轴
                              data["joints"][i, chain, 1],   # Y 轴
                              data["joints"][i, chain, 2],   # Z 轴 (up)
                              linewidth=linewidth, color=line_color)

            # 将当前帧保存为PNG文件
            frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
            plt.savefig(frame_path, dpi=80) 

            # 打印进度
            if (i+1) % 100 == 0:
                print(f"已生成 {i+1}/{frame_number} 帧...")

        # 3. 使用 ffmpeg 将图像序列合成为视频
        print("所有帧已生成, 开始使用 ffmpeg 合成视频...")
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        ffmpeg_cmd = [
            'ffmpeg',
            '-r', str(fps),                   
            '-i', f'{temp_dir}/frame_%05d.png',
            '-vcodec', 'libx264',             
            '-pix_fmt', 'yuv420p',            
            '-y',                             
            save_path
        ]
        result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"成功保存动画到 {save_path}")

    except subprocess.CalledProcessError as e:
        print("ffmpeg 合成视频失败！")
        print("FFMPEG stdout:", e.stdout)
        print("FFMPEG stderr:", e.stderr)
    except Exception as e:
        print(f"生成动画时发生错误: {e}")
    finally:
        # 4. 清理临时文件夹
        print(f"清理临时文件夹: {temp_dir}")
        shutil.rmtree(temp_dir)

    plt.close(fig)


def plot_3d_motion(save_path, kinematic_tree, mp_joints, title, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    def init():
        ax.set_xlim3d([-radius / 4, radius / 4])
        ax.set_ylim3d([0, radius / 2])
        ax.set_zlim3d([0, radius / 2])
        # print(title)
        fig.suptitle(title, fontsize=20)
        ax.grid(b=False)
        
    def plot_xyPlane(minx, maxx, miny, maxy, minz):
        ## Plot a plane XY
        verts = [
            [minx, miny, minz],
            [minx, maxy, minz],
            [maxx, maxy, minz],
            [maxx, miny, minz]
        ]
        xy_plane = Poly3DCollection([verts])
        xy_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xy_plane)
        
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    init()

    mp_data = []
    frame_number = min([data.shape[0] for data in mp_joints])
    print(frame_number)

    # colors = ['red', 'blue', 'black', 'red', 'blue',
    #           'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
    #           'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    #
    colors = ['red', 'green', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    mp_offset = list(range(-len(mp_joints)//2, len(mp_joints)//2, 1))
    mp_colors = [[colors[i]] * 15 for i in range(len(mp_offset))]

    for i,joints in enumerate(mp_joints):

        # (seq_len, joints_num, 3)
        data = joints.copy().reshape(len(joints), -1, 3)

        MINS = data.min(axis=0).min(axis=0)
        MAXS = data.max(axis=0).max(axis=0)

        height_offset = MINS[up_axis]
        data[:, :, up_axis] -= height_offset
        trajec = data[:, 0, [0, 1]]

        mp_data.append({"joints":data,
                        "MINS":MINS,
                        "MAXS":MAXS,
                        "trajec":trajec, })

    def update(index):
        ax.lines = []
        ax.collections = []
        # ax.view_init(elev=120, azim=-90)
        ax.dist = 15#7.5
        #         ax =
        plot_xyPlane(-6, 6, -6, 6, 0)
        for pid,data in enumerate(mp_data):
            for i, (chain, color) in enumerate(zip(kinematic_tree, mp_colors[pid])):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data["joints"][index, chain, 0], data["joints"][index, chain, 1], data["joints"][index, chain, 2], linewidth=linewidth,
                          color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False)

    writer = FFMpegFileWriter(fps=fps)
    # writer = PillowWriter(fps=fps)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"Saved animation to {save_path}")

def plot_t2m(mp_data, save_path, caption): # Changed result_path to save_path for clarity
    mp_joint = []
    for i, data in enumerate(mp_data):
        if i == 0:
            joint = data[:,:22*3].reshape(-1,22,3)
        else:
            joint = data[:,:22*3].reshape(-1,22,3)

        mp_joint.append(joint)

    # Pass the full path with the extension directly
    plot_3d_motion_alternative(save_path, t2m_kinematic_chain, mp_joint, title=caption, fps=30)

def generate_one_sample(motion_output_multiple, name, result_path):
    # Append the .mp4 extension to the result path
    # This is the fix!
    os.makedirs(result_path, exist_ok=True)
    save_path = os.path.join(result_path, f"{name}.mp4")
    
    # Pass the corrected path to the plotting function
    if motion_output_multiple[0].ndim == 2:
        plot_t2m(motion_output_multiple, save_path, name)
    elif motion_output_multiple[0].ndim == 3:
        plot_t2m([motion[0] for motion in motion_output_multiple], save_path, name)
    else:
        raise ValueError(f"Invalid number of motions: {len(motion_output_multiple)}")