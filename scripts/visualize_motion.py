"""
Render Motion to Video
======================
Converts generated motion to stick figure video.
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def render_motion_video(motion: np.ndarray, output_path: str = "outputs/motion_video.mp4", fps: int = 20):
    """Render motion as stick figure animation."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    
    n_frames = len(motion)
    
    # Extract root trajectory and simplified joint data
    root_vel = motion[:, :3]
    
    # Simulate root position by integrating velocity
    root_pos = np.cumsum(root_vel / fps, axis=0)
    
    # Create simple humanoid from joint data (simplified 2D projection)
    # Use dims 3-24 as approximate joint positions
    joint_data = motion[:, 3:24].reshape(n_frames, 7, 3)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Top-down view of trajectory
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-1, 10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z (forward)')
    ax1.set_title('Root Trajectory (Top View)')
    ax1.grid(True, alpha=0.3)
    trajectory_line, = ax1.plot([], [], 'b-', alpha=0.5, linewidth=1)
    current_pos, = ax1.plot([], [], 'ro', markersize=10)
    
    # Right: Stick figure side view
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.5, 2.5)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y (height)')
    ax2.set_title('Stick Figure (Side View)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Stick figure elements
    joints = ax2.scatter([], [], s=80, c='blue', zorder=5)
    
    # Simple skeleton connections (head, torso, arms, legs)
    bones = []
    bone_connections = [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5), (1, 6), (6, 7)]  # Simplified
    for _ in range(len(bone_connections)):
        line, = ax2.plot([], [], 'b-', linewidth=3)
        bones.append(line)
    
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=12)
    
    def init():
        trajectory_line.set_data([], [])
        current_pos.set_data([], [])
        joints.set_offsets(np.zeros((7, 2)))
        for bone in bones:
            bone.set_data([], [])
        time_text.set_text('')
        return [trajectory_line, current_pos, joints, time_text] + bones
    
    def animate(frame):
        # Update trajectory
        trajectory_line.set_data(root_pos[:frame+1, 0], root_pos[:frame+1, 2])
        current_pos.set_data([root_pos[frame, 0]], [root_pos[frame, 2]])
        
        # Update stick figure
        jd = joint_data[frame]
        # Normalize and position joints
        jd_norm = jd - jd.mean(axis=0)
        jd_norm = jd_norm * 0.3 + np.array([0, 1, 0])  # Scale and center
        
        # 2D projection (X, Y for side view)
        joint_positions = np.column_stack([jd_norm[:, 0], jd_norm[:, 1]])
        joints.set_offsets(joint_positions)
        
        # Update bones
        for i, (j1, j2) in enumerate(bone_connections):
            if j1 < 7 and j2 < 7:
                bones[i].set_data(
                    [joint_positions[j1, 0], joint_positions[j2, 0]],
                    [joint_positions[j1, 1], joint_positions[j2, 1]]
                )
        
        time_text.set_text(f'Frame: {frame}/{n_frames} | Time: {frame/fps:.2f}s')
        
        return [trajectory_line, current_pos, joints, time_text] + bones
    
    print(f"Rendering {n_frames} frames at {fps}fps...")
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=1000/fps, blit=True)
    
    # Save as video
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"Video saved: {output_path}")
    except Exception as e:
        # Fallback to GIF
        gif_path = output_path.replace('.mp4', '.gif')
        anim.save(gif_path, writer='pillow', fps=fps)
        print(f"GIF saved: {gif_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/generated_motion.npy")
    parser.add_argument("--output", default="outputs/motion_video.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Motion file not found: {args.input}")
        print("Generate motion first: py scripts/test_motion_ssm.py 'A person walks'")
        return
    
    motion = np.load(args.input)
    print(f"Loaded motion: {motion.shape}")
    
    render_motion_video(motion, args.output, args.fps)


if __name__ == "__main__":
    main()
