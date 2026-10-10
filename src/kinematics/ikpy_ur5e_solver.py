"""
IKPy baselines for the UR5e.

ikpy_chain_from_dh() builds an IKPy chain from the same Denavit-Hartenberg table
as every other solver in this repository (src/kinematics/jacobian.py), so IKPy
solves exactly the same problem on the same targets: position plus a fixed tool
orientation. This is the chain used by experiments/benchmark_solveurs.py.

IKPYUR5eSolver loads src/kinematics/ur5e.urdf. That URDF does not describe the
same chain as the DH table (no tool offset, offsets on other joints): for the
same joint angles the two place the end-effector 1.3 to 1.8 m apart. It is kept
for reference only.
"""
import numpy as np
import ikpy.chain
from ikpy.link import OriginLink, URDFLink
from scipy.spatial.transform import Rotation


def ikpy_chain_from_dh(dh):
    """
    dh: rows [a, alpha, d, theta_offset], standard DH, one per revolute joint.

    Joint i contributes Rz(q_i + off_i) Tz(d_i) Tx(a_i) Rx(alpha_i). Rz and Tz
    commute, so link i is a fixed transform Tx(a_{i-1}) Rx(alpha_{i-1}) Tz(d_i)
    Rz(off_i) followed by a rotation q_i about z; a last fixed link carries
    Tx(a_6) Rx(alpha_6).
    """
    links = [OriginLink()]
    a_prev, al_prev = 0.0, 0.0
    for i, (a, al, d, off) in enumerate(dh):
        R = Rotation.from_euler('x', al_prev).as_matrix() @ Rotation.from_euler('z', off).as_matrix()
        links.append(URDFLink(
            name=f"joint_{i + 1}",
            origin_translation=np.array([a_prev, -d * np.sin(al_prev), d * np.cos(al_prev)]),
            origin_orientation=Rotation.from_matrix(R).as_euler('xyz'),
            rotation=np.array([0.0, 0.0, 1.0]),
            bounds=(-2 * np.pi, 2 * np.pi)))
        a_prev, al_prev = a, al
    links.append(URDFLink(
        name="tool",
        origin_translation=np.array([a_prev, 0.0, 0.0]),
        origin_orientation=np.array([al_prev, 0.0, 0.0]),
        joint_type='fixed'))
    mask = [False] + [True] * len(dh) + [False]
    return ikpy.chain.Chain(links, active_links_mask=mask, name="ur5e_dh")


class IKPYUR5eSolver:
    def __init__(self, urdf_path="ur5e.urdf"):
        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, True, True, True]
        )

    def solve_ik(self, target_xyz_robot, target_orientation=None):
        """Position-only unless a 3x3 target_orientation is given."""
        if target_orientation is not None:
            R = np.eye(4)
            R[:3, :3] = target_orientation
            R[:3, 3] = target_xyz_robot
            q_full = self.chain.inverse_kinematics_frame(R, orientation_mode="all")
        else:
            q_full = self.chain.inverse_kinematics(target_xyz_robot)

        q_joints = q_full[1:]
        T_fk = self.chain.forward_kinematics(q_full)
        err_mm = np.linalg.norm(T_fk[:3, 3] - target_xyz_robot) * 1000.0
        return q_joints, True, err_mm
