import os
import pickle
import yaml
import numpy as np
import torch
import argparse

from scipy.spatial.transform import Rotation as Rot
from mano_pybullet.hand_model import HandModel45

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--th_full_pose_path",
    type=str,
    help="Path to the numpy file storing th_full_pose parameters"
)
parser.add_argument(
    "--hand_side",
    type=str,
    choices=["right", "left"],
    help="Side of the hand",
    default="right"
)
args = parser.parse_args()

# Get raw dataset dir.
raw_dir = "misc/dex-ycb"

# Load MANO model.
mano = {}
for k, name in zip(("right", "left"), ("RIGHT", "LEFT")):
    mano_file = os.path.join(
        os.path.dirname(__file__), "misc", "mano", "MANO_{}.pkl".format(name)
    )
    with open(mano_file, "rb") as f:
        mano[k] = pickle.load(f, encoding="latin1")

# Load meta.
n = "20200709-subject-01"
if args.hand_side == "right":
    s = "20200709_151032"
else:
    s = "20200709_152624"
name = os.path.join(n, s)
meta_file = os.path.join(raw_dir, name, "meta.yml")
with open(meta_file, "r") as f:
    meta = yaml.load(f, Loader=yaml.FullLoader)
print("The input th_full_pose parameters should be  obtained from the {} hand.".format(meta["mano_sides"]))

# Load extrinsics.
extr_file = os.path.join(
    raw_dir,
    "calibration",
    "extrinsics_{}".format(meta["extrinsics"]),
    "extrinsics.yml",
)
with open(extr_file, "r") as f:
    extr = yaml.load(f, Loader=yaml.FullLoader)
tag_T = np.array(extr["extrinsics"]["apriltag"], dtype=np.float32).reshape(3, 4)
tag_R = tag_T[:, :3]
tag_t = tag_T[:, 3]
tag_R_inv = tag_R.T
tag_t_inv = np.matmul(tag_R_inv, -tag_t)

# Process MANO pose.
mano_betas = []
root_trans = []
comp = []
mean = []
for s, c in zip(meta["mano_sides"], meta["mano_calib"]):
    mano_calib_file = os.path.join(
        raw_dir, "calibration", "mano_{}".format(c), "mano.yml"
    )
    with open(mano_calib_file, "r") as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
    betas = mano_calib["betas"]
    mano_betas.append(betas)
    v = mano[s]["shapedirs"].dot(betas) + mano[s]["v_template"]
    r = mano[s]["J_regressor"][0].dot(v)[0]
    root_trans.append(r)
    comp.append(mano[s]["hands_components"])
    mean.append(mano[s]["hands_mean"])
root_trans = np.array(root_trans, dtype=np.float32)
comp = np.array(comp, dtype=np.float32)
mean = np.array(mean, dtype=np.float32)

pose_m = np.zeros((1, 1, 52))
pose_m[0, 0, :48] = np.load(args.th_full_pose_path)
q = pose_m[:, :, 0:3]
t = pose_m[:, :, 48:51]

def transform(q, t, tag_R_inv, tag_t_inv):
    """Transforms 6D pose to tag coordinates."""
    q_trans = np.zeros((*q.shape[:2], 4), dtype=q.dtype)
    t_trans = np.zeros(t.shape, dtype=t.dtype)

    i = np.any(q != 0, axis=2) | np.any(t != 0, axis=2)
    q = q[i]
    t = t[i]

    if q.shape[1] == 4:
        R = Rot.from_quat(q).as_matrix().astype(np.float32)
    if q.shape[1] == 3:
        R = Rot.from_rotvec(q).as_matrix().astype(np.float32)
    R = np.matmul(tag_R_inv, R)
    t = np.matmul(tag_R_inv, t.T).T + tag_t_inv
    q = Rot.from_matrix(R).as_quat().astype(np.float32)

    q_trans[i] = q
    t_trans[i] = t

    return q_trans, t_trans

i = np.any(pose_m != 0.0, axis=2)
t[i] += root_trans[np.nonzero(i)[1]]
q, t = transform(q, t, tag_R_inv, tag_t_inv)
t[i] -= root_trans[np.nonzero(i)[1]]

p = pose_m[:, :, 3:48]
# Notes: We already done this in manopth
# p = np.einsum("abj,bjk->abk", p, comp) + mean
p[~i] = 0.0

q_i = q[i]
q_i = Rot.from_quat(q_i).as_rotvec().astype(np.float32)
q = np.zeros((*q.shape[:2], 3), dtype=q.dtype)
q[i] = q_i
q = np.dstack((q, p))
for o, (s, b) in enumerate(zip(meta["mano_sides"], mano_betas)):
    model_dir = os.path.join(os.path.dirname(__file__), "misc", "mano")
    model = HandModel45(
        left_hand=s == "left", models_dir=model_dir, betas=b
    )
    origin = model.origins(b)[0]
    sid = np.nonzero(np.any(q[:, o] != 0, axis=1))[0][0]
    eid = np.nonzero(np.any(q[:, o] != 0, axis=1))[0][-1]
    for f in range(sid, eid + 1):
        mano_pose = q[f, o]
        trans = t[f, o]
        angles, basis = model.mano_to_angles(mano_pose)
        trans = trans + origin - basis @ origin
        q[f, o, 3:48] = angles
        t[f, o] = trans
q_i = q[i]
q_i_base = q_i[:, 0:3]
q_i_pose = q_i[:, 3:48].reshape(-1, 3)
q_i_base = Rot.from_rotvec(q_i_base).as_quat().astype(np.float32)
q_i_pose = Rot.from_euler("XYZ", q_i_pose).as_quat().astype(np.float32)
q_i_pose = q_i_pose.reshape(-1, 60)
q_i = np.hstack((q_i_base, q_i_pose))
q = np.zeros((*q.shape[:2], 64), dtype=q.dtype)
q[i] = q_i

i = np.any(q != 0.0, axis=2)
q_i = q[i]
q = np.zeros((*q.shape[:2], 48), dtype=q.dtype)
for o in range(q.shape[1]):
    q_i_o = q_i[np.nonzero(i)[1] == o]
    q_i_o = q_i_o.reshape(-1, 4)
    q_i_o = Rot.from_quat(q_i_o).as_euler("XYZ").astype(np.float32)
    q_i_o = q_i_o.reshape(-1, 48)
    # https://math.stackexchange.com/questions/463748/getting-cumulative-euler-angle-from-a-single-quaternion
    q_i_o[:, 0:3] = np.unwrap(q_i_o[:, 0:3], axis=0)
    q[i[:, o], o] = q_i_o

pose_m = np.dstack((t, q))
print("Parameters for the MANO hand : {}".format(torch.tensor(pose_m)))