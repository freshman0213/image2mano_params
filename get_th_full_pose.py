import argparse
from copy import deepcopy
import os
import pickle
import sys

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from handobjectdatasets.queries import TransQueries, BaseQueries
from handobjectdatasets.viz2d import visualize_joints_2d_cv2

from mano_train.exputils import argutils
from mano_train.netscripts.reload import reload_model
from mano_train.visualize import displaymano
from mano_train.demo.preprocess import prepare_input, preprocess_frame


def forward_pass_3d(model, input_image, hand_side, pred_obj=True):
    sample = {}
    sample[TransQueries.images] = input_image
    sample[BaseQueries.sides] = [hand_side]
    sample[TransQueries.joints3d] = input_image.new_ones((1, 21, 3)).float()
    sample["root"] = "wrist"
    if pred_obj:
        sample[TransQueries.objpoints3d] = input_image.new_ones(
            (1, 600, 3)
        ).float()
    _, results, _ = model.forward(sample, no_loss=True)
    return results


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        help="Path to image",
        default="readme_assets/images/can.jpg",
    )
    parser.add_argument(
        "--hand_side",
        type=str,
        choices=["right", "left"],
        help="Side of the hand",
        default="right"
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip the image"
    )
    parser.add_argument(
        "--no_beta", action="store_true", help="Force shape to average"
    )
    args = parser.parse_args()
    argutils.print_args(args)

    # Initialize network
    checkpoint_path = 'release_models/hands_only/checkpoint.pth.tar'
    checkpoint = os.path.dirname(checkpoint_path)
    with open(os.path.join(checkpoint, "opt.pkl"), "rb") as opt_f:
        opts = pickle.load(opt_f)
    model = reload_model(checkpoint_path, opts, no_beta=args.no_beta)
    model.eval()

    # Preprocess image
    frame = cv2.imread(args.image_path)
    frame = preprocess_frame(frame)
    img = Image.fromarray(frame.copy())
    hand_crop = cv2.resize(np.array(img), (256, 256))
    hand_image = prepare_input(hand_crop, flip_left_right=args.flip)

    # Forward pass
    if args.flip:
        hand_side = "right" if args.hand_side == "left" else "left"
    else:
        hand_side = args.hand_side
    output = forward_pass_3d(model, hand_image, hand_side)
    verts = output["verts"].cpu().detach().numpy()[0]

    # Visualization
    with open(f"misc/mano/MANO_{hand_side.upper()}.pkl", "rb") as p_f:
        mano_data = pickle.load(p_f, encoding="latin1")
        faces = mano_data["f"]

    fig = plt.figure(figsize=(4, 4))
    fig.clf()

    ax = fig.add_subplot(1, 2, 1)
    inpimage = deepcopy(hand_crop)
    if args.flip:
        ax.imshow(np.flip(inpimage[:, :, ::-1], axis=1))
    else:
        ax.imshow(inpimage[:, :, ::-1])

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    displaymano.add_mesh(ax, verts, faces)
    plt.show()
