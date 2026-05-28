#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERP (equirectangular panorama, 2:1) -> Cube Map (6 faces)

Dependencies:
  pip install opencv-python numpy

Example usage:
  python erp2cubemap.py input.jpg --size 1024 --out out_dir
  python erp2cubemap.py input.jpg --size 1024 --layout cross --out cube_cross.png
"""
import os
import math
import argparse
import numpy as np
import cv2


def build_face_map(face_size, face):
    """
    Build the mapping from cube-face pixels to ERP pixels for a given face.
    Returns: map_x, map_y (float32, for cv2.remap)
    Face orientations (right-handed coords: X right, Y up, Z forward):
      +X: right
      -X: left
      +Y: top
      -Y: bottom
      +Z: front
      -Z: back
    Each face has FOV = 90 deg; u,v in [-1, 1] cover the face.
    """
    # Sample at pixel centers
    s = face_size
    jj, ii = np.meshgrid(np.arange(s, dtype=np.float32),
                         np.arange(s, dtype=np.float32))
    # Map pixel coords to [-1, 1] (center at 0) so we sample pixel centers
    a = 2.0 * (jj + 0.5) / s - 1.0
    b = 2.0 * (ii + 0.5) / s - 1.0

    # Direction vectors per face (dx, dy, dz)
    if face == 'right':      # +X
        dx, dy, dz = np.ones_like(a), -b, -a
    elif face == 'left':     # -X
        dx, dy, dz = -np.ones_like(a), -b, a
    elif face == 'top':      # +Y
        dx, dy, dz = a, np.ones_like(a), b
    elif face == 'bottom':   # -Y
        dx, dy, dz = a, -np.ones_like(a), -b
    elif face == 'front':    # +Z
        dx, dy, dz = a, -b, np.ones_like(a)
    elif face == 'back':     # -Z
        dx, dy, dz = -a, -b, -np.ones_like(a)
    else:
        raise ValueError(f'Unknown face: {face}')

    # Normalize the direction
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    dx /= norm; dy /= norm; dz /= norm

    # ERP mapping (lat/lon -> pixels)
    # Longitude theta in (-pi, pi] via atan2(X, Z); latitude phi in [-pi/2, pi/2]
    theta = np.arctan2(dz, dx)          # horizontal angle: +Z -> 0 (correct front)
    phi   = np.arcsin(dy)               # vertical angle: +Y -> +pi/2 (top)

    # Outputs map_x, map_y are ERP image coordinates (col=x, row=y)
    # Assuming source width=W, height=H:
    #   x = (theta + pi) / (2*pi) * W
    #   y = (pi/2 - phi) / pi * H   (phi=+pi/2 -> y=0 top)
    # W,H are placeholders; we apply the actual size when remapping.
    # To support arbitrary input sizes, we emit normalized coords [0,1) and scale by W/H later.
    map_x_norm = (theta + math.pi) / (2.0 * math.pi)
    map_y_norm = (math.pi/2 - phi) / math.pi

    return map_x_norm.astype(np.float32), map_y_norm.astype(np.float32)


def remap_face(erp_img, map_x_norm, map_y_norm, interp, border):
    H, W = erp_img.shape[:2]
    map_x = map_x_norm * (W - 1)
    map_y = map_y_norm * (H - 1)
    return cv2.remap(erp_img, map_x, map_y, interpolation=interp, borderMode=border)


def save_six_faces(faces_dict, out_dir, base):
    os.makedirs(out_dir, exist_ok=True)
    for name, img in faces_dict.items():
        cv2.imwrite(os.path.join(out_dir, f"{base}_{name}.png"), img)


def make_cross_layout(faces, face_size):
    """
    Produce the common 4x3 landscape cross layout:
         [    ][top ][    ][    ]
         [left][front][right][back]
         [    ][bottom][    ][    ]
    Canvas size: (3H, 4W) = (3S, 4S); empty cells filled with black.
    """
    S = face_size
    canvas = np.zeros((3*S, 4*S, 3), dtype=np.uint8)

    # Place the faces
    def put(name, row, col):
        canvas[row*S:(row+1)*S, col*S:(col+1)*S] = faces[name]

    put('top',    0, 1)
    put('left',   1, 0)
    put('front',  1, 1)
    put('right',  1, 2)
    put('back',   1, 3)
    put('bottom', 2, 1)
    return canvas


def parse_args():
    ap = argparse.ArgumentParser(description='ERP -> CubeMap converter')
    ap.add_argument('input', help='input ERP image path (aspect ratio ~ 2:1)')
    ap.add_argument('--size', type=int, default=1024, help='cube-face size in pixels (default 1024)')
    ap.add_argument('--out', default='out', help='output directory (six-face mode) or file (cross mode)')
    ap.add_argument('--layout', choices=['six', 'cross'], default='six',
                    help='output layout: six=six separate faces, cross=cross collage')
    ap.add_argument('--interp', choices=['linear','nearest','cubic','lanczos'], default='lanczos',
                    help='resampling interpolation')
    ap.add_argument('--border', choices=['wrap','reflect','constant'], default='wrap',
                    help='longitude wrap mode (wrap recommended); latitude is clamped or filled')
    return ap.parse_args()


def main():
    args = parse_args()

    interp_map = {
        'nearest': cv2.INTER_NEAREST,
        'linear' : cv2.INTER_LINEAR,
        'cubic'  : cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4,
    }
    border_map = {
        'wrap'    : cv2.BORDER_WRAP,
        'reflect' : cv2.BORDER_REFLECT_101,
        'constant': cv2.BORDER_CONSTANT,
    }

    erp = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if erp is None:
        raise SystemExit(f'failed to read input: {args.input}')
    H, W = erp.shape[:2]
    if abs((W / max(1,H)) - 2.0) > 0.2:
        print(f'WARN: input aspect ratio does not look like 2:1 (got {W}:{H}); confirm the image is an ERP panorama.')

    faces_order = ['right','left','top','bottom','front','back']
    maps = {}
    for name in faces_order:
        maps[name] = build_face_map(args.size, name)

    faces = {}
    for name in faces_order:
        map_x_norm, map_y_norm = maps[name]
        face_img = remap_face(erp, map_x_norm, map_y_norm,
                              interp=interp_map[args.interp],
                              border=border_map[args.border])
        faces[name] = face_img

    if args.layout == 'six':
        base = os.path.splitext(os.path.basename(args.input))[0]
        save_six_faces(faces, args.out, base)
        print(f'wrote six face images to: {args.out}\nface order: {faces_order}')
    else:
        cross = make_cross_layout(faces, args.size)
        ok = cv2.imwrite(args.out, cross)
        if not ok:
            raise SystemExit(f'failed to write output: {args.out}')
        print(f'wrote cross collage: {args.out}\nface order (layout: see code comments): {faces_order}')


if __name__ == '__main__':
    main()
