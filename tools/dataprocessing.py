# -*- coding: utf-8 -*-
import argparse
import os

import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm


def get_img_file(file_name):
    imagelist = []
    for parent, _, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')
            ):
                imagelist.append(os.path.join(parent, filename))
        return imagelist
    return imagelist


def rgb2y(img):
    return img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    y = np.zeros([endc, win * win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            y[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return y.reshape([endc, win, win, total_pat_num])


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10, upper_percentile=90):
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


def build_h5(ir_dir, vi_dir, out_dir, data_name, img_size, stride):
    ir_files = sorted(get_img_file(ir_dir))
    vis_files = sorted(get_img_file(vi_dir))
    assert len(ir_files) == len(vis_files)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{data_name}_imgsize_{img_size}_stride_{stride}.h5')
    h5f = h5py.File(out_path, 'w')
    h5_ir = h5f.create_group('ir_patchs')
    h5_vis = h5f.create_group('vis_patchs')

    train_num = 0
    for i in tqdm(range(len(ir_files))):
        i_vis = imread(vis_files[i]).astype(np.float32).transpose(2, 0, 1) / 255.0
        i_vis = rgb2y(i_vis)
        i_ir = imread(ir_files[i]).astype(np.float32)[None, :, :] / 255.0

        i_ir_patch_group = Im2Patch(i_ir, img_size, stride)
        i_vis_patch_group = Im2Patch(i_vis, img_size, stride)

        for ii in range(i_ir_patch_group.shape[-1]):
            bad_ir = is_low_contrast(i_ir_patch_group[0, :, :, ii])
            bad_vis = is_low_contrast(i_vis_patch_group[0, :, :, ii])
            if not (bad_ir or bad_vis):
                avl_ir = i_ir_patch_group[0, :, :, ii][None, ...]
                avl_vis = i_vis_patch_group[0, :, :, ii][None, ...]
                h5_ir.create_dataset(str(train_num), data=avl_ir, dtype=avl_ir.dtype, shape=avl_ir.shape)
                h5_vis.create_dataset(str(train_num), data=avl_vis, dtype=avl_vis.dtype, shape=avl_vis.shape)
                train_num += 1

    h5f.close()
    with h5py.File(out_path, 'r') as f:
        for key in f.keys():
            print(f[key], key, f[key].name)
    print(f'Saved {train_num} patch pairs to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Build Text-FSFuse H5 training patches.')
    parser.add_argument('--ir_dir', default=r'E:\yizuo_SCI\2_Datasets\MSRS-main\train\ir')
    parser.add_argument('--vi_dir', default=r'E:\yizuo_SCI\2_Datasets\MSRS-main\train\vi')
    parser.add_argument('--out_dir', default='./data')
    parser.add_argument('--data_name', default='MSRS_train')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    build_h5(args.ir_dir, args.vi_dir, args.out_dir, args.data_name, args.img_size, args.stride)


if __name__ == '__main__':
    main()
