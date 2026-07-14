# -*- coding: utf-8 -*-
import argparse
import os

import h5py
import numpy as np
from skimage.io import imread
from tqdm import tqdm


IMAGE_SUFFIXES = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', '.npy')


def get_img_file(file_name):
    imagelist = []
    for parent, _, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_SUFFIXES):
                imagelist.append(os.path.join(parent, filename))
    return imagelist


def _pair_image_files(ir_dir, vi_dir):
    def index_by_stem(directory):
        result = {}
        for path in get_img_file(directory):
            stem = os.path.splitext(os.path.basename(path))[0].casefold()
            if stem in result:
                raise ValueError(f'Duplicate image stem "{stem}" in {directory}.')
            result[stem] = path
        return result

    ir_by_stem, vis_by_stem = index_by_stem(ir_dir), index_by_stem(vi_dir)
    if set(ir_by_stem) != set(vis_by_stem):
        missing_ir = sorted(set(vis_by_stem) - set(ir_by_stem))
        missing_vis = sorted(set(ir_by_stem) - set(vis_by_stem))
        raise ValueError(
            'IR and visible file names do not match. '
            f'Missing IR: {missing_ir[:5]}; missing visible: {missing_vis[:5]}.'
        )
    return [(ir_by_stem[stem], vis_by_stem[stem]) for stem in sorted(ir_by_stem)]


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


def build_h5(ir_dir, vi_dir, out_dir, data_name, img_size, stride, output_suffix=''):
    image_pairs = _pair_image_files(ir_dir, vi_dir)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{data_name}_imgsize_{img_size}_stride_{stride}{output_suffix}.h5')
    h5f = h5py.File(out_path, 'w')
    h5_ir = h5f.create_group('ir_patchs')
    h5_vis = h5f.create_group('vis_patchs')
    h5_vis_rgb = h5f.create_group('vis_rgb_patchs')

    train_num = 0
    for ir_path, vis_path in tqdm(image_pairs):
        i_vis = imread(vis_path).astype(np.float32).transpose(2, 0, 1) / 255.0
        i_vis_rgb = i_vis.astype(np.float32, copy=True)
        i_vis_y = rgb2y(i_vis_rgb)
        i_ir = imread(ir_path).astype(np.float32)[None, :, :] / 255.0
        if i_ir.shape[1:] != i_vis_rgb.shape[1:]:
            raise ValueError(
                f'Paired image size mismatch: {os.path.basename(ir_path)} has {i_ir.shape[1:]}, '
                f'{os.path.basename(vis_path)} has {i_vis_rgb.shape[1:]}.'
            )

        i_ir_patch_group = Im2Patch(i_ir, img_size, stride)
        i_vis_patch_group = Im2Patch(i_vis_y, img_size, stride)
        i_vis_rgb_patch_group = Im2Patch(i_vis_rgb, img_size, stride)

        for ii in range(i_ir_patch_group.shape[-1]):
            bad_ir = is_low_contrast(i_ir_patch_group[0, :, :, ii])
            bad_vis = is_low_contrast(i_vis_patch_group[0, :, :, ii])
            if not (bad_ir or bad_vis):
                avl_ir = i_ir_patch_group[0, :, :, ii][None, ...]
                avl_vis = i_vis_patch_group[0, :, :, ii][None, ...]
                avl_vis_rgb = i_vis_rgb_patch_group[:, :, :, ii].astype(np.float32, copy=False)
                h5_ir.create_dataset(str(train_num), data=avl_ir, dtype=avl_ir.dtype, shape=avl_ir.shape)
                h5_vis.create_dataset(str(train_num), data=avl_vis, dtype=avl_vis.dtype, shape=avl_vis.shape)
                h5_vis_rgb.create_dataset(
                    str(train_num),
                    data=avl_vis_rgb,
                    dtype=avl_vis_rgb.dtype,
                    shape=avl_vis_rgb.shape,
                )
                train_num += 1

    h5f.close()
    with h5py.File(out_path, 'r') as f:
        for key in f.keys():
            print(f[key], key, f[key].name)
    print(f'Saved {train_num} patch pairs to {out_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Build Text-FSFuse H5 training patches.')
    parser.add_argument('--ir_dir', default=r'F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS-main\train\ir')
    parser.add_argument('--vi_dir', default=r'F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS\MSRS-main\train\vi')
    parser.add_argument('--out_dir', default=r'F:\1_paper_pan\2_Image_Fusion\2_Datasets\1_MSRS')
    parser.add_argument('--data_name', default='MSRS_train')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--stride', type=int, default=200)
    parser.add_argument('--output_suffix', default='')
    return parser.parse_args()


def main():
    args = parse_args()
    build_h5(args.ir_dir, args.vi_dir, args.out_dir, args.data_name, args.img_size, args.stride, args.output_suffix)


if __name__ == '__main__':
    main()
