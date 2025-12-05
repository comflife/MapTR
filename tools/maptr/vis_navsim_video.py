# -*- coding: utf-8 -*-
"""
NavSim GT + Camera 영상 생성 도구
카메라 8개 + GT BEV Map을 합쳐서 영상으로 저장

Usage:
    cd /home/byounggun/MapTR
    python tools/maptr/vis_navsim_video.py \
        projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
        --output ./vis_navsim_video.mp4 \
        --num-samples 100 \
        --fps 5
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import mmcv
import torch
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
import cv2

# NavSim 8개 카메라
NAVSIM_CAMS = ['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0']

# Map class colors (BGR for OpenCV)
COLORS_CV2 = {
    0: (0, 165, 255),   # divider - orange
    1: (255, 0, 0),     # ped_crossing - blue
    2: (0, 255, 0),     # boundary - green
}
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim GT + Camera 영상 생성')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--output', default='./vis_navsim_video.mp4', help='output video path')
    parser.add_argument('--num-samples', type=int, default=100, help='number of samples')
    parser.add_argument('--fps', type=int, default=5, help='video fps')
    parser.add_argument('--start-idx', type=int, default=0, help='start sample index')
    args = parser.parse_args()
    return args


def draw_gt_on_bev(gt_bboxes_3d, gt_labels_3d, pc_range, img_size=800):
    """GT map을 OpenCV 이미지로 그리기"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    def ego_to_img(ego_x, ego_y):
        scale = img_size / (pc_range[3] - pc_range[0])
        img_x = int(((-ego_y) - pc_range[1]) * scale)
        img_y = int((pc_range[3] - ego_x) * scale)
        return img_x, img_y
    
    num_elements = {0: 0, 1: 0, 2: 0}
    
    try:
        if hasattr(gt_bboxes_3d, 'fixed_num_sampled_points'):
            gt_lines = gt_bboxes_3d.fixed_num_sampled_points
        elif hasattr(gt_bboxes_3d, '_fixed_pts'):
            gt_lines = gt_bboxes_3d._fixed_pts
        else:
            return img, num_elements
        
        for pts, label in zip(gt_lines, gt_labels_3d):
            if isinstance(pts, torch.Tensor):
                pts = pts.numpy()
            
            label_idx = int(label)
            if label_idx < 0 or label_idx > 2:
                continue
            
            valid_mask = (pts[:, 0] > -9000) & (pts[:, 1] > -9000)
            pts = pts[valid_mask]
            
            if len(pts) < 2:
                continue
            
            img_pts = np.array([ego_to_img(p[0], p[1]) for p in pts])
            
            color = COLORS_CV2[label_idx]
            for i in range(len(img_pts) - 1):
                pt1 = tuple(img_pts[i])
                pt2 = tuple(img_pts[i + 1])
                cv2.line(img, pt1, pt2, color, 2)
            
            for pt in img_pts:
                cv2.circle(img, tuple(pt), 3, color, -1)
            
            num_elements[label_idx] += 1
    except Exception as e:
        pass
    
    # 차량 그리기
    center_x, center_y = ego_to_img(0, 0)
    car_pts = np.array([
        ego_to_img(2.5, -1),
        ego_to_img(2.5, 1),
        ego_to_img(-2.5, 1),
        ego_to_img(-2.5, -1),
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_pts], (128, 128, 128))
    
    front_pt = ego_to_img(4, 0)
    cv2.arrowedLine(img, (center_x, center_y), front_pt, (0, 0, 255), 3, tipLength=0.3)
    
    # 범례
    y_offset = 30
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_CV2[i]
        cv2.rectangle(img, (10, y_offset + i*25), (30, y_offset + i*25 + 15), color, -1)
        cv2.putText(img, f"{name} ({num_elements[i]})", (40, y_offset + i*25 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.putText(img, "GT Map (BEV)", (img_size//2 - 60, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img, num_elements


def create_camera_grid(data, target_h=270, target_w=480):
    """8개 카메라를 3x3 그리드로 배치"""
    img_metas = data['img_metas']
    if hasattr(img_metas, 'data'):
        img_metas = img_metas.data
    
    if isinstance(img_metas, dict):
        img_metas = img_metas.get(0, img_metas)
    elif isinstance(img_metas, list) and len(img_metas) > 0:
        img_metas = img_metas[0]
        if isinstance(img_metas, list) and len(img_metas) > 0:
            img_metas = img_metas[0]
    
    cam_images = {}
    if isinstance(img_metas, dict) and 'filename' in img_metas:
        filenames = img_metas['filename']
        for i, cam_name in enumerate(NAVSIM_CAMS):
            if i < len(filenames):
                img_path = filenames[i]
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        cam_images[cam_name] = img
    
    if not cam_images:
        return None
    
    def resize_cam(img, cam_name):
        img = cv2.resize(img, (target_w, target_h))
        cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        return img
    
    row1_cams = ['CAM_L0', 'CAM_F0', 'CAM_R0']
    row2_cams = ['CAM_L1', 'CAM_B0', 'CAM_R1']
    row3_cams = ['CAM_L2', None, 'CAM_R2']
    
    def make_row(cam_list, frame_idx=None):
        row_imgs = []
        for cam in cam_list:
            if cam is None:
                info_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 50
                cv2.putText(info_img, "NavSim", (target_w//2-50, target_h//2-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(info_img, "MapTR", (target_w//2-40, target_h//2+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                if frame_idx is not None:
                    cv2.putText(info_img, f"Frame: {frame_idx}", (target_w//2-50, target_h//2+40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                row_imgs.append(info_img)
            elif cam in cam_images:
                row_imgs.append(resize_cam(cam_images[cam], cam))
            else:
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                cv2.putText(blank, cam + " (missing)", (10, target_h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                row_imgs.append(blank)
        return cv2.hconcat(row_imgs) if row_imgs else None
    
    return make_row, row1_cams, row2_cams, row3_cams, cam_images


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # Import plugins
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            plugin_dir = cfg.plugin_dir
            _module_dir = os.path.dirname(plugin_dir)
            _module_dir = _module_dir.split('/')
            _module_path = _module_dir[0]
            for m in _module_dir[1:]:
                _module_path = _module_path + '.' + m
            plg_lib = importlib.import_module(_module_path)
    
    # Build dataset
    print("Building dataset...")
    dataset = build_dataset(cfg.data.train)
    
    pc_range = cfg.point_cloud_range
    
    # Video settings
    target_h, target_w = 270, 480
    cam_grid_h = target_h * 3
    cam_grid_w = target_w * 3
    gt_size = cam_grid_h  # 정사각형
    
    total_w = cam_grid_w + gt_size
    total_h = cam_grid_h
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (total_w, total_h))
    
    print(f"\n{'='*60}")
    print(f"Generating video with {args.num_samples} frames")
    print(f"Output: {args.output}")
    print(f"Resolution: {total_w}x{total_h}, FPS: {args.fps}")
    print(f"{'='*60}\n")
    
    prog_bar = mmcv.ProgressBar(args.num_samples)
    frames_written = 0
    
    for idx in range(args.start_idx, min(args.start_idx + args.num_samples, len(dataset))):
        try:
            data = dataset[idx]
            
            # GT 추출
            gt_labels = data.get('gt_labels_3d')
            gt_bboxes = data.get('gt_bboxes_3d')
            
            if gt_labels is None or gt_bboxes is None:
                prog_bar.update()
                continue
            
            if hasattr(gt_labels, 'data'):
                gt_labels = gt_labels.data
            if hasattr(gt_bboxes, 'data'):
                gt_bboxes = gt_bboxes.data
            
            # GT BEV 그리기
            gt_img, num_elements = draw_gt_on_bev(gt_bboxes, gt_labels, pc_range, img_size=gt_size)
            
            # 카메라 그리드 생성
            img_metas = data['img_metas']
            if hasattr(img_metas, 'data'):
                img_metas = img_metas.data
            if isinstance(img_metas, dict):
                img_metas = img_metas.get(0, img_metas)
            elif isinstance(img_metas, list) and len(img_metas) > 0:
                img_metas = img_metas[0]
                if isinstance(img_metas, list) and len(img_metas) > 0:
                    img_metas = img_metas[0]
            
            cam_images = {}
            if isinstance(img_metas, dict) and 'filename' in img_metas:
                filenames = img_metas['filename']
                for i, cam_name in enumerate(NAVSIM_CAMS):
                    if i < len(filenames):
                        img_path = filenames[i]
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                cam_images[cam_name] = img
            
            if not cam_images:
                prog_bar.update()
                continue
            
            def resize_cam(img, cam_name):
                img = cv2.resize(img, (target_w, target_h))
                cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, cam_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                return img
            
            # 카메라 그리드 구성
            row1_cams = ['CAM_L0', 'CAM_F0', 'CAM_R0']
            row2_cams = ['CAM_L1', 'CAM_B0', 'CAM_R1']
            row3_cams = ['CAM_L2', None, 'CAM_R2']
            
            def make_row(cam_list):
                row_imgs = []
                for cam in cam_list:
                    if cam is None:
                        info_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 50
                        cv2.putText(info_img, "NavSim + MapTR", (target_w//2-80, target_h//2-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(info_img, f"Frame: {idx}", (target_w//2-40, target_h//2+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        row_imgs.append(info_img)
                    elif cam in cam_images:
                        row_imgs.append(resize_cam(cam_images[cam], cam))
                    else:
                        blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                        row_imgs.append(blank)
                return cv2.hconcat(row_imgs)
            
            row1 = make_row(row1_cams)
            row2 = make_row(row2_cams)
            row3 = make_row(row3_cams)
            
            cam_grid = cv2.vconcat([row1, row2, row3])
            
            # 합치기
            combined = cv2.hconcat([cam_grid, gt_img])
            
            # 프레임 번호 추가
            cv2.putText(combined, f"Sample: {idx}", (10, total_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, f"Sample: {idx}", (10, total_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            
            out.write(combined)
            frames_written += 1
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
        
        prog_bar.update()
    
    out.release()
    
    print(f"\n\n{'='*60}")
    print(f"Done! Video saved to: {args.output}")
    print(f"Total frames: {frames_written}")
    print(f"Duration: {frames_written / args.fps:.1f} seconds")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
