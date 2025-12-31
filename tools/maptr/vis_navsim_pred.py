# -*- coding: utf-8 -*-
"""
NavSim Prediction 시각화 도구
학습된 MapTR 모델로 inference해서 GT와 예측 결과를 함께 시각화

Usage:
    cd /home/byounggun/MapTR
    conda activate navsim
    python tools/maptr/vis_navsim_pred.py \
        projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
        work_dirs/maptr_tiny_r50_navsim_24e/epoch_8.pth \
        --show-dir ./vis_navsim_pred \
        --num-samples 5
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
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
import cv2

# NavSim 8개 카메라
NAVSIM_CAMS = ['CAM_F0', 'CAM_L0', 'CAM_L1', 'CAM_L2', 'CAM_R0', 'CAM_R1', 'CAM_R2', 'CAM_B0']

# Map class colors (BGR for OpenCV)
COLORS_GT = {
    0: (0, 165, 255),   # divider - orange
    1: (255, 0, 0),     # ped_crossing - blue
    2: (0, 255, 0),     # boundary - green
}
COLORS_PRED = {
    0: (0, 100, 200),   # divider - dark orange
    1: (200, 0, 0),     # ped_crossing - dark blue
    2: (0, 200, 0),     # boundary - dark green
}
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim Prediction 시각화')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--show-dir', default='./vis_navsim_pred', help='output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold for predictions')
    parser.add_argument('--start-idx', type=int, default=0, help='start sample index')
    args = parser.parse_args()
    return args


def draw_pts_on_img(img, pts_list, labels, colors, img_size, pc_range, line_width=2, point_size=3):
    """점들을 이미지에 그리기"""
    def ego_to_img(ego_x, ego_y):
        scale = img_size / (pc_range[3] - pc_range[0])
        img_x = int(((-ego_y) - pc_range[1]) * scale)
        img_y = int((pc_range[3] - ego_x) * scale)
        return img_x, img_y
    
    num_elements = {0: 0, 1: 0, 2: 0}
    
    for pts, label in zip(pts_list, labels):
        if isinstance(pts, torch.Tensor):
            pts = pts.cpu().numpy()
        
        label_idx = int(label)
        if label_idx < 0 or label_idx > 2:
            continue
        
        # 유효한 점만 필터링
        if len(pts.shape) == 1:
            continue
        valid_mask = (pts[:, 0] > -9000) & (pts[:, 1] > -9000)
        pts = pts[valid_mask]
        
        if len(pts) < 2:
            continue
        
        # 이미지 좌표로 변환
        img_pts = np.array([ego_to_img(p[0], p[1]) for p in pts])
        
        # 선 그리기
        color = colors[label_idx]
        for i in range(len(img_pts) - 1):
            pt1 = tuple(img_pts[i])
            pt2 = tuple(img_pts[i + 1])
            cv2.line(img, pt1, pt2, color, line_width)
        
        # 점 그리기
        for pt in img_pts:
            cv2.circle(img, tuple(pt), point_size, color, -1)
        
        num_elements[label_idx] += 1
    
    return num_elements


def draw_gt_on_bev(gt_bboxes_3d, gt_labels_3d, pc_range, img_size=800):
    """GT map을 OpenCV 이미지로 그리기"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    num_elements = {0: 0, 1: 0, 2: 0}
    
    try:
        if hasattr(gt_bboxes_3d, 'fixed_num_sampled_points'):
            gt_lines = gt_bboxes_3d.fixed_num_sampled_points
        elif hasattr(gt_bboxes_3d, '_fixed_pts'):
            gt_lines = gt_bboxes_3d._fixed_pts
        else:
            return img, num_elements
        
        num_elements = draw_pts_on_img(img, gt_lines, gt_labels_3d, COLORS_GT, img_size, pc_range)
    except Exception as e:
        print(f"Error drawing GT: {e}")
    
    return img, num_elements


def draw_pred_on_bev(pred_pts, pred_labels, pred_scores, pc_range, img_size=800, score_thr=0.3):
    """Prediction을 OpenCV 이미지로 그리기"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    num_elements = {0: 0, 1: 0, 2: 0}
    
    if pred_pts is None or len(pred_pts) == 0:
        return img, num_elements
    
    # Score threshold 적용
    if pred_scores is not None:
        mask = pred_scores >= score_thr
        pred_pts = pred_pts[mask]
        pred_labels = pred_labels[mask]
        pred_scores = pred_scores[mask]
    
    if len(pred_pts) == 0:
        return img, num_elements
    
    num_elements = draw_pts_on_img(img, pred_pts, pred_labels, COLORS_PRED, img_size, pc_range, line_width=3, point_size=4)
    
    return img, num_elements


def draw_combined_bev(gt_bboxes_3d, gt_labels_3d, pred_pts, pred_labels, pred_scores, pc_range, img_size=800, score_thr=0.3):
    """GT와 Prediction을 함께 그리기"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    gt_num = {0: 0, 1: 0, 2: 0}
    pred_num = {0: 0, 1: 0, 2: 0}
    
    # GT 그리기 (얇은 선)
    try:
        if hasattr(gt_bboxes_3d, 'fixed_num_sampled_points'):
            gt_lines = gt_bboxes_3d.fixed_num_sampled_points
        elif hasattr(gt_bboxes_3d, '_fixed_pts'):
            gt_lines = gt_bboxes_3d._fixed_pts
        else:
            gt_lines = []
        
        if len(gt_lines) > 0:
            gt_num = draw_pts_on_img(img, gt_lines, gt_labels_3d, COLORS_GT, img_size, pc_range, line_width=2, point_size=2)
    except Exception as e:
        pass
    
    # Prediction 그리기 (굵은 선)
    if pred_pts is not None and len(pred_pts) > 0:
        if pred_scores is not None:
            mask = pred_scores >= score_thr
            pred_pts = pred_pts[mask]
            pred_labels = pred_labels[mask]
        
        if len(pred_pts) > 0:
            pred_num = draw_pts_on_img(img, pred_pts, pred_labels, COLORS_PRED, img_size, pc_range, line_width=3, point_size=4)
    
    return img, gt_num, pred_num


def draw_car_and_legend(img, img_size, pc_range, gt_num, pred_num):
    """차량과 범례 추가"""
    def ego_to_img(ego_x, ego_y):
        scale = img_size / (pc_range[3] - pc_range[0])
        img_x = int(((-ego_y) - pc_range[1]) * scale)
        img_y = int((pc_range[3] - ego_x) * scale)
        return img_x, img_y
    
    # 차량 그리기
    center = ego_to_img(0, 0)
    car_pts = np.array([
        ego_to_img(2.5, -1),
        ego_to_img(2.5, 1),
        ego_to_img(-2.5, 1),
        ego_to_img(-2.5, -1),
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_pts], (128, 128, 128))
    front = ego_to_img(4, 0)
    cv2.arrowedLine(img, center, front, (0, 0, 255), 3)
    
    # 범례 - GT
    y_offset = 20
    cv2.putText(img, "GT (thin):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset += 20
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_GT[i]
        cv2.rectangle(img, (10, y_offset + i*20), (25, y_offset + i*20 + 12), color, -1)
        cv2.putText(img, f"{name}: {gt_num.get(i, 0)}", (30, y_offset + i*20 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # 범례 - Pred
    y_offset += 70
    cv2.putText(img, "Pred (bold):", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    y_offset += 20
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_PRED[i]
        cv2.rectangle(img, (10, y_offset + i*20), (25, y_offset + i*20 + 12), color, -1)
        cv2.putText(img, f"{name}: {pred_num.get(i, 0)}", (30, y_offset + i*20 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return img


def create_camera_grid(img_metas, target_h=270, target_w=480):
    """8개 카메라를 3x3 그리드로 배치"""
    if hasattr(img_metas, 'data'):
        img_metas = img_metas.data
    if isinstance(img_metas, list):
        img_metas = img_metas[0]
    if isinstance(img_metas, list):
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
    
    def make_row(cam_list, info_text=""):
        row_imgs = []
        for cam in cam_list:
            if cam is None:
                info_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 50
                cv2.putText(info_img, "NavSim + MapTR", (target_w//2-80, target_h//2-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_img, info_text, (target_w//2-50, target_h//2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                row_imgs.append(info_img)
            elif cam in cam_images:
                row_imgs.append(resize_cam(cam_images[cam], cam))
            else:
                blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                row_imgs.append(blank)
        return cv2.hconcat(row_imgs)
    
    row1 = make_row(row1_cams)
    row2 = make_row(row2_cams)
    row3 = make_row(row3_cams, "Prediction")
    
    return cv2.vconcat([row1, row2, row3])


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
    
    # Build model
    print("Building model...")
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # Build dataset - use val/test config for proper data loading
    print("Building dataset...")
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    
    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )
    
    os.makedirs(args.show_dir, exist_ok=True)
    
    pc_range = cfg.point_cloud_range
    img_size = 810
    
    print(f"\n{'='*60}")
    print(f"Generating predictions for {args.num_samples} samples")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score threshold: {args.score_thr}")
    print(f"Output: {args.show_dir}")
    print(f"{'='*60}\n")
    
    prog_bar = mmcv.ProgressBar(args.num_samples)
    
    data_iter = iter(data_loader)
    for idx in range(args.start_idx + args.num_samples):
        try:
            data = next(data_iter)
            
            if idx < args.start_idx:
                continue
            
            # Get img_metas for camera visualization
            img_metas = data['img_metas'].data[0][0]
            
            # Get GT (may not exist in test mode)
            gt_labels = None
            gt_bboxes = None
            if 'gt_labels_3d' in data:
                gt_labels = data['gt_labels_3d'].data[0][0] if hasattr(data['gt_labels_3d'], 'data') else data['gt_labels_3d']
            if 'gt_bboxes_3d' in data:
                gt_bboxes = data['gt_bboxes_3d'].data[0][0] if hasattr(data['gt_bboxes_3d'], 'data') else data['gt_bboxes_3d']
            
            # Run inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Parse predictions
            pred_pts = None
            pred_labels = None
            pred_scores = None
            
            if result and len(result) > 0:
                res = result[0]
                if 'pts_bbox' in res:
                    pts_bbox = res['pts_bbox']
                    if 'pts' in pts_bbox:
                        pred_pts = pts_bbox['pts'].cpu().numpy()
                    if 'labels' in pts_bbox:
                        pred_labels = pts_bbox['labels'].cpu().numpy()
                    if 'scores' in pts_bbox:
                        pred_scores = pts_bbox['scores'].cpu().numpy()
            
            # Create visualizations
            sample_dir = os.path.join(args.show_dir, f'sample_{idx:05d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 1. GT only
            gt_img, gt_num = draw_gt_on_bev(gt_bboxes, gt_labels, pc_range, img_size)
            gt_img = draw_car_and_legend(gt_img, img_size, pc_range, gt_num, {0:0, 1:0, 2:0})
            cv2.putText(gt_img, "Ground Truth", (img_size//2 - 70, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(sample_dir, 'gt_bev.jpg'), gt_img)
            
            # 2. Prediction only
            pred_img, pred_num = draw_pred_on_bev(pred_pts, pred_labels, pred_scores, pc_range, img_size, args.score_thr)
            pred_img = draw_car_and_legend(pred_img, img_size, pc_range, {0:0, 1:0, 2:0}, pred_num)
            cv2.putText(pred_img, f"Prediction (thr={args.score_thr})", (img_size//2 - 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(sample_dir, 'pred_bev.jpg'), pred_img)
            
            # 3. Combined (GT + Pred)
            combined_img, gt_num, pred_num = draw_combined_bev(
                gt_bboxes, gt_labels, pred_pts, pred_labels, pred_scores, 
                pc_range, img_size, args.score_thr
            )
            combined_img = draw_car_and_legend(combined_img, img_size, pc_range, gt_num, pred_num)
            cv2.putText(combined_img, "GT (thin) + Pred (bold)", (img_size//2 - 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.imwrite(os.path.join(sample_dir, 'combined_bev.jpg'), combined_img)
            
            # 4. With cameras
            cam_grid = create_camera_grid(data['img_metas'])
            if cam_grid is not None:
                cam_h, cam_w = cam_grid.shape[:2]
                combined_resized = cv2.resize(combined_img, (cam_h, cam_h))
                full_combined = cv2.hconcat([cam_grid, combined_resized])
                cv2.imwrite(os.path.join(sample_dir, 'full_combined.jpg'), full_combined,
                           [cv2.IMWRITE_JPEG_QUALITY, 90])
            
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
        
        prog_bar.update()
    
    print(f"\n\n{'='*60}")
    print(f"Done! Results saved to: {args.show_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
