# -*- coding: utf-8 -*-
"""
NavSim MapTR 테스트 및 시각화 도구
학습된 모델로 inference해서 GT와 예측 결과를 시각화
학습 코드에 영향 없이 독립적으로 동작

Usage:
    cd /home/byounggun/MapTR
    conda activate navsim
    python tools/maptr/test_navsim.py \
    projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
    work_dirs/maptr_tiny_r50_navsim_24e/epoch_24.pth \
    --show-dir ./vis_navsim_pred \
    --num-samples 5 \
    --score-thr 0.3
"""

import argparse
import os
import sys
import copy

# 프로젝트 루트 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import mmcv
import torch
import numpy as np
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
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
    0: (0, 80, 180),    # divider - dark orange
    1: (180, 0, 0),     # ped_crossing - dark blue
    2: (0, 180, 0),     # boundary - dark green
}
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim MapTR 테스트 및 시각화')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')
    parser.add_argument('--show-dir', default='./vis_navsim_pred', help='output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='number of samples to visualize')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold for predictions')
    parser.add_argument('--start-idx', type=int, default=0, help='start sample index')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    args = parser.parse_args()
    return args


def ego_to_img(ego_x, ego_y, img_size, pc_range):
    """ego 좌표를 이미지 좌표로 변환"""
    scale = img_size / (pc_range[3] - pc_range[0])
    img_x = int(((-ego_y) - pc_range[1]) * scale)
    img_y = int((pc_range[3] - ego_x) * scale)
    return img_x, img_y


def draw_pts_on_img(img, pts_list, labels, colors, img_size, pc_range, line_width=2, point_size=3):
    """점들을 이미지에 그리기"""
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
        img_pts = np.array([ego_to_img(p[0], p[1], img_size, pc_range) for p in pts])
        
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


def draw_car_and_legend(img, img_size, pc_range, gt_num, pred_num, title=""):
    """차량과 범례 추가"""
    # 차량 그리기
    center = ego_to_img(0, 0, img_size, pc_range)
    car_pts = np.array([
        ego_to_img(2.5, -1, img_size, pc_range),
        ego_to_img(2.5, 1, img_size, pc_range),
        ego_to_img(-2.5, 1, img_size, pc_range),
        ego_to_img(-2.5, -1, img_size, pc_range),
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_pts], (128, 128, 128))
    front = ego_to_img(4, 0, img_size, pc_range)
    cv2.arrowedLine(img, center, front, (0, 0, 255), 3)
    
    # 타이틀
    if title:
        cv2.putText(img, title, (img_size//2 - 80, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # 범례 - GT
    y_offset = 45
    cv2.putText(img, "GT:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 18
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_GT[i]
        cv2.rectangle(img, (10, y_offset + i*18), (22, y_offset + i*18 + 10), color, -1)
        cv2.putText(img, f"{name}: {gt_num.get(i, 0)}", (26, y_offset + i*18 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    # 범례 - Pred
    y_offset += 60
    cv2.putText(img, "Pred:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    y_offset += 18
    for i, name in enumerate(CLASS_NAMES):
        color = COLORS_PRED[i]
        cv2.rectangle(img, (10, y_offset + i*18), (22, y_offset + i*18 + 10), color, -1)
        cv2.putText(img, f"{name}: {pred_num.get(i, 0)}", (26, y_offset + i*18 + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    
    return img


def create_camera_grid(img_metas, target_h=270, target_w=480, info_text=""):
    """8개 카메라를 3x3 그리드로 배치"""
    cam_images = {}
    
    if 'filename' in img_metas:
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
    
    def make_row(cam_list):
        row_imgs = []
        for cam in cam_list:
            if cam is None:
                info_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 50
                cv2.putText(info_img, "NavSim + MapTR", (target_w//2-80, target_h//2-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if info_text:
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
    row3 = make_row(row3_cams)
    
    return cv2.vconcat([row1, row2, row3])


def main():
    args = parse_args()
    
    # Config 로드
    cfg = Config.fromfile(args.config)
    
    # Plugin 로드
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
    
    # 모델 빌드
    print("Building model...")
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # 체크포인트 로드
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # 데이터셋 빌드 (train 모드로 GT 포함)
    print("Building dataset...")
    # train dataset을 사용해서 GT를 가져옴 (test dataset은 GT가 없음)
    cfg.data.train.test_mode = False
    dataset = build_dataset(cfg.data.train)
    
    # 데이터로더 빌드
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')),
    )
    
    os.makedirs(args.show_dir, exist_ok=True)
    
    pc_range = cfg.point_cloud_range
    img_size = 810
    
    print(f"\n{'='*60}")
    print(f"NavSim MapTR Test & Visualization")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score threshold: {args.score_thr}")
    print(f"Num samples: {args.num_samples}")
    print(f"Output: {args.show_dir}")
    print(f"{'='*60}\n")
    
    prog_bar = mmcv.ProgressBar(min(args.num_samples, len(dataset)))
    
    for idx, data in enumerate(data_loader):
        if idx < args.start_idx:
            continue
        if idx >= args.start_idx + args.num_samples:
            break
        
        try:
            # img_metas 추출 (DataContainer 처리)
            # union2one에서 metas_map 형태로 저장됨: {0: actual_meta_dict}
            img_metas_raw = data['img_metas']
            if hasattr(img_metas_raw, 'data'):
                metas_dict = img_metas_raw.data[0][0]  # {0: {...}}
                # 실제 메타데이터는 metas_dict[0]에 있음
                img_metas = metas_dict[0] if isinstance(metas_dict, dict) and 0 in metas_dict else metas_dict
            elif isinstance(img_metas_raw, list):
                metas_dict = img_metas_raw[0][0] if isinstance(img_metas_raw[0], list) else img_metas_raw[0]
                img_metas = metas_dict[0] if isinstance(metas_dict, dict) and 0 in metas_dict else metas_dict
            else:
                img_metas = img_metas_raw
            
            # GT 추출 (있는 경우)
            gt_labels = None
            gt_pts = None
            gt_bboxes = None
            
            if 'gt_labels_3d' in data:
                gt_labels_raw = data['gt_labels_3d']
                if hasattr(gt_labels_raw, 'data'):
                    gt_labels = gt_labels_raw.data[0][0]
                elif isinstance(gt_labels_raw, list):
                    gt_labels = gt_labels_raw[0][0] if isinstance(gt_labels_raw[0], list) else gt_labels_raw[0]
                else:
                    gt_labels = gt_labels_raw
            
            if 'gt_bboxes_3d' in data:
                gt_bboxes_raw = data['gt_bboxes_3d']
                if hasattr(gt_bboxes_raw, 'data'):
                    gt_bboxes = gt_bboxes_raw.data[0][0]
                elif isinstance(gt_bboxes_raw, list):
                    gt_bboxes = gt_bboxes_raw[0][0] if isinstance(gt_bboxes_raw[0], list) else gt_bboxes_raw[0]
                else:
                    gt_bboxes = gt_bboxes_raw
                
            if gt_bboxes is not None:
                if hasattr(gt_bboxes, 'fixed_num_sampled_points'):
                    gt_pts = gt_bboxes.fixed_num_sampled_points
                elif hasattr(gt_bboxes, '_fixed_pts'):
                    gt_pts = gt_bboxes._fixed_pts
            
            # Convert GT to numpy
            if gt_pts is not None:
                if isinstance(gt_pts, torch.Tensor):
                    gt_pts = gt_pts.cpu().numpy()
            if gt_labels is not None:
                if isinstance(gt_labels, torch.Tensor):
                    gt_labels = gt_labels.cpu().numpy()
            
            # Fix img_metas format for model inference
            # union2one creates {0: {...}, 1: {...}} but model expects [{...}, {...}]
            if 'img_metas' in data and hasattr(data['img_metas'], 'data'):
                metas_data = data['img_metas'].data
                # metas_data is [[{0: {...}}]]
                for batch_idx in range(len(metas_data)):
                    for sample_idx in range(len(metas_data[batch_idx])):
                        metas_dict = metas_data[batch_idx][sample_idx]
                        if isinstance(metas_dict, dict):
                            # Convert {0: {...}, 1: {...}} to [{...}, {...}]
                            metas_list = [metas_dict[k] for k in sorted(metas_dict.keys())]
                            metas_data[batch_idx][sample_idx] = metas_list
            
            # Inference
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            
            # Prediction 파싱
            pred_pts = None
            pred_labels = None
            pred_scores = None
            
            if result and len(result) > 0:
                res = result[0]
                if 'pts_bbox' in res:
                    pts_bbox = res['pts_bbox']
                    # Keys can be 'pts'/'labels'/'scores' or 'pts_3d'/'labels_3d'/'scores_3d'
                    if 'pts_3d' in pts_bbox:
                        pred_pts = pts_bbox['pts_3d'].cpu().numpy()
                    elif 'pts' in pts_bbox:
                        pred_pts = pts_bbox['pts'].cpu().numpy()
                    
                    if 'labels_3d' in pts_bbox:
                        pred_labels = pts_bbox['labels_3d'].cpu().numpy()
                    elif 'labels' in pts_bbox:
                        pred_labels = pts_bbox['labels'].cpu().numpy()
                    
                    if 'scores_3d' in pts_bbox:
                        pred_scores = pts_bbox['scores_3d'].cpu().numpy()
                    elif 'scores' in pts_bbox:
                        pred_scores = pts_bbox['scores'].cpu().numpy()
            
            # Score threshold 적용
            if pred_pts is not None and pred_scores is not None:
                mask = pred_scores >= args.score_thr
                pred_pts = pred_pts[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
            
            # 시각화
            sample_dir = os.path.join(args.show_dir, f'sample_{idx:05d}')
            os.makedirs(sample_dir, exist_ok=True)
            
            # 1. GT BEV
            gt_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            gt_num = {0: 0, 1: 0, 2: 0}
            if gt_pts is not None and gt_labels is not None:
                gt_num = draw_pts_on_img(gt_img, gt_pts, gt_labels, COLORS_GT, img_size, pc_range, line_width=2)
            gt_img = draw_car_and_legend(gt_img, img_size, pc_range, gt_num, {0:0, 1:0, 2:0}, "Ground Truth")
            cv2.imwrite(os.path.join(sample_dir, 'gt_bev.jpg'), gt_img)
            
            # 2. Pred BEV
            pred_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            pred_num = {0: 0, 1: 0, 2: 0}
            if pred_pts is not None and pred_labels is not None:
                pred_num = draw_pts_on_img(pred_img, pred_pts, pred_labels, COLORS_PRED, img_size, pc_range, line_width=3)
            pred_img = draw_car_and_legend(pred_img, img_size, pc_range, {0:0, 1:0, 2:0}, pred_num, f"Prediction (thr={args.score_thr})")
            cv2.imwrite(os.path.join(sample_dir, 'pred_bev.jpg'), pred_img)
            
            # 3. Combined (GT + Pred)
            combined_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            if gt_pts is not None and gt_labels is not None:
                draw_pts_on_img(combined_img, gt_pts, gt_labels, COLORS_GT, img_size, pc_range, line_width=2, point_size=2)
            if pred_pts is not None and pred_labels is not None:
                draw_pts_on_img(combined_img, pred_pts, pred_labels, COLORS_PRED, img_size, pc_range, line_width=3, point_size=4)
            combined_img = draw_car_and_legend(combined_img, img_size, pc_range, gt_num, pred_num, "GT + Pred")
            cv2.imwrite(os.path.join(sample_dir, 'combined_bev.jpg'), combined_img)
            
            # 4. Full combined (카메라 + BEV)
            cam_grid = create_camera_grid(img_metas, info_text=f"Sample {idx}")
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
