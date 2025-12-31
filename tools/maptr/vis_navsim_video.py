# -*- coding: utf-8 -*-
"""
NavSim GT + Prediction + Camera 영상 생성 도구
카메라 8개 + GT BEV Map + Prediction BEV Map을 분리해서 영상으로 저장
각 scene을 별도의 영상 파일로 저장 (테스트 데이터 전체 처리 가능)

먼저 이미지를 저장한 후 ffmpeg로 영상 생성

Usage:
    cd /home/byounggun/MapTR
    python tools/maptr/vis_navsim_video.py \
    projects/configs/maptr/maptr_tiny_r50_navsim_24e.py \
    --checkpoint /home/byounggun/MapTR/work_dirs_good/maptr_tiny_r50_navsim_24e/epoch_24.pth \
    --output-dir ./vis_navsim_videos/ \
    --all-scenes \
    --fps 5 \
    --score-thr 0.5
"""

import argparse
import os
import sys
import shutil
import subprocess

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
    0: (0, 165, 255),   # divider - orange (같은 색상 사용)
    1: (255, 0, 0),     # ped_crossing - blue
    2: (0, 255, 0),     # boundary - green
}
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def parse_args():
    parser = argparse.ArgumentParser(description='NavSim GT + Prediction 영상 생성 (각 scene별 별도 저장)')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', 
                        default='/home/byounggun/MapTR/work_dirs_good/maptr_tiny_r50_navsim_24e/epoch_24.pth',
                        help='checkpoint file path')
    parser.add_argument('--output-dir', default='./vis_navsim_videos/', help='output video directory')
    parser.add_argument('--all-scenes', action='store_true', help='process all scenes in test data')
    parser.add_argument('--num-scenes', type=int, default=5, help='number of scenes to include (ignored if --all-scenes)')
    parser.add_argument('--frames-per-scene', type=int, default=0, help='max frames per scene (0 = all frames)')
    parser.add_argument('--fps', type=int, default=5, help='video fps')
    parser.add_argument('--start-idx', type=int, default=0, help='start sample index')
    parser.add_argument('--score-thr', type=float, default=0.3, help='score threshold for predictions')
    parser.add_argument('--skip-scenes', type=int, default=0, help='number of scenes to skip')
    parser.add_argument('--min-frames', type=int, default=5, help='minimum frames per scene to include')
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
    
    if pts_list is None or labels is None:
        return num_elements
    
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


def draw_car(img, img_size, pc_range):
    """차량 그리기"""
    center = ego_to_img(0, 0, img_size, pc_range)
    car_pts = np.array([
        ego_to_img(2.5, -1, img_size, pc_range),
        ego_to_img(2.5, 1, img_size, pc_range),
        ego_to_img(-2.5, 1, img_size, pc_range),
        ego_to_img(-2.5, -1, img_size, pc_range),
    ], dtype=np.int32)
    cv2.fillPoly(img, [car_pts], (128, 128, 128))
    front = ego_to_img(4, 0, img_size, pc_range)
    cv2.arrowedLine(img, center, front, (0, 0, 255), 3, tipLength=0.3)


def draw_legend(img, num_elements, colors, y_start=30):
    """범례 그리기"""
    for i, name in enumerate(CLASS_NAMES):
        color = colors[i]
        y = y_start + i * 20
        cv2.rectangle(img, (10, y), (25, y + 12), color, -1)
        cv2.putText(img, f"{name}: {num_elements.get(i, 0)}", (30, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_single_bev(pts, labels, colors, pc_range, img_size, title=""):
    """단일 BEV 맵 그리기 (GT 또는 Pred)"""
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 맵 그리기
    num_elements = draw_pts_on_img(img, pts, labels, colors, img_size, pc_range, 
                                    line_width=2, point_size=3)
    
    # 차량 그리기
    draw_car(img, img_size, pc_range)
    
    # 타이틀
    cv2.putText(img, title, (img_size//2 - 40, 22),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # 범례
    draw_legend(img, num_elements, colors, y_start=40)
    
    return img, num_elements


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
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # Build dataset
    print("Building dataset...")
    cfg.data.train.test_mode = False
    dataset = build_dataset(cfg.data.train)
    
    pc_range = cfg.point_cloud_range
    
    # Scene 정보 미리 계산 (시작 인덱스와 프레임 수)
    print("Scanning scenes...")
    scene_info_list = []  # [(start_idx, frame_count, scene_token), ...]
    current_scene = None
    scene_start = 0
    
    for i in range(len(dataset)):
        info = dataset.data_infos[i]
        scene_token = info.get('scene_token', '')
        if scene_token != current_scene:
            # 이전 scene 저장
            if current_scene is not None:
                frame_count = i - scene_start
                scene_info_list.append((scene_start, frame_count, current_scene))
            scene_start = i
            current_scene = scene_token
    
    # 마지막 scene 저장
    if current_scene is not None:
        frame_count = len(dataset) - scene_start
        scene_info_list.append((scene_start, frame_count, current_scene))
    
    print(f"Found {len(scene_info_list)} total scenes")
    
    # 최소 프레임 수 필터링 (frames_per_scene 이상인 scene만)
    min_frames = max(5, args.frames_per_scene)  # 최소 5프레임 또는 요청 프레임 수
    valid_scenes = [(start, count, token) for start, count, token in scene_info_list if count >= min_frames]
    print(f"Scenes with >= {min_frames} frames: {len(valid_scenes)}")
    
    # 필요한 샘플 인덱스만 선택
    sample_indices = []
    selected_scenes = valid_scenes[args.skip_scenes : args.skip_scenes + args.num_scenes]
    
    for start, count, token in selected_scenes:
        end = start + min(args.frames_per_scene, count)
        sample_indices.extend(range(start, end))
    
    # 전체 scene 또는 지정된 수만큼 선택
    if args.all_scenes:
        selected_scenes = valid_scenes[args.skip_scenes:]
    else:
        selected_scenes = valid_scenes[args.skip_scenes : args.skip_scenes + args.num_scenes]
    
    print(f"Processing {len(selected_scenes)} scenes")
    
    # Video settings - GT와 Pred 분리해서 표시 (Inference 더 크게)
    # 레이아웃: [카메라 그리드 (3x3)] [GT BEV (작게) / Pred BEV (크게) 세로로]
    target_h, target_w = 270, 480
    cam_grid_h = target_h * 3  # 810
    cam_grid_w = target_w * 3  # 1440
    gt_bev_size = 300      # GT BEV 작게
    pred_bev_size = 510    # Pred BEV 크게 (inference)
    
    # 레이아웃: 카메라 + BEV (GT/Pred 세로로 쌓음 = 810)
    # H.264는 width/height 둘 다 짝수여야 함
    bev_column_width = max(gt_bev_size, pred_bev_size)
    total_w = cam_grid_w + bev_column_width  # 1440 + 510 = 1950 (짝수)
    total_h = cam_grid_h  # 810 (짝수)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating videos for each scene (SEPARATE FILES)")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Score threshold: {args.score_thr}")
    print(f"Total scenes to process: {len(selected_scenes)}")
    print(f"Skip scenes: {args.skip_scenes}")
    print(f"Output directory: {args.output_dir}")
    print(f"Resolution: {total_w}x{total_h}, FPS: {args.fps}")
    print(f"GT BEV size: {gt_bev_size}, Pred BEV size: {pred_bev_size}")
    print(f"{'='*60}\n")
    
    total_videos_created = 0
    
    # 각 scene별로 별도 영상 생성
    for scene_idx, (scene_start, scene_frame_count, scene_token) in enumerate(selected_scenes):
        scene_num = scene_idx + 1
        
        # 프레임 수 결정
        if args.frames_per_scene > 0:
            num_frames = min(args.frames_per_scene, scene_frame_count)
        else:
            num_frames = scene_frame_count
        
        sample_indices = list(range(scene_start, scene_start + num_frames))
        
        # 임시 이미지 저장 디렉토리 (scene별)
        temp_dir = os.path.join(args.output_dir, f'temp_frames_scene_{scene_num:03d}')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 출력 비디오 파일명
        safe_token = scene_token[:30].replace('/', '_').replace('\\', '_')
        output_video_path = os.path.join(args.output_dir, f'scene_{scene_num:03d}_{safe_token}.mp4')
        
        # 이미 생성된 영상이 있으면 스킵
        if os.path.exists(output_video_path):
            print(f"\n[Scene {scene_num}/{len(selected_scenes)}] SKIP - already exists: {os.path.basename(output_video_path)}")
            total_videos_created += 1  # 기존 영상도 카운트
            continue
        
        print(f"\n[Scene {scene_num}/{len(selected_scenes)}] {scene_token[:50]}")
        print(f"  Frames: {num_frames}, Output: {os.path.basename(output_video_path)}")
        
        prog_bar = mmcv.ProgressBar(len(sample_indices))
        frames_written = 0
        
        for sample_idx in sample_indices:
            try:
                # 데이터 로드 (인덱스로 직접 접근)
                data = dataset[sample_idx]
                
                # collate 처리 (배치 형태로 변환)
                from mmcv.parallel import collate
                data = collate([data], samples_per_gpu=1)
                
                # GPU로 이동
                if 'img' in data:
                    if hasattr(data['img'], 'data'):
                        data['img'] = data['img'].data[0].cuda()
                    elif isinstance(data['img'], list):
                        data['img'] = data['img'][0].cuda()
                    else:
                        data['img'] = data['img'].cuda()
                
                # img_metas 추출
                img_metas_raw = data['img_metas']
                if hasattr(img_metas_raw, 'data'):
                    metas_dict = img_metas_raw.data[0][0]
                    img_metas = metas_dict[0] if isinstance(metas_dict, dict) and 0 in metas_dict else metas_dict
                elif isinstance(img_metas_raw, list):
                    metas_dict = img_metas_raw[0][0] if isinstance(img_metas_raw[0], list) else img_metas_raw[0]
                    img_metas = metas_dict[0] if isinstance(metas_dict, dict) and 0 in metas_dict else metas_dict
                else:
                    img_metas = img_metas_raw
                
                # GT 추출
                gt_labels = None
                gt_pts = None
                
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
                if gt_pts is not None and isinstance(gt_pts, torch.Tensor):
                    gt_pts = gt_pts.cpu().numpy()
                if gt_labels is not None and isinstance(gt_labels, torch.Tensor):
                    gt_labels = gt_labels.cpu().numpy()
                
                # Fix img_metas format for model inference
                if 'img_metas' in data and hasattr(data['img_metas'], 'data'):
                    metas_data = data['img_metas'].data
                    for batch_idx in range(len(metas_data)):
                        for s_idx in range(len(metas_data[batch_idx])):
                            metas_dict = metas_data[batch_idx][s_idx]
                            if isinstance(metas_dict, dict):
                                metas_list = [metas_dict[k] for k in sorted(metas_dict.keys())]
                                metas_data[batch_idx][s_idx] = metas_list
                
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
                
                # GT BEV 그리기 (작게)
                gt_bev, gt_num = draw_single_bev(gt_pts, gt_labels, COLORS_GT, pc_range, gt_bev_size, "GT")
                
                # Pred BEV 그리기 (크게 - Inference)
                pred_bev, pred_num = draw_single_bev(pred_pts, pred_labels, COLORS_PRED, pc_range, pred_bev_size, 
                                                      f"Inference (thr={args.score_thr})")
                
                # GT와 Pred를 세로로 쌓기 (크기가 다르므로 패딩 추가)
                # GT를 Pred width에 맞게 패딩
                gt_padded = np.ones((gt_bev_size, bev_column_width, 3), dtype=np.uint8) * 200
                gt_offset = (bev_column_width - gt_bev_size) // 2
                gt_padded[:, gt_offset:gt_offset + gt_bev_size, :] = gt_bev
                
                # Pred도 column width에 맞게 (이미 510이면 그대로)
                pred_padded = np.ones((pred_bev_size, bev_column_width, 3), dtype=np.uint8) * 200
                pred_offset = (bev_column_width - pred_bev_size) // 2
                pred_padded[:, pred_offset:pred_offset + pred_bev_size, :] = pred_bev
                
                bev_combined = cv2.vconcat([gt_padded, pred_padded])
                
                # 카메라 그리드 생성
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
                            cv2.putText(info_img, "NavSim + MapTR", (target_w//2-80, target_h//2-30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(info_img, f"Frame: {frames_written+1}", (target_w//2-40, target_h//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                            cv2.putText(info_img, f"Scene: {scene_num}", (target_w//2-45, target_h//2+25),
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
                
                cam_grid = cv2.vconcat([row1, row2, row3])
                
                # 합치기: 카메라 + BEV (GT/Pred 세로)
                combined = cv2.hconcat([cam_grid, bev_combined])
                
                # 프레임 정보 추가
                info_text = f"Scene {scene_num}/{len(selected_scenes)} | Frame {frames_written+1}/{len(sample_indices)}"
                cv2.putText(combined, info_text, (10, total_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(combined, info_text, (10, total_h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                
                # 이미지 저장
                frame_path = os.path.join(temp_dir, f'frame_{frames_written:05d}.jpg')
                cv2.imwrite(frame_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frames_written += 1
                
            except Exception as e:
                print(f"\nError processing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()
            
            prog_bar.update()
        
        # ffmpeg로 영상 생성 (각 scene별)
        print(f"\n  Generating video with ffmpeg...")
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(args.fps),
            '-i', os.path.join(temp_dir, 'frame_%05d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_video_path
        ]
        
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"  Video created: {output_video_path}")
            total_videos_created += 1
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg error: {e.stderr.decode()}")
            # Fallback: OpenCV VideoWriter
            print("  Trying OpenCV VideoWriter as fallback...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, args.fps, (total_w, total_h))
            for i in range(frames_written):
                frame_path = os.path.join(temp_dir, f'frame_{i:05d}.jpg')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    out.write(frame)
            out.release()
            total_videos_created += 1
        except FileNotFoundError:
            print("  ffmpeg not found. Trying OpenCV VideoWriter...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, args.fps, (total_w, total_h))
            for i in range(frames_written):
                frame_path = os.path.join(temp_dir, f'frame_{i:05d}.jpg')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    out.write(frame)
            out.release()
            total_videos_created += 1
        
        # 임시 파일 정리
        shutil.rmtree(temp_dir)
    
    print(f"\n{'='*60}")
    print(f"Done! All videos saved to: {args.output_dir}")
    print(f"Total videos created: {total_videos_created}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
