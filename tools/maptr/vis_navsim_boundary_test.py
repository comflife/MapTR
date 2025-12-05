# -*- coding: utf-8 -*-
"""
NavSim Boundary 테스트 시각화
다양한 boundary 옵션을 비교해서 시각화

Usage:
    cd /home/byounggun/MapTR
    conda activate navsim
    python tools/maptr/vis_navsim_boundary_test.py
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import box, LineString, MultiLineString, MultiPolygon
from shapely import ops, affinity
from pyquaternion import Quaternion

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB
from projects.mmdet3d_plugin.datasets.navsim_map_dataset import NuPlanMapWrapper

# 설정
DATA_ROOT = 'data/navsim'
MAP_ROOT = 'data/navsim/download/maps'
PATCH_SIZE = (102.4, 102.4)
LOCATION = 'us-nv-las-vegas-strip'

# 테스트할 위치 (ego position)
TEST_POSITIONS = [
    (664500, 3998000),  # 위치 1
    (664800, 3999500),  # 위치 2
    (665000, 3997500),  # 위치 3
]

# 색상
COLORS = {
    'divider': (0, 165, 255),      # orange
    'ped_crossing': (255, 0, 0),   # blue
    'boundary_0': (0, 255, 0),     # green - type 0
    'boundary_2': (255, 255, 0),   # cyan - type 2
    'boundary_3': (0, 0, 255),     # red - type 3
    'road_segments': (255, 0, 255), # magenta - road_segments 외곽
}


def init_map():
    """맵 초기화"""
    maps_db = GPKGMapsDB(map_root=MAP_ROOT, map_version="nuplan-maps-v1.0")
    map_api = NuPlanMapWrapper(maps_db, LOCATION)
    return map_api


def get_boundaries_by_type(map_api, patch_box, patch_angle, boundary_type_fid):
    """특정 boundary_type_fid의 boundary 추출"""
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch = map_api.get_patch_coord(patch_box, patch_angle)
    
    max_x = PATCH_SIZE[1] / 2
    max_y = PATCH_SIZE[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    lines = []
    try:
        records = map_api.load_vector_layer('boundaries')
        for idx, row in records.iterrows():
            if row.get('boundary_type_fid', -1) != boundary_type_fid:
                continue
            
            geometry = row['geometry']
            if geometry is None or geometry.is_empty:
                continue
            
            new_line = geometry.intersection(patch)
            if new_line.is_empty:
                continue
            
            new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            new_line = new_line.intersection(local_patch)
            
            if new_line.is_empty:
                continue
            
            if new_line.geom_type == 'LineString':
                if new_line.length > 1.0:
                    lines.append(new_line)
            elif new_line.geom_type == 'MultiLineString':
                for line in new_line.geoms:
                    if line.length > 1.0:
                        lines.append(line)
    except Exception as e:
        print(f"Error: {e}")
    
    return lines


def get_road_segments_boundary(map_api, patch_box, patch_angle):
    """road_segments를 union해서 도로 외곽선 추출 (가장 바깥 boundary)"""
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch = map_api.get_patch_coord(patch_box, patch_angle)
    
    max_x = PATCH_SIZE[1] / 2
    max_y = PATCH_SIZE[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    polygon_list = []
    try:
        records = map_api.load_vector_layer('road_segments')
        for idx, row in records.iterrows():
            geometry = row['geometry']
            if geometry is None or geometry.is_empty:
                continue
            
            new_polygon = geometry.intersection(patch)
            if new_polygon.is_empty:
                continue
            
            new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            
            if new_polygon.geom_type == 'Polygon':
                polygon_list.append(new_polygon)
            elif new_polygon.geom_type == 'MultiPolygon':
                polygon_list.extend(list(new_polygon.geoms))
    except Exception as e:
        print(f"Error loading road_segments: {e}")
        return []
    
    if not polygon_list:
        return []
    
    # Union all polygons
    try:
        union_road = ops.unary_union(polygon_list)
    except:
        return []
    
    if union_road.is_empty:
        return []
    
    # Extract exterior boundaries
    if union_road.geom_type == 'Polygon':
        union_road = MultiPolygon([union_road])
    
    lines = []
    for poly in union_road.geoms:
        # Exterior (outer boundary)
        ext = LineString(poly.exterior.coords)
        clipped = ext.intersection(local_patch)
        if not clipped.is_empty:
            if clipped.geom_type == 'MultiLineString':
                clipped = ops.linemerge(clipped)
            if clipped.geom_type == 'LineString' and clipped.length > 1.0:
                lines.append(clipped)
            elif clipped.geom_type == 'MultiLineString':
                for line in clipped.geoms:
                    if line.length > 1.0:
                        lines.append(line)
        
        # Interiors (holes - islands etc)
        for interior in poly.interiors:
            inter_line = LineString(interior.coords)
            clipped = inter_line.intersection(local_patch)
            if not clipped.is_empty:
                if clipped.geom_type == 'MultiLineString':
                    clipped = ops.linemerge(clipped)
                if clipped.geom_type == 'LineString' and clipped.length > 1.0:
                    lines.append(clipped)
                elif clipped.geom_type == 'MultiLineString':
                    for line in clipped.geoms:
                        if line.length > 1.0:
                            lines.append(line)
    
    return lines


def get_lane_dividers(map_api, patch_box, patch_angle):
    """lanes_polygons에서 divider 추출"""
    patch_x, patch_y = patch_box[0], patch_box[1]
    patch = map_api.get_patch_coord(patch_box, patch_angle)
    
    max_x = PATCH_SIZE[1] / 2
    max_y = PATCH_SIZE[0] / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    
    lane_polygons = []
    try:
        records = map_api.load_vector_layer('lanes_polygons')
        for idx, row in records.iterrows():
            geometry = row['geometry']
            if geometry is None or geometry.is_empty:
                continue
            clipped = geometry.intersection(patch)
            if clipped.is_empty:
                continue
            clipped = affinity.rotate(clipped, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            clipped = affinity.affine_transform(clipped, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            
            if clipped.geom_type == 'Polygon':
                lane_polygons.append(clipped)
            elif clipped.geom_type == 'MultiPolygon':
                lane_polygons.extend(list(clipped.geoms))
    except Exception as e:
        return []
    
    if len(lane_polygons) < 2:
        return []
    
    try:
        all_lanes_union = ops.unary_union(lane_polygons)
    except:
        return []
    
    all_boundaries = []
    for poly in lane_polygons:
        all_boundaries.append(poly.exterior)
    
    try:
        all_lines = ops.unary_union(all_boundaries)
    except:
        return []
    
    if all_lanes_union.geom_type == 'Polygon':
        outer_boundary = all_lanes_union.exterior
    elif all_lanes_union.geom_type == 'MultiPolygon':
        outer_boundary = ops.unary_union([p.exterior for p in all_lanes_union.geoms])
    else:
        outer_boundary = None
    
    if outer_boundary is not None:
        try:
            divider_lines = all_lines.difference(outer_boundary.buffer(0.5))
        except:
            divider_lines = all_lines
    else:
        divider_lines = all_lines
    
    line_list = []
    
    def extract_lines(geom):
        if geom is None or geom.is_empty:
            return
        if geom.geom_type == 'LineString':
            clipped = geom.intersection(local_patch)
            if not clipped.is_empty and clipped.length > 2.0:
                if clipped.geom_type == 'LineString':
                    line_list.append(clipped)
                elif clipped.geom_type == 'MultiLineString':
                    for line in clipped.geoms:
                        if line.length > 2.0:
                            line_list.append(line)
        elif geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                extract_lines(line)
        elif geom.geom_type == 'GeometryCollection':
            for g in geom.geoms:
                extract_lines(g)
        elif geom.geom_type == 'LinearRing':
            extract_lines(LineString(geom.coords))
    
    extract_lines(divider_lines)
    return line_list


def draw_lines_on_img(img, lines, color, img_size, pc_range):
    """선들을 이미지에 그리기"""
    def ego_to_img(ego_x, ego_y):
        scale = img_size / (pc_range[3] - pc_range[0])
        img_x = int(((-ego_y) - pc_range[1]) * scale)
        img_y = int((pc_range[3] - ego_x) * scale)
        return img_x, img_y
    
    for line in lines:
        if line.geom_type != 'LineString':
            continue
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            pt1 = ego_to_img(coords[i][0], coords[i][1])
            pt2 = ego_to_img(coords[i+1][0], coords[i+1][1])
            cv2.line(img, pt1, pt2, color, 2)
    
    return len(lines)


def visualize_comparison(map_api, ego_pos, output_path):
    """여러 boundary 옵션을 비교 시각화"""
    patch_box = (ego_pos[0], ego_pos[1], PATCH_SIZE[0], PATCH_SIZE[1])
    patch_angle = 0
    
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    img_size = 800
    
    # 6개 서브플롯: divider, boundary_0, boundary_2, boundary_3, road_segments, combined
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    titles = [
        'Divider (lanes_polygons)',
        'Boundary Type 0',
        'Boundary Type 2',
        'Boundary Type 3',
        'Road Segments (외곽)',
        'Combined View'
    ]
    
    # 각 타입별 추출
    dividers = get_lane_dividers(map_api, patch_box, patch_angle)
    boundary_0 = get_boundaries_by_type(map_api, patch_box, patch_angle, 0)
    boundary_2 = get_boundaries_by_type(map_api, patch_box, patch_angle, 2)
    boundary_3 = get_boundaries_by_type(map_api, patch_box, patch_angle, 3)
    road_seg = get_road_segments_boundary(map_api, patch_box, patch_angle)
    
    all_lines = [dividers, boundary_0, boundary_2, boundary_3, road_seg]
    all_colors = [
        COLORS['divider'],
        COLORS['boundary_0'],
        COLORS['boundary_2'],
        COLORS['boundary_3'],
        COLORS['road_segments']
    ]
    
    for idx, ax in enumerate(axes.flatten()):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        if idx < 5:
            # 개별 타입
            count = draw_lines_on_img(img, all_lines[idx], all_colors[idx], img_size, pc_range)
            ax.set_title(f'{titles[idx]} ({count} lines)', fontsize=12)
        else:
            # Combined
            for lines, color in zip(all_lines, all_colors):
                draw_lines_on_img(img, lines, color, img_size, pc_range)
            ax.set_title(titles[idx], fontsize=12)
        
        # 차량 표시
        def ego_to_img(ego_x, ego_y):
            scale = img_size / (pc_range[3] - pc_range[0])
            img_x = int(((-ego_y) - pc_range[1]) * scale)
            img_y = int((pc_range[3] - ego_x) * scale)
            return img_x, img_y
        
        center = ego_to_img(0, 0)
        cv2.circle(img, center, 10, (128, 128, 128), -1)
        front = ego_to_img(5, 0)
        cv2.arrowedLine(img, center, front, (0, 0, 255), 3)
        
        # BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis('off')
    
    # 범례 추가
    legend_labels = [
        f'Divider (orange): {len(dividers)}',
        f'Boundary 0 (green): {len(boundary_0)}',
        f'Boundary 2 (cyan): {len(boundary_2)}',
        f'Boundary 3 (red): {len(boundary_3)}',
        f'Road Segments (magenta): {len(road_seg)}'
    ]
    fig.text(0.5, 0.02, ' | '.join(legend_labels), ha='center', fontsize=10)
    
    plt.suptitle(f'Boundary Comparison at ({ego_pos[0]:.0f}, {ego_pos[1]:.0f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    print(f"  Divider: {len(dividers)}, Boundary0: {len(boundary_0)}, Boundary2: {len(boundary_2)}, Boundary3: {len(boundary_3)}, RoadSeg: {len(road_seg)}")


def main():
    print("Initializing map...")
    map_api = init_map()
    
    output_dir = './vis_boundary_test'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating comparison visualizations...")
    for i, pos in enumerate(TEST_POSITIONS):
        output_path = os.path.join(output_dir, f'boundary_comparison_{i+1}.png')
        visualize_comparison(map_api, pos, output_path)
    
    print(f"\nDone! Results saved to {output_dir}/")
    print("\n범례:")
    print("  - Divider (orange): 차선 사이 구분선")
    print("  - Boundary 0 (green): boundary_type_fid=0")
    print("  - Boundary 2 (cyan): boundary_type_fid=2") 
    print("  - Boundary 3 (red): boundary_type_fid=3")
    print("  - Road Segments (magenta): road_segments polygon 외곽선 (가장 바깥 도로 경계)")


if __name__ == '__main__':
    main()
