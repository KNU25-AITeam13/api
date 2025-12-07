# -*- coding: utf-8 -*-
"""
음식 영양 분석 파이프라인

3개 AI 모델을 통합하여 음식 이미지에서 영양 정보를 추정합니다:
1. Depth Pro: 깊이 맵 생성
2. YOLO Segmentation: 객체 검출 및 부피 계산
3. Food Classification: 음식 분류
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
import cv2
import urllib.request

# Submodule paths
current_dir = Path(__file__).parent
sys.path.append(str(current_dir / 'depth_pro' / 'src'))
sys.path.append(str(current_dir / 'volume_assumption'))
sys.path.append(str(current_dir / 'food_classification' / 'src'))

# Imports from submodules
import depth_pro
from volume_test import volume_calculation_core
from ultralytics import YOLO
from predict import predict_single

# Nutrition database
from ai.nutrition import NutritionDatabase


class FoodAnalyzer:
    """음식 영양 분석 파이프라인"""

    # 레퍼런스 물체 기본 크기 (단위: cm)
    DEFAULT_SIZES = {
        'spoon': 18.0,
        'fork': 19.0,
        'knife': 22.0,
        'chopsticks': 21.0,
    }

    # YOLO 클래스 분류
    CUTLERY_LIKE = {'spoon', 'fork', 'knife', 'chopsticks'}
    PLATE_LIKE = {'plate', 'bowl', 'cup', 'wine glass', 'tray'}
    FOOD_LIKE = {
        'food', 'rice', 'noodles', 'pizza', 'sandwich', 'salad', 'cake', 'donut',
        'banana', 'apple', 'orange', 'broccoli', 'carrot', 'hot dog', 'burger',
        'steak', 'bread'
    }

    def __init__(self, yolo_weights_path: str, food_model_path: str):
        """
        FoodAnalyzer 초기화

        Args:
            yolo_weights_path: YOLO segmentation 모델 경로 (부피 측정용)
            food_model_path: 음식 분류 모델 경로
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[FoodAnalyzer] Using device: {self.device}")

        # 0. Depth Pro 체크포인트 확인 및 다운로드
        checkpoint_path = self._ensure_depth_pro_checkpoint()

        # 1. Depth Pro 모델 로드
        print("[FoodAnalyzer] Loading Depth Pro model...")

        # 절대 경로로 config 생성
        from depth_pro.depth_pro import DepthProConfig
        config = DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=str(checkpoint_path),  # 절대 경로 사용
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )

        self.depth_model, self.depth_transform = depth_pro.create_model_and_transforms(
            config=config,
            device=self.device
        )
        self.depth_model.eval()

        # 2. YOLO Segmentation 모델 로드 (부피 측정용)
        print(f"[FoodAnalyzer] Loading YOLO segmentation model from {yolo_weights_path}...")
        self.yolo_model = YOLO(yolo_weights_path)

        # 3. Food Classification 모델 로드
        print(f"[FoodAnalyzer] Loading Food Classification model from {food_model_path}...")
        self.food_model = YOLO(food_model_path)

        # 4. Nutrition Database 로드
        print("[FoodAnalyzer] Loading Nutrition Database...")
        self.nutrition_db = NutritionDatabase()

        print("[FoodAnalyzer] All models loaded successfully!")

    def analyze(self, image_path: str) -> dict:
        """
        음식 이미지를 분석하여 영양 정보를 추정합니다.

        Args:
            image_path: 이미지 파일 경로

        Returns:
            {
                'food_name': str,          # 음식 이름
                'confidence': float,        # 분류 신뢰도 (0~1)
                'volume_ml': float,         # 부피 (ml)
                'mass_g': float,            # 질량 (g)
                'nutrition': {              # 영양소 정보
                    'calories_kcal': float,
                    'protein_g': float,
                    'fat_g': float,
                    'carbs_g': float
                }
            }
        """
        print(f"\n[FoodAnalyzer] Analyzing image: {image_path}")
        print("=" * 60)

        # Step 1: Food Classification
        print("[1/4] Running Food Classification...")
        food_result = predict_single(self.food_model, image_path)
        food_name = food_result.get('top1_class', 'Unknown')
        confidence = food_result.get('top1_confidence', 0.0)
        print(f"  → Detected: {food_name} (confidence: {confidence:.2%})")

        # Step 2: Depth Map 생성
        print("[2/4] Generating Depth Map with Depth Pro...")
        image, _, f_px = depth_pro.load_rgb(image_path)
        image_tensor = self.depth_transform(image).to(self.device)

        with torch.no_grad():
            prediction = self.depth_model.infer(image_tensor, f_px=f_px)

        depth_map = prediction["depth"].cpu().numpy()

        # Depth Pro의 초점거리 추출 (Fallback 모드에서 활용)
        focallength_px = prediction["focallength_px"]
        f_px_val = focallength_px.item() if focallength_px is not None else None
        print(f"  → Depth map shape: {depth_map.shape}")
        if f_px_val:
            print(f"  → Focal length (px): {f_px_val:.1f}")

        # Step 3: YOLO Segmentation
        print("[3/4] Running YOLO Segmentation...")
        detected_refs, food_mask, bg_candidate_mask = self._run_yolo_segmentation(
            image_path, depth_map.shape
        )

        # Step 4: 레퍼런스 물체 확인 및 부피 계산
        print("[4/4] Calculating Volume...")
        ref_px_len, ref_real_cm, is_fallback = self._detect_reference_object(detected_refs)

        # 음식별 밀도 조회 (DB 기반 또는 fallback)
        density = self.nutrition_db.get_density(food_name)

        vol_result = volume_calculation_core(
            Z_scene=depth_map,
            food_mask=food_mask,
            bg_mask_candidate=bg_candidate_mask,
            ref_px_len=ref_px_len,
            ref_real_cm=ref_real_cm,
            density_g_per_ml=density,  # DB에서 조회한 밀도 사용
            is_fallback_mode=is_fallback,
            provided_f_px=f_px_val  # Depth Pro 초점거리 전달
        )

        print(f"  → Density: {density:.2f} g/ml")
        print(f"  → Volume: {vol_result['volume_ml']:.1f} ml")
        print(f"  → Mass: {vol_result['mass_g']:.1f} g")
        print(f"  → Method: {vol_result['method']}")

        # Step 5: 영양소 계산 (DB 기반)
        nutrition = self._calculate_nutrition(food_name, vol_result['mass_g'])

        print("=" * 60)
        print("[FoodAnalyzer] Analysis complete!\n")

        return {
            'food_name': food_name,
            'confidence': confidence,
            'volume_ml': vol_result['volume_ml'],
            'mass_g': vol_result['mass_g'],
            'nutrition': nutrition
        }

    def _run_yolo_segmentation(self, image_path: str, depth_shape_hw: tuple) -> tuple:
        """
        YOLO segmentation을 실행하여 음식, 레퍼런스 물체, 배경을 분리합니다.

        volume_test.py의 yolo_inference 로직을 복사하여 구현
        (미리 로드된 모델 사용으로 성능 향상)

        Args:
            image_path: 이미지 경로
            depth_shape_hw: Depth map 크기 (H, W)

        Returns:
            (detected_refs, food_mask, bg_candidate_mask)
        """
        H, W = depth_shape_hw

        # YOLO inference
        results = self.yolo_model(image_path, imgsz=640, conf=0.15, verbose=False)

        if not results or results[0].masks is None:
            # 아무것도 감지되지 않음
            return {}, np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=bool)

        res = results[0]
        masks = res.masks.data.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        names = res.names

        detected_refs = {}  # 레퍼런스 물체 (수저 등)
        max_area_per_cls = {}  # 클래스별 최대 면적
        food_mask_accum = np.zeros((H, W), dtype=bool)
        bg_candidate_mask = np.zeros((H, W), dtype=bool)

        for mi, ci in zip(masks, clses):
            cls_name = names[ci]
            m_resized = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            current_area = m_resized.sum()

            if cls_name in self.FOOD_LIKE:
                # 음식 마스크
                food_mask_accum |= m_resized

            elif cls_name in self.CUTLERY_LIKE:
                # 레퍼런스 물체 (수저류)
                bg_candidate_mask |= m_resized
                if cls_name not in max_area_per_cls or current_area > max_area_per_cls[cls_name]:
                    detected_refs[cls_name] = m_resized
                    max_area_per_cls[cls_name] = current_area

            elif cls_name in self.PLATE_LIKE or cls_name == 'dining table':
                # 배경 후보 (접시, 테이블 등)
                bg_candidate_mask |= m_resized

        return detected_refs, food_mask_accum, bg_candidate_mask

    def _detect_reference_object(self, detected_refs: dict) -> tuple:
        """
        레퍼런스 물체를 감지하여 스케일 정보를 반환합니다.

        Args:
            detected_refs: 감지된 레퍼런스 물체들 (클래스명 → 마스크)

        Returns:
            (ref_px_len, ref_real_cm, is_fallback)
        """
        priority = ['spoon', 'fork', 'knife', 'chopsticks']

        for item in priority:
            if item in detected_refs:
                ref_px_len = self._get_max_pixel_dimension(detected_refs[item])
                if ref_px_len > 10:  # 최소 크기 필터
                    ref_real_cm = self.DEFAULT_SIZES[item]
                    print(f"  → Reference object detected: {item} ({ref_real_cm}cm)")
                    return ref_px_len, ref_real_cm, False

        print("  → No reference object detected (Fallback mode)")
        return 0, 0, True  # Fallback mode

    @staticmethod
    def _get_max_pixel_dimension(mask: np.ndarray) -> int:
        """마스크의 최대 픽셀 치수를 계산합니다."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return 0
        largest = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest)
        return int(max(rect[1]))

    @staticmethod
    def _ensure_depth_pro_checkpoint():
        """
        Depth Pro 체크포인트가 없으면 자동으로 다운로드합니다.
        Docker 빌드 시 이미 포함되어 있으면 스킵됩니다.

        Returns:
            Path: 체크포인트 파일 경로
        """
        checkpoint_dir = current_dir / 'depth_pro' / 'checkpoints'
        checkpoint_path = checkpoint_dir / 'depth_pro.pt'

        if checkpoint_path.exists():
            print(f"[FoodAnalyzer] Depth Pro checkpoint found at {checkpoint_path}")
            return checkpoint_path

        print("[FoodAnalyzer] Depth Pro checkpoint not found. Downloading...")
        print("=" * 60)

        # 체크포인트 디렉토리 생성
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 다운로드 URL
        url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"

        try:
            # 다운로드 (진행 상황 표시)
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                bar_length = 50
                filled = int(bar_length * downloaded / total_size)
                bar = '█' * filled + '░' * (bar_length - filled)
                print(f"\r  Progress: [{bar}] {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end='')

            urllib.request.urlretrieve(url, checkpoint_path, reporthook=show_progress)
            print()  # 새 줄
            print(f"[FoodAnalyzer] Download complete! Checkpoint saved to {checkpoint_path}")
            print("=" * 60)

        except Exception as e:
            print(f"\n[FoodAnalyzer] Failed to download checkpoint: {e}")
            print("Please download manually:")
            print(f"  cd {checkpoint_dir.parent}")
            print(f"  bash get_pretrained_models.sh")
            raise

        return checkpoint_path

    def _calculate_nutrition(self, food_name: str, mass_g: float) -> dict:
        """
        CSV 데이터베이스에서 실제 영양소 정보를 조회합니다.

        Args:
            food_name: 음식 이름 (YOLO 클래스명)
            mass_g: 질량 (g)

        Returns:
            영양소 정보 (14개 필드: 4개 필수 + 10개 선택)
        """
        return self.nutrition_db.get_nutrition(food_name, mass_g)
