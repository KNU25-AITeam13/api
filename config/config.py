"""
애플리케이션 설정 관리
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # 모델 경로
    yolo_seg_weights: str = "yolo11x-seg.pt"  # YOLOv11 Extra Large Segmentation
    food_model_weights: str = "ai/food_classification/models/best_mixed_food_v1.pt"

    # 파일 업로드 설정
    max_upload_size_mb: int = 10

    # API 설정
    api_title: str = "AI API"
    api_description: str = "음식 이미지를 통한 영양 성분 분석 API"
    api_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 싱글톤 인스턴스
settings = Settings()
