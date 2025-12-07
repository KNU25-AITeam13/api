"""
영양정보 모듈

YOLOv11 음식 인식 모델과 전국통합식품영양성분정보 DB를 연동하여
실제 영양정보를 제공합니다.
"""

from ai.nutrition.nutrition_lookup import NutritionDatabase

__all__ = ['NutritionDatabase']
