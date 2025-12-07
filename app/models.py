"""
FastAPI 응답 모델 (Pydantic 스키마)
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic.alias_generators import to_camel


class NutritionInfo(BaseModel):
    """영양소 정보 (확장)"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

    # 핵심 영양소 (필수)
    calories_kcal: float = Field(..., ge=0, description="열량 (kcal)")
    protein_g: float = Field(..., ge=0, description="단백질 (g)")
    fat_g: float = Field(..., ge=0, description="지방 (g)")
    carbs_g: float = Field(..., ge=0, description="탄수화물 (g)")

    # 주요 영양소 (선택 - CSV DB에서만 제공)
    water_g: Optional[float] = Field(None, ge=0, description="수분 (g)")
    sugars_g: Optional[float] = Field(None, ge=0, description="당류 (g)")
    dietary_fiber_g: Optional[float] = Field(None, ge=0, description="식이섬유 (g)")
    sodium_mg: Optional[float] = Field(None, ge=0, description="나트륨 (mg)")
    cholesterol_mg: Optional[float] = Field(None, ge=0, description="콜레스테롤 (mg)")
    saturated_fat_g: Optional[float] = Field(None, ge=0, description="포화지방산 (g)")

    # 주요 미네랄 (선택)
    calcium_mg: Optional[float] = Field(None, ge=0, description="칼슘 (mg)")
    iron_mg: Optional[float] = Field(None, ge=0, description="철 (mg)")

    # 주요 비타민 (선택)
    vitamin_a_ug: Optional[float] = Field(None, ge=0, description="비타민 A (μg RAE)")
    vitamin_c_mg: Optional[float] = Field(None, ge=0, description="비타민 C (mg)")

    @field_validator('*')
    @classmethod
    def round_to_two_decimals(cls, v):
        """소수점 2자리로 반올림 (None 허용)"""
        return round(v, 2) if v is not None else None


class AnalysisResponse(BaseModel):
    """음식 분석 결과"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "foodName": "비빔밥",
                "confidence": 0.94,
                "volumeMl": 350.50,
                "massG": 350.50,
                "nutrition": {
                    "caloriesKcal": 525.75,
                    "proteinG": 28.04,
                    "fatG": 17.53,
                    "carbsG": 87.63
                }
            }
        }
    )

    food_name: str = Field(..., description="음식 이름")
    confidence: float = Field(..., ge=0, le=1, description="분류 신뢰도 (0~1)")
    volume_ml: float = Field(..., ge=0, description="부피 (ml)")
    mass_g: float = Field(..., ge=0, description="질량 (g)")
    nutrition: NutritionInfo

    @field_validator('confidence', 'volume_ml', 'mass_g')
    @classmethod
    def round_values(cls, v):
        """소수점 2자리로 반올림"""
        return round(v, 2)


class ErrorResponse(BaseModel):
    """에러 응답"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

    detail: str = Field(..., description="에러 메시지")


class ProgressUpdate(BaseModel):
    """진행 상황 업데이트 (SSE용)"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True
    )

    step: int = Field(..., ge=1, le=4, description="현재 단계 (1~4)")
    total_steps: int = Field(4, description="전체 단계 수")
    message: str = Field(..., description="진행 상황 메시지")
    status: Literal["in_progress", "completed", "error"] = Field(..., description="상태")
    result: Optional[AnalysisResponse] = Field(None, description="최종 결과 (완료 시)")
