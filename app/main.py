from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from pathlib import Path
from typing import Optional

# ============================================================================
# Pydantic Models
# ============================================================================


class Prediction(BaseModel):
    """단일 예측 결과"""

    class_id: int
    class_name: str
    confidence: float
    confidence_pct: str


class SinglePredictionResponse(BaseModel):
    """단일 이미지 예측 응답"""

    success: bool
    top1_class: str
    top1_confidence: float
    predictions: list[Prediction]


class ImagePredictionResult(BaseModel):
    """배치 예측 시 개별 이미지 결과"""

    filename: str
    success: bool
    message: Optional[str] = None
    top1_class: Optional[str] = None
    top1_confidence: Optional[float] = None
    predictions: Optional[list[Prediction]] = None


class BatchPredictionResponse(BaseModel):
    """배치 예측 응답"""

    total: int
    success_count: int
    failure_count: int
    results: list[ImagePredictionResult]


class ClassListResponse(BaseModel):
    """클래스 목록 응답"""

    total: int
    classes: dict[int, str]


class HealthResponse(BaseModel):
    """헬스 체크 응답"""

    status: str
    model_loaded: bool
    model_path: str
    num_classes: int


# ============================================================================
# App Initialization
# ============================================================================

MODEL_PATH = Path("ai/food_classification/models/best_mixed_food_v1.pt")
CONFIDENCE_THRESHOLD = 0.5  # 50% 이하 신뢰도 시 실패 처리

app = FastAPI(
    title="Food Classification API",
    description="YOLOv11 기반 음식 분류 API - 39개 음식 클래스(한식 20개 + 국제음식 19개) 지원",
    version="1.0.0",
)

# 모델 전역 로드 (앱 시작 시 1회)
model: YOLO = None


@app.on_event("startup")
async def load_model():
    """앱 시작 시 모델 로드"""
    global model
    if MODEL_PATH.exists():
        model = YOLO(str(MODEL_PATH))
        print(f"✅ 모델 로드 완료: {MODEL_PATH}")
        print(f"   - 클래스 수: {len(model.names)}")
    else:
        print(f"⚠️ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")


# ============================================================================
# Helper Functions
# ============================================================================


def process_prediction(result, top_k: int) -> tuple[bool, list[Prediction], str, float]:
    """
    예측 결과를 처리하여 Top-K 예측 리스트 반환

    Returns:
        (success, predictions, top1_class, top1_confidence)
    """
    probs = result.probs
    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    top1_class = model.names[top1_idx]

    # Top-K 인덱스와 신뢰도 추출
    top_k_indices = probs.top5  # Ultralytics는 top5까지 제공
    top_k_confs = probs.top5conf.tolist()

    # top_k가 5보다 크면 전체 정렬 필요
    if top_k > 5:
        all_probs = probs.data.tolist()
        sorted_indices = sorted(range(len(all_probs)), key=lambda i: all_probs[i], reverse=True)[:top_k]
        top_k_indices = sorted_indices
        top_k_confs = [all_probs[i] for i in sorted_indices]
    else:
        top_k_indices = top_k_indices[:top_k]
        top_k_confs = top_k_confs[:top_k]

    predictions = [
        Prediction(
            class_id=idx,
            class_name=model.names[idx],
            confidence=conf,
            confidence_pct=f"{conf * 100:.2f}%",
        )
        for idx, conf in zip(top_k_indices, top_k_confs)
    ]

    success = top1_conf >= CONFIDENCE_THRESHOLD
    return success, predictions, top1_class, top1_conf


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/")
async def root():
    return {"message": "Food Classification API", "docs": "/docs"}


@app.post("/predict", response_model=SinglePredictionResponse)
async def predict_food(
    file: UploadFile = File(..., description="분류할 음식 이미지 파일"),
    top_k: int = Query(default=5, ge=1, le=39, description="반환할 예측 결과 수 (1-39)"),
):
    """
    단일 이미지 음식 분류

    - **file**: 업로드할 이미지 파일 (jpg, png, webp 등)
    - **top_k**: 반환할 상위 예측 결과 수 (기본값: 5)

    Top-1 신뢰도가 50% 미만인 경우 400 에러를 반환합니다.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    # 이미지를 메모리에서 직접 처리
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 파일을 읽을 수 없습니다: {str(e)}")

    # 예측 수행
    results = model(image, verbose=False)
    success, predictions, top1_class, top1_conf = process_prediction(results[0], top_k)

    if not success:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "인식 실패: 신뢰도가 50% 미만입니다",
                "top1_class": top1_class,
                "top1_confidence": top1_conf,
                "predictions": [p.model_dump() for p in predictions],
            },
        )

    return SinglePredictionResponse(
        success=True,
        top1_class=top1_class,
        top1_confidence=top1_conf,
        predictions=predictions,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_food_batch(
    files: list[UploadFile] = File(..., description="분류할 음식 이미지 파일들"),
    top_k: int = Query(default=5, ge=1, le=39, description="반환할 예측 결과 수 (1-39)"),
):
    """
    배치 이미지 음식 분류

    - **files**: 업로드할 이미지 파일들
    - **top_k**: 반환할 상위 예측 결과 수 (기본값: 5)

    각 이미지별로 성공/실패를 개별 판정하여 부분 성공을 지원합니다.
    Top-1 신뢰도가 50% 미만인 이미지는 실패로 처리됩니다.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    # 이미지들을 메모리에서 로드
    images = []
    filenames = []
    load_errors = []

    for file in files:
        try:
            image_bytes = await file.read()
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            images.append(image)
            filenames.append(file.filename or "unknown")
        except Exception as e:
            load_errors.append((file.filename or "unknown", str(e)))

    # 배치 예측 수행
    results_list = []
    success_count = 0
    failure_count = len(load_errors)

    if images:
        batch_results = model(images, verbose=False)

        for filename, result in zip(filenames, batch_results):
            success, predictions, top1_class, top1_conf = process_prediction(result, top_k)

            if success:
                success_count += 1
                results_list.append(
                    ImagePredictionResult(
                        filename=filename,
                        success=True,
                        top1_class=top1_class,
                        top1_confidence=top1_conf,
                        predictions=predictions,
                    )
                )
            else:
                failure_count += 1
                results_list.append(
                    ImagePredictionResult(
                        filename=filename,
                        success=False,
                        message=f"인식 실패: 신뢰도 {top1_conf * 100:.2f}% (50% 미만)",
                        top1_class=top1_class,
                        top1_confidence=top1_conf,
                        predictions=predictions,
                    )
                )

    # 로드 실패한 이미지 추가
    for filename, error in load_errors:
        results_list.append(
            ImagePredictionResult(
                filename=filename,
                success=False,
                message=f"이미지 로드 실패: {error}",
            )
        )

    return BatchPredictionResponse(
        total=len(files),
        success_count=success_count,
        failure_count=failure_count,
        results=results_list,
    )


@app.get("/classes", response_model=ClassListResponse)
async def get_classes():
    """
    지원하는 음식 클래스 목록 조회

    39개 클래스 (한식 20개 + 국제음식 19개) 반환
    """
    if model is None:
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다")

    return ClassListResponse(
        total=len(model.names),
        classes=model.names,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    서버 및 모델 상태 확인
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        num_classes=len(model.names) if model else 0,
    )