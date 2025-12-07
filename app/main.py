from contextlib import asynccontextmanager
import sys
from pathlib import Path
import json
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import torch

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent))

from ai.pipeline import FoodAnalyzer
from app.models import AnalysisResponse, ErrorResponse, NutritionInfo, ProgressUpdate
from app.utils import save_upload_file, cleanup_temp_file, validate_image_file, dict_to_camel_case
from config.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 로직"""
    # Startup
    print("=" * 60)
    print("Starting AI API Server...")
    print("=" * 60)

    try:
        # 모델 로드
        analyzer = FoodAnalyzer(
            yolo_weights_path=settings.yolo_seg_weights,
            food_model_path=settings.food_model_weights
        )

        app.state.analyzer = analyzer
        print("\n" + "=" * 60)
        print("All models loaded successfully!")
        print("Server is ready to accept requests")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"Failed to load models: {e}")
        raise

    yield  # 앱 실행

    # Shutdown
    print("\n" + "=" * 60)
    print("Shutting down AI API Server...")

    if hasattr(app.state, 'analyzer'):
        del app.state.analyzer

    # GPU 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

    print("Shutdown complete!")
    print("=" * 60 + "\n")


app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)


@app.get("/", tags=["Health"])
async def read_root():
    """서버 상태 확인"""
    return {
        "message": "Welcome to the AI API",
        "status": "running",
        "version": settings.api_version
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """서버 헬스 체크"""
    return {
        "status": "healthy",
        "models_loaded": hasattr(app.state, 'analyzer')
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Analysis"]
)
async def analyze_food(
    request: Request,
    file: UploadFile = File(..., description="음식 이미지 파일 (JPG, PNG)")
) -> AnalysisResponse:
    """
    음식 이미지를 분석하여 영양 성분을 추정합니다.

    - **file**: 음식 사진 (JPG 또는 PNG)

    반환값:
    - **food_name**: 음식 이름
    - **confidence**: 분류 신뢰도 (0~1)
    - **volume_ml**: 부피 (ml)
    - **mass_g**: 질량 (g)
    - **nutrition**: 영양소 정보
    """
    # 파일 검증
    validate_image_file(file)

    temp_path = None

    try:
        # 임시 파일 저장
        temp_path = await save_upload_file(file)

        # 이미지 유효성 검증
        try:
            img = Image.open(temp_path)
            img.verify()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid or corrupted image: {str(e)}"
            )

        # FoodAnalyzer 실행
        analyzer = request.app.state.analyzer

        try:
            result = analyzer.analyze(temp_path)
        except Exception as e:
            import traceback
            traceback.print_exc()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model inference failed: {str(e)}"
            )

        # 응답 구성
        response = AnalysisResponse(
            food_name=result['food_name'],
            confidence=result['confidence'],
            volume_ml=result['volume_ml'],
            mass_g=result['mass_g'],
            nutrition=NutritionInfo(**result['nutrition'])
        )

        return response

    except HTTPException:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )

    finally:
        # 임시 파일 정리
        if temp_path:
            cleanup_temp_file(temp_path)


@app.post(
    "/analyze-stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"}
    },
    tags=["Analysis"]
)
async def analyze_food_stream(
    request: Request,
    file: UploadFile = File(..., description="음식 이미지 파일 (JPG, PNG)")
):
    """
    음식 이미지를 분석하면서 진행 상황을 Server-Sent Events로 스트리밍합니다.

    - **file**: 음식 사진 (JPG 또는 PNG)

    반환값 (SSE 스트림):
    - **진행 상황**: step, message, status (in_progress)
    - **최종 결과**: status (completed), result (AnalysisResponse)
    - **에러**: status (error), message
    """
    # 파일 검증
    validate_image_file(file)

    temp_path = None

    async def event_generator():
        nonlocal temp_path

        try:
            # 임시 파일 저장
            temp_path = await save_upload_file(file)

            # 이미지 유효성 검증
            try:
                img = Image.open(temp_path)
                img.verify()
            except Exception as e:
                error_data = dict_to_camel_case({"status": "error", "message": f"Invalid or corrupted image: {str(e)}"})
                yield f"data: {json.dumps(error_data)}\n\n"
                return

            # FoodAnalyzer 실행 (스트리밍 모드)
            analyzer = request.app.state.analyzer

            try:
                for progress in analyzer.analyze_stream(temp_path):
                    # SSE 형식으로 데이터 전송 (camelCase 변환)
                    camel_progress = dict_to_camel_case(progress)
                    yield f"data: {json.dumps(camel_progress)}\n\n"
                    # 비동기 처리를 위한 짧은 대기
                    await asyncio.sleep(0)

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_data = dict_to_camel_case({"status": "error", "message": f"Model inference failed: {str(e)}"})
                yield f"data: {json.dumps(error_data)}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_data = dict_to_camel_case({"status": "error", "message": f"Unexpected error: {str(e)}"})
            yield f"data: {json.dumps(error_data)}\n\n"

        finally:
            # 임시 파일 정리
            if temp_path:
                cleanup_temp_file(temp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Nginx 버퍼링 비활성화
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """예상치 못한 에러를 일관된 형식으로 반환"""
    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error"
        }
    )