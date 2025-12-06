"""
FastAPI 유틸리티 함수
"""

import os
import tempfile
from pathlib import Path
from fastapi import UploadFile, HTTPException, status
import aiofiles


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    업로드된 파일을 임시 디렉토리에 저장합니다.

    Args:
        upload_file: FastAPI UploadFile 객체

    Returns:
        저장된 파일의 절대 경로

    Raises:
        Exception: 파일 저장 실패
    """
    # 파일 확장자 추출
    ext = Path(upload_file.filename).suffix.lower() if upload_file.filename else ''
    if not ext:
        ext = '.jpg'  # 기본값

    # 임시 파일 생성
    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix='food_')

    try:
        # 비동기로 파일 저장
        async with aiofiles.open(temp_path, 'wb') as f:
            # 청크 단위로 읽어서 저장 (메모리 효율)
            chunk_size = 1024 * 1024  # 1MB
            while True:
                chunk = await upload_file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)

        os.close(temp_fd)
        return temp_path

    except Exception as e:
        # 에러 시 임시 파일 삭제
        os.close(temp_fd)
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise e


def cleanup_temp_file(file_path: str) -> None:
    """
    임시 파일을 안전하게 삭제합니다.

    Args:
        file_path: 삭제할 파일 경로
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        # 로깅만 하고 에러는 무시 (cleanup 실패가 응답에 영향 X)
        print(f"Warning: Failed to cleanup temp file {file_path}: {e}")


def validate_image_file(upload_file: UploadFile) -> None:
    """
    업로드된 파일이 유효한 이미지인지 검증합니다.

    Args:
        upload_file: 검증할 파일

    Raises:
        HTTPException: 유효하지 않은 파일
    """
    # 파일 존재 확인
    if not upload_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded"
        )

    # Content-Type 검증
    allowed_types = {"image/jpeg", "image/png", "image/jpg"}
    if upload_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {upload_file.content_type}. "
                   f"Allowed types: {', '.join(allowed_types)}"
        )

    # 파일 확장자 검증
    if upload_file.filename:
        ext = Path(upload_file.filename).suffix.lower()
        allowed_exts = {".jpg", ".jpeg", ".png"}
        if ext and ext not in allowed_exts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension: {ext}. "
                       f"Allowed extensions: {', '.join(allowed_exts)}"
            )
