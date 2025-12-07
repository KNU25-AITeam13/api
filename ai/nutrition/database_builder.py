"""
영양 데이터베이스 빌더

전국통합식품영양성분정보 CSV를 읽어서 YOLOv11 39개 클래스에 매칭되는
음식만 추출하고, 변형들의 평균값을 계산하여 SQLite 데이터베이스를 생성합니다.

Usage:
    python -m ai.nutrition.database_builder
"""

import sqlite3
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from ai.nutrition.food_name_mapping import (
    YOLO_TO_CSV_MAPPING,
    DENSITY_DEFAULTS,
    GLOBAL_DEFAULT_DENSITY,
)


# CSV 파일 경로
CSV_PATH = Path(__file__).parent.parent.parent / "전국통합식품영양성분정보_음식_표준데이터.csv"
DB_PATH = Path(__file__).parent / "nutrition.db"


# CSV 컬럼명 매핑 (한글 → 내부 키)
COLUMN_MAPPING = {
    '식품코드': 'food_code',
    '식품명': 'food_name',
    '영양성분함량기준량': 'reference_basis',
    '식품중량': 'serving_weight',
    '에너지(kcal)': 'energy_kcal',
    '단백질(g)': 'protein_g',
    '지방(g)': 'fat_g',
    '탄수화물(g)': 'carbs_g',
    '수분(g)': 'water_g',
    '당류(g)': 'sugars_g',
    '식이섬유(g)': 'dietary_fiber_g',
    '나트륨(mg)': 'sodium_mg',
    '콜레스테롤(mg)': 'cholesterol_mg',
    '포화지방산(g)': 'saturated_fat_g',
    '칼슘(mg)': 'calcium_mg',
    '철(mg)': 'iron_mg',
    '비타민 A(μg RAE)': 'vitamin_a_ug_rae',
    '비타민 C(mg)': 'vitamin_c_mg',
}


def load_csv() -> pd.DataFrame:
    """CSV 파일을 로드합니다."""
    print(f"[1/5] CSV 파일 로드 중: {CSV_PATH}")

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}\n"
            f"프로젝트 루트에 '전국통합식품영양성분정보_음식_표준데이터.csv' 파일이 있는지 확인하세요."
        )

    df = pd.read_csv(CSV_PATH, encoding='utf-8')
    print(f"   총 {len(df)} 개 레코드 로드됨")
    return df


def extract_numeric(value: str) -> Optional[float]:
    """문자열에서 숫자 추출 (예: '230.3', '170ml' → 230.3, 170.0)"""
    if pd.isna(value):
        return None

    # 숫자와 소수점만 추출
    match = re.search(r'(\d+\.?\d*)', str(value))
    if match:
        return float(match.group(1))
    return None


def calculate_density(row: pd.Series) -> float:
    """
    3-Tier 밀도 계산 전략

    Tier 1: CSV 데이터 기반 계산
    Tier 2: 카테고리별 기본값 (YOLO 클래스 기반)
    Tier 3: 글로벌 기본값 (1.0 g/ml)
    """
    reference_basis = row.get('reference_basis', '')
    yolo_class = row.get('yolo_class', '')

    # Tier 1: '100ml' 기준인 경우 serving_weight에서 밀도 계산
    if reference_basis == '100ml':
        serving_weight = row.get('serving_weight', '')
        numeric_weight = extract_numeric(serving_weight)
        if numeric_weight and numeric_weight > 0:
            # 100ml 기준 영양정보가 numeric_weight(g)에 해당
            density = 100.0 / numeric_weight
            return round(density, 2)

    # Tier 1b: '100g' 기준인 경우 수분 함량으로 추정
    if reference_basis == '100g':
        water_g = row.get('water_g', None)
        if pd.notna(water_g) and water_g > 0:
            water_ratio = water_g / 100.0
            # 물: 1.0 g/ml, 고형물: 1.5 g/ml로 가정
            density = water_ratio * 1.0 + (1 - water_ratio) * 1.5
            return round(density, 2)

    # Tier 2: 카테고리별 기본값
    if yolo_class in DENSITY_DEFAULTS:
        return DENSITY_DEFAULTS[yolo_class]

    # Tier 3: 글로벌 기본값
    return GLOBAL_DEFAULT_DENSITY


def calculate_average_nutrition(
    df: pd.DataFrame,
    yolo_class: str,
    food_names: List[str]
) -> Optional[Dict]:
    """
    여러 변형의 영양정보 평균값을 계산합니다.

    Args:
        df: CSV DataFrame
        yolo_class: YOLO 클래스명
        food_names: CSV에서 찾을 음식명 리스트

    Returns:
        평균 영양정보 딕셔너리 (없으면 None)
    """
    # CSV에서 매칭되는 행 찾기 (식품명 정확히 일치)
    matched_rows = df[df['식품명'].isin(food_names)]

    if len(matched_rows) == 0:
        print(f"   경고: '{yolo_class}' 에 매칭되는 CSV 데이터 없음: {food_names}")
        return None

    print(f"   '{yolo_class}': {len(matched_rows)}개 변형 찾음 → 평균 계산")

    # 컬럼명 변환
    matched_rows = matched_rows.rename(columns=COLUMN_MAPPING)

    # 평균 계산을 위한 데이터 준비
    nutrition_data = {
        'yolo_class': yolo_class,
        'variant_count': len(matched_rows),
    }

    # 첫 번째 매칭 행에서 메타데이터 가져오기
    first_row = matched_rows.iloc[0]
    nutrition_data['food_code'] = first_row.get('food_code', '')
    nutrition_data['food_name'] = first_row.get('food_name', '')
    nutrition_data['reference_basis'] = first_row.get('reference_basis', '100g')

    # 영양소 평균 계산 (NULL 값 제외)
    numeric_columns = [
        'energy_kcal', 'protein_g', 'fat_g', 'carbs_g',
        'water_g', 'sugars_g', 'dietary_fiber_g',
        'sodium_mg', 'cholesterol_mg', 'saturated_fat_g',
        'calcium_mg', 'iron_mg',
        'vitamin_a_ug_rae', 'vitamin_c_mg',
    ]

    for col in numeric_columns:
        if col in matched_rows.columns:
            # NaN이 아닌 값들의 평균
            values = matched_rows[col].dropna()
            if len(values) > 0:
                nutrition_data[col] = round(values.mean(), 2)
            else:
                nutrition_data[col] = None if col not in ['energy_kcal', 'protein_g'] else 0.0
        else:
            nutrition_data[col] = None

    # 밀도 계산 (평균 레코드에 대해)
    # 각 변형의 밀도를 계산한 후 평균
    densities = []
    for _, row in matched_rows.iterrows():
        row_with_class = row.copy()
        row_with_class['yolo_class'] = yolo_class
        density = calculate_density(row_with_class)
        densities.append(density)

    nutrition_data['density_g_per_ml'] = round(np.mean(densities), 2)

    return nutrition_data


def create_database(nutrition_records: List[Dict]):
    """SQLite 데이터베이스를 생성하고 데이터를 삽입합니다."""
    print(f"\n[4/5] SQLite 데이터베이스 생성 중: {DB_PATH}")

    # 기존 DB 삭제
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 테이블 생성
    cursor.execute("""
        CREATE TABLE nutrition_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            yolo_class TEXT NOT NULL,
            food_code TEXT,
            food_name TEXT NOT NULL,
            reference_basis TEXT NOT NULL,

            energy_kcal REAL NOT NULL,
            protein_g REAL NOT NULL,
            fat_g REAL,
            carbs_g REAL,

            water_g REAL,
            sugars_g REAL,
            dietary_fiber_g REAL,
            sodium_mg REAL,
            cholesterol_mg REAL,
            saturated_fat_g REAL,
            calcium_mg REAL,
            iron_mg REAL,
            vitamin_a_ug_rae REAL,
            vitamin_c_mg REAL,

            density_g_per_ml REAL NOT NULL,
            variant_count INTEGER DEFAULT 1,

            UNIQUE(yolo_class)
        )
    """)

    cursor.execute("CREATE INDEX idx_yolo_class ON nutrition_data(yolo_class)")

    # 데이터 삽입
    for record in nutrition_records:
        cursor.execute("""
            INSERT INTO nutrition_data (
                yolo_class, food_code, food_name, reference_basis,
                energy_kcal, protein_g, fat_g, carbs_g,
                water_g, sugars_g, dietary_fiber_g,
                sodium_mg, cholesterol_mg, saturated_fat_g,
                calcium_mg, iron_mg,
                vitamin_a_ug_rae, vitamin_c_mg,
                density_g_per_ml, variant_count
            ) VALUES (
                :yolo_class, :food_code, :food_name, :reference_basis,
                :energy_kcal, :protein_g, :fat_g, :carbs_g,
                :water_g, :sugars_g, :dietary_fiber_g,
                :sodium_mg, :cholesterol_mg, :saturated_fat_g,
                :calcium_mg, :iron_mg,
                :vitamin_a_ug_rae, :vitamin_c_mg,
                :density_g_per_ml, :variant_count
            )
        """, record)

    conn.commit()
    conn.close()

    print(f"   데이터베이스 생성 완료: {len(nutrition_records)}개 레코드")


def validate_database():
    """생성된 데이터베이스를 검증합니다."""
    print(f"\n[5/5] 데이터베이스 검증 중...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 레코드 수 확인
    cursor.execute("SELECT COUNT(*) FROM nutrition_data")
    count = cursor.fetchone()[0]
    print(f"   총 레코드 수: {count}")

    # 샘플 조회
    cursor.execute("""
        SELECT yolo_class, food_name, energy_kcal, density_g_per_ml, variant_count
        FROM nutrition_data
        LIMIT 5
    """)

    print("\n   샘플 데이터:")
    for row in cursor.fetchall():
        yolo_class, food_name, kcal, density, variant_count = row
        print(f"      {yolo_class}: {food_name} | "
              f"{kcal}kcal/100g | 밀도={density}g/ml | 변형={variant_count}개")

    # 밀도 범위 확인
    cursor.execute("SELECT MIN(density_g_per_ml), MAX(density_g_per_ml) FROM nutrition_data")
    min_density, max_density = cursor.fetchone()
    print(f"\n   밀도 범위: {min_density} ~ {max_density} g/ml")

    conn.close()
    print("\n✅ 데이터베이스 구축 완료!")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("영양 데이터베이스 빌더")
    print("=" * 60)

    # 1. CSV 로드
    df = load_csv()

    # 2. 39개 YOLO 클래스 처리
    print(f"\n[2/5] YOLO 클래스 매핑 중 (총 {len(YOLO_TO_CSV_MAPPING)}개 클래스)")

    nutrition_records = []
    skipped_classes = []

    for yolo_class, csv_food_names in YOLO_TO_CSV_MAPPING.items():
        if csv_food_names is None:
            # CSV에 없는 음식 (더미 데이터 사용)
            print(f"   '{yolo_class}': CSV에 없음 → 스킵 (더미 데이터 사용)")
            skipped_classes.append(yolo_class)
            continue

        # 평균 영양정보 계산
        avg_nutrition = calculate_average_nutrition(df, yolo_class, csv_food_names)

        if avg_nutrition:
            nutrition_records.append(avg_nutrition)
        else:
            skipped_classes.append(yolo_class)

    print(f"\n[3/5] 처리 결과:")
    print(f"   DB에 추가: {len(nutrition_records)}개 클래스")
    print(f"   스킵: {len(skipped_classes)}개 클래스")
    if skipped_classes:
        print(f"   스킵된 클래스: {', '.join(skipped_classes)}")

    # 3. 데이터베이스 생성
    create_database(nutrition_records)

    # 4. 검증
    validate_database()


if __name__ == "__main__":
    main()
