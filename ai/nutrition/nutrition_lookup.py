"""
영양정보 조회 모듈

SQLite 데이터베이스에서 음식의 영양정보와 밀도를 조회하고,
실제 질량에 맞게 스케일링합니다.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional

from ai.nutrition.food_name_mapping import DENSITY_DEFAULTS, GLOBAL_DEFAULT_DENSITY


DB_PATH = Path(__file__).parent / "nutrition.db"


class NutritionDatabase:
    """영양정보 데이터베이스 조회 클래스"""

    def __init__(self, db_path: Optional[Path] = None):
        """
        영양정보 데이터베이스를 로드합니다.

        Args:
            db_path: 데이터베이스 파일 경로 (None이면 기본 경로 사용)
        """
        self.db_path = db_path or DB_PATH

        if not self.db_path.exists():
            print(f"[경고] 영양 DB 파일이 없습니다: {self.db_path}")
            print(f"       더미 데이터를 사용합니다. DB를 생성하려면:")
            print(f"       python -m ai.nutrition.database_builder")
            self.db_available = False
            self._cache = {}
            return

        # 데이터베이스를 메모리에 캐싱
        self.db_available = True
        self._cache = self._load_all_data()
        print(f"[NutritionDatabase] {len(self._cache)}개 음식 데이터 로드 완료")

    def _load_all_data(self) -> Dict[str, Dict]:
        """데이터베이스의 모든 데이터를 메모리에 로드합니다."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM nutrition_data")
        rows = cursor.fetchall()

        cache = {}
        for row in rows:
            yolo_class = row['yolo_class']
            cache[yolo_class] = dict(row)

        conn.close()
        return cache

    def get_density(self, food_name: str) -> float:
        """
        음식의 밀도를 조회합니다.

        Args:
            food_name: YOLO 클래스명

        Returns:
            밀도 (g/ml)
        """
        # DB에서 조회
        if self.db_available and food_name in self._cache:
            return self._cache[food_name]['density_g_per_ml']

        # Fallback: 카테고리별 기본값
        if food_name in DENSITY_DEFAULTS:
            return DENSITY_DEFAULTS[food_name]

        # Final fallback: 글로벌 기본값
        return GLOBAL_DEFAULT_DENSITY

    def get_nutrition(self, food_name: str, mass_g: float) -> Dict:
        """
        음식의 영양정보를 조회하고 질량에 맞게 스케일링합니다.

        Args:
            food_name: YOLO 클래스명
            mass_g: 음식의 실제 질량 (g)

        Returns:
            영양정보 딕셔너리 (14개 필드)
        """
        # DB에서 조회
        if self.db_available and food_name in self._cache:
            row = self._cache[food_name]
            return self._scale_nutrition(row, mass_g)

        # Fallback: 더미 데이터 반환
        return self._get_dummy_nutrition(mass_g)

    def _scale_nutrition(self, row: Dict, mass_g: float) -> Dict:
        """
        100g 기준 영양정보를 실제 질량으로 스케일링합니다.

        Args:
            row: DB에서 조회한 레코드
            mass_g: 실제 질량 (g)

        Returns:
            스케일링된 영양정보
        """
        scale = mass_g / 100.0

        def scale_value(value):
            """값을 스케일링 (None은 None 유지)"""
            if value is None:
                return None
            return round(value * scale, 2)

        return {
            # 필수 필드
            'calories_kcal': scale_value(row['energy_kcal']),
            'protein_g': scale_value(row['protein_g']),
            'fat_g': scale_value(row.get('fat_g', 0.0)),
            'carbs_g': scale_value(row.get('carbs_g', 0.0)),

            # 선택 필드 (10개)
            'water_g': scale_value(row.get('water_g')),
            'sugars_g': scale_value(row.get('sugars_g')),
            'dietary_fiber_g': scale_value(row.get('dietary_fiber_g')),
            'sodium_mg': scale_value(row.get('sodium_mg')),
            'cholesterol_mg': scale_value(row.get('cholesterol_mg')),
            'saturated_fat_g': scale_value(row.get('saturated_fat_g')),
            'calcium_mg': scale_value(row.get('calcium_mg')),
            'iron_mg': scale_value(row.get('iron_mg')),
            'vitamin_a_ug': scale_value(row.get('vitamin_a_ug_rae')),
            'vitamin_c_mg': scale_value(row.get('vitamin_c_mg')),
        }

    def _get_dummy_nutrition(self, mass_g: float) -> Dict:
        """
        더미 영양정보를 반환합니다 (DB에 없는 음식용).

        Args:
            mass_g: 음식의 질량 (g)

        Returns:
            더미 영양정보
        """
        # 100g당 평균 영양소 (대략적인 추정치)
        base_nutrition = {
            'calories_kcal': 150,
            'protein_g': 8,
            'fat_g': 5,
            'carbs_g': 25,
        }

        # 질량에 비례하여 조정
        scale = mass_g / 100.0

        return {
            # 필수 필드
            'calories_kcal': round(base_nutrition['calories_kcal'] * scale, 2),
            'protein_g': round(base_nutrition['protein_g'] * scale, 2),
            'fat_g': round(base_nutrition['fat_g'] * scale, 2),
            'carbs_g': round(base_nutrition['carbs_g'] * scale, 2),

            # 선택 필드 (모두 None)
            'water_g': None,
            'sugars_g': None,
            'dietary_fiber_g': None,
            'sodium_mg': None,
            'cholesterol_mg': None,
            'saturated_fat_g': None,
            'calcium_mg': None,
            'iron_mg': None,
            'vitamin_a_ug': None,
            'vitamin_c_mg': None,
        }


# 테스트 코드 (직접 실행 시)
if __name__ == "__main__":
    print("=" * 60)
    print("영양정보 조회 모듈 테스트")
    print("=" * 60)

    db = NutritionDatabase()

    # 테스트 음식들
    test_foods = [
        ('비빔밥', 200.0),
        ('라자냐', 150.0),  # DB에 없음 (더미 데이터)
        ('피자', 100.0),
    ]

    for food_name, mass_g in test_foods:
        print(f"\n[{food_name}] {mass_g}g:")

        # 밀도 조회
        density = db.get_density(food_name)
        print(f"  밀도: {density} g/ml")

        # 영양정보 조회
        nutrition = db.get_nutrition(food_name, mass_g)
        print(f"  열량: {nutrition['calories_kcal']} kcal")
        print(f"  단백질: {nutrition['protein_g']} g")
        print(f"  지방: {nutrition['fat_g']} g")
        print(f"  탄수화물: {nutrition['carbs_g']} g")
        print(f"  나트륨: {nutrition.get('sodium_mg', 'N/A')} mg")

    print("\n✅ 테스트 완료!")
