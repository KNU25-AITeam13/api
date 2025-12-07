# AGENTS.md

This file provides guidance to AI Agents when working with code in this repository.

## Project Overview

FastAPI-based nutrition analysis API that integrates three AI models via git submodules:
1. **Depth Pro** (Apple): Monocular depth estimation
2. **Volume Assumption** (KNU AI Team): YOLO segmentation + volume calculation
3. **Food Classification** (KNU AI Team): YOLOv11 food classification

The pipeline: Image → Food Classification + Depth Map → Volume Calculation → Nutrition Estimation

## Essential Commands

### Development
```bash
# Install dependencies
uv sync

# Run development server (auto-downloads models on first run)
uv run fastapi dev app/main.py

# Run production server
uv run fastapi run app/main.py --port 80
```

### Git Submodules
```bash
# Clone with submodules
git clone --recurse-submodules <url>

# Update submodules after clone
git submodule update --init --recursive

# Update submodules to latest
git submodule update --remote --merge
```

### Docker
```bash
# Build image (includes Depth Pro checkpoint download)
docker build -t ai-api .

# Run container
docker run -p 80:80 ai-api

# Run with GPU
docker run --gpus all -p 80:80 ai-api
```

### Nutrition Database
```bash
# Rebuild nutrition database from CSV (if updated)
uv run python -m ai.nutrition.database_builder

# Test nutrition lookup module
uv run python -m ai.nutrition.nutrition_lookup
```

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Analyze food image (standard response)
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@food.jpg"

# Analyze with progress streaming (SSE)
curl -N -X POST "http://localhost:8000/analyze-stream" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@food.jpg"
```

## Architecture

### Pipeline Flow

**FoodAnalyzer Class** (`ai/pipeline.py`):
```
Input Image
    ↓
1. Food Classification (YOLOv11) → food_name, confidence
    ↓
2. Depth Pro → depth_map, focallength_px
    ↓
3. YOLO Segmentation → object masks (food, utensils, plates)
    ↓
4. Density Lookup (NutritionDatabase) → density_g_per_ml
    ↓
5. Reference Object Detection → scale calibration
    ↓
6. Volume Calculation → volume_ml, mass_g (using food-specific density)
    ↓
7. Nutrition Lookup (NutritionDatabase) → 14 nutrition fields
```

### Model Loading Strategy

**Lifespan Pattern** (`app/main.py`):
- All models loaded once at server startup via FastAPI lifespan context
- Models stored in `app.state.analyzer` for request reuse
- GPU memory cleared on shutdown

**Auto-Download Behavior**:
1. **Depth Pro** (`depth_pro.pt`, 1.8GB): Runtime auto-download if missing, or Docker build-time download
2. **YOLO Segmentation** (`yolo11x-seg.pt`, ~155MB): Ultralytics auto-downloads on first use
3. **Food Classification** (`best_mixed_food_v1.pt`, 25MB): Included in git repository
4. **Nutrition Database** (`nutrition.db`, ~50KB): Pre-built SQLite DB, included in repository

### Volume Calculation Accuracy Hierarchy

**Reference-Based Measurement** (`ai/volume_assumption/volume_test.py`):
1. **Best**: Utensil detection (spoon 18cm, fork 19cm, knife 22cm, chopsticks 21cm)
2. **Good**: Depth Pro focal length estimation (fallback mode)
3. **Acceptable**: Fixed 72° FOV (last resort)

The `volume_calculation_core` function uses `provided_f_px` parameter to pass Depth Pro's focal length for improved accuracy when no reference objects are detected.

### YOLO Segmentation Classes

**Object Categories** (`ai/pipeline.py`):
- `CUTLERY_LIKE`: spoon, fork, knife, chopsticks (reference objects)
- `PLATE_LIKE`: plate, bowl, cup, wine glass, tray (background)
- `FOOD_LIKE`: food, rice, noodles, pizza, etc. (volume calculation targets)

### Configuration System

**Settings** (`config/config.py`):
- Uses `pydantic-settings` for environment-based config
- Model paths: `yolo_seg_weights`, `food_model_weights`
- Nutrition database: `nutrition_db_path` (default: `ai/nutrition/nutrition.db`)
- Supports `.env` file overrides

### Concurrent Request Handling

**Thread-Safe Design** (`ai/pipeline.py`):
- Uses `asyncio.Lock` for safe concurrent request handling
- PyTorch model inference runs in dedicated `ThreadPoolExecutor` (max_workers=1)
- Requests are queued and processed sequentially to prevent race conditions
- Both `analyze()` and `analyze_stream()` are async methods with lock protection

**Implementation Details**:
```python
class FoodAnalyzer:
    def __init__(...):
        self._lock = asyncio.Lock()  # Protects model access
        self._executor = ThreadPoolExecutor(max_workers=1)  # Single worker for sequential processing

    async def analyze(self, image_path: str) -> dict:
        async with self._lock:  # Only one request processes at a time
            # Run blocking PyTorch inference in executor
            result = await loop.run_in_executor(self._executor, self._inference_sync, ...)
```

**Behavior**:
- Multiple concurrent requests are accepted by FastAPI
- Requests wait in queue when another request is processing
- GPU/CPU resources are protected from simultaneous access
- No memory conflicts or race conditions
- Executor cleanly shuts down with `analyzer.shutdown()` in lifespan

**Trade-offs**:
- Sequential processing (one request at a time) prevents GPU memory conflicts
- Higher throughput requires horizontal scaling (multiple instances) rather than parallel processing
- Simple implementation with predictable resource usage

### Response Schema

**Pydantic Models** (`app/models.py`):
```python
AnalysisResponse:
  - food_name: str
  - confidence: float (0-1)
  - volume_ml: float
  - mass_g: float
  - nutrition: NutritionInfo
    # Required fields (4)
    - calories_kcal: float
    - protein_g: float
    - fat_g: float
    - carbs_g: float
    # Optional fields (10) - from CSV database
    - water_g: Optional[float]
    - sugars_g: Optional[float]
    - dietary_fiber_g: Optional[float]
    - sodium_mg: Optional[float]
    - cholesterol_mg: Optional[float]
    - saturated_fat_g: Optional[float]
    - calcium_mg: Optional[float]
    - iron_mg: Optional[float]
    - vitamin_a_ug: Optional[float]
    - vitamin_c_mg: Optional[float]
```

Total: **14 nutrition fields** (4 required + 10 optional from Korean Food Nutrition DB)

**JSON Response Format (camelCase)**:
- All models use `alias_generator=to_camel` for automatic camelCase conversion
- Python attributes remain snake_case, but JSON output is camelCase
- Example response:
```json
{
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
```

### File Upload Flow

**Request Handling** (`app/main.py` + `app/utils.py`):
1. Validate file type (JPG/PNG only)
2. Save to temp file with `aiofiles` (async, chunked)
3. Run analysis pipeline
4. Clean up temp file in `finally` block (guaranteed cleanup)

### Progress Streaming (SSE)

**Endpoint**: `POST /analyze-stream`

Real-time progress updates via Server-Sent Events for frontend progress indicators.

**Implementation** (`ai/pipeline.py:103` + `app/main.py:178`):
```python
# FoodAnalyzer.analyze_stream() - Generator pattern
def analyze_stream(self, image_path: str):
    yield {"step": 1, "message": "음식 분류 중...", "status": "in_progress"}
    # ... run food classification

    yield {"step": 2, "message": "깊이 맵 생성 중...", "status": "in_progress"}
    # ... run depth estimation

    yield {"step": 3, "message": "객체 분할 중...", "status": "in_progress"}
    # ... run YOLO segmentation

    yield {"step": 4, "message": "부피 계산 및 영양소 분석 중...", "status": "in_progress"}
    # ... calculate volume and nutrition

    yield {"status": "completed", "result": {...}}  # Final result
```

**SSE Response Format**:
```
data: {"step": 1, "message": "음식 분류 중...", "status": "in_progress"}

data: {"step": 2, "message": "깊이 맵 생성 중...", "status": "in_progress"}

data: {"step": 3, "message": "객체 분할 중...", "status": "in_progress"}

data: {"step": 4, "message": "부피 계산 및 영양소 분석 중...", "status": "in_progress"}

data: {"status": "completed", "result": {"foodName": "비빔밥", ...}}
```

**Client Integration** (JavaScript example):
```javascript
const response = await fetch('/analyze-stream', {
  method: 'POST',
  body: formData
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const {done, value} = await reader.read();
  if (done) break;

  const text = decoder.decode(value);
  const lines = text.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));

      if (data.status === 'in_progress') {
        updateProgressBar(data.step, 4);  // Show progress (1-4/4)
      } else if (data.status === 'completed') {
        displayResult(data.result);
      }
    }
  }
}
```

**Key Features**:
- Non-blocking: Client receives updates as each pipeline stage completes
- Error handling: Errors streamed as `{"status": "error", "message": "..."}`
- Automatic cleanup: Temp files cleaned up in `finally` block
- camelCase output: All JSON keys converted via `dict_to_camel_case()` helper

## Important Implementation Details

### Depth Pro Checkpoint Handling

The model expects checkpoint at specific path. Implementation uses absolute path:
```python
# ai/pipeline.py
checkpoint_path = self._ensure_depth_pro_checkpoint()  # Returns Path object
config = DepthProConfig(checkpoint_uri=str(checkpoint_path))  # Absolute path
```

Never use relative path `./checkpoints/depth_pro.pt` as it fails depending on CWD.

### YOLO Model Reuse

The `yolo_inference` function in `volume_test.py` creates new YOLO instance each call. Our `FoodAnalyzer._run_yolo_segmentation` reuses pre-loaded `self.yolo_model` for efficiency:
```python
# DON'T: model = YOLO(weights) on every inference
# DO: self.yolo_model = YOLO(weights) in __init__
results = self.yolo_model(image_path, imgsz=640, conf=0.15, verbose=False)
```

### Nutrition Database Integration

**Korean Food Nutrition Database** (`ai/nutrition/`):

The system uses the official Korean food nutrition database (전국통합식품영양성분정보_음식_표준데이터.csv) integrated with YOLOv11's 39 food classes.

**Database Structure**:
- **Source CSV**: 14,582 food items from Korean Ministry of Food and Drug Safety
- **Filtered SQLite DB**: 34 food classes (39 YOLO classes, 5 missing in CSV)
- **Averaging Strategy**: Multiple food variants averaged per YOLO class
- **Location**: `ai/nutrition/nutrition.db` (~50KB)

**Density Calculation (3-Tier Strategy)**:
```python
# ai/pipeline.py
density = self.nutrition_db.get_density(food_name)  # 0.19 ~ 1.41 g/ml

# Tier 1: CSV-based calculation (best accuracy)
#   - For 100ml basis: density = 100g / serving_weight
#   - For 100g basis: water_ratio * 1.0 + (1 - water_ratio) * 1.5

# Tier 2: Category-based defaults (fallback)
#   - Rice dishes: 0.6 g/ml
#   - Soups/stews: 1.0 g/ml
#   - Noodles: 0.8 g/ml
#   - Fried foods: 0.7 g/ml

# Tier 3: Global default (final fallback)
#   - 1.0 g/ml (water density)
```

**Nutrition Lookup**:
```python
# ai/pipeline.py
nutrition = self._calculate_nutrition(food_name, mass_g)
# Returns 14 fields: 4 required + 10 optional from CSV
```

**Database Rebuild**:
```bash
# If CSV is updated, rebuild the database
python -m ai.nutrition.database_builder
```

**Key Files**:
- `ai/nutrition/food_name_mapping.py`: Maps 39 YOLO classes to CSV food names
- `ai/nutrition/database_builder.py`: Builds SQLite from CSV (averages variants)
- `ai/nutrition/nutrition_lookup.py`: `NutritionDatabase` class for queries
- `ai/nutrition/nutrition.db`: SQLite database (34 food classes)

### PyTorch Installation

`pyproject.toml` uses platform-specific PyTorch sources:
```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }
```

For GPU: Remove these overrides and use default PyPI (includes CUDA).

### Submodule Integration

All submodules are in `ai/` directory and added to `sys.path`:
```python
# ai/pipeline.py
sys.path.append(str(current_dir / 'depth_pro' / 'src'))
sys.path.append(str(current_dir / 'volume_assumption'))
sys.path.append(str(current_dir / 'food_classification' / 'src'))
```

This allows direct imports: `import depth_pro`, `from volume_test import ...`, `from predict import ...`

## Docker Build Notes

**Dockerfile Strategy**:
- Base: `python:3.13-slim`
- System deps: `libgl1`, `libglib2.0-0` (OpenCV), `wget`, `gcc`, `g++`
- Checkpoint download in build step (cached in image)
- Runtime download skipped if checkpoint exists

**Build Optimization**:
```dockerfile
RUN if [ ! -f /app/ai/depth_pro/checkpoints/depth_pro.pt ]; then
    wget -q --show-progress ... ;
fi
```

Conditional download prevents re-downloading if checkpoint already in source.

## Testing Workflow

1. Start server: `uv run fastapi dev app/main.py`
2. Wait for "All models loaded successfully!" (first run downloads models)
3. Open Swagger UI: `http://localhost:8000/docs`
4. Test `/analyze` endpoint with food image
5. Verify response format matches `AnalysisResponse` schema

## Submodule Documentation

Each AI submodule has its own README:
- `ai/depth_pro/README.md`: Depth estimation details
- `ai/volume_assumption/README.md`: Volume calculation algorithm
- `ai/food_classification/README.md`: Model training and inference (see also `AGENTS.md`)

Refer to submodule docs for model-specific configuration and troubleshooting.
