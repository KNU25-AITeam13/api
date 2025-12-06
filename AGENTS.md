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

### API Testing
```bash
# Health check
curl http://localhost:8000/health

# Analyze food image
curl -X POST "http://localhost:8000/analyze" \
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
4. Reference Object Detection → scale calibration
    ↓
5. Volume Calculation → volume_ml, mass_g
    ↓
6. Nutrition Estimation → calories, protein, fat, carbs
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
- Supports `.env` file overrides

### Response Schema

**Pydantic Models** (`app/models.py`):
```python
AnalysisResponse:
  - food_name: str
  - confidence: float (0-1)
  - volume_ml: float
  - mass_g: float
  - nutrition: NutritionInfo
    - calories_kcal: float
    - protein_g: float
    - fat_g: float
    - carbs_g: float
```

### File Upload Flow

**Request Handling** (`app/main.py` + `app/utils.py`):
1. Validate file type (JPG/PNG only)
2. Save to temp file with `aiofiles` (async, chunked)
3. Run analysis pipeline
4. Clean up temp file in `finally` block (guaranteed cleanup)

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

### Density Calculation

Currently fixed at 1.0 g/ml (water density). Future CSV integration planned:
```python
# ai/pipeline.py
def _calculate_nutrition_dummy(self, food_name, mass_g):
    # TODO: Replace with actual CSV lookup
    # File: 전국통합식품영양성분정보_음식_표준데이터.csv
```

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
