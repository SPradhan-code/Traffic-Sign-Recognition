.PHONY: install data train evaluate tflite test lint api web cam docker
install:  ; pip install -r requirements-dev.txt && pip install -e .
data:     ; bash scripts/download_data.sh
train:    ; python -m tsr.train
evaluate: ; python -m tsr.evaluate
tflite:   ; python -m tsr.export_tflite
test:     ; pytest -q
lint:     ; black src apps tests && flake8 src apps tests --max-line-length 100
api:      ; uvicorn apps.api:app --reload
web:      ; streamlit run apps/web_app.py
cam:      ; python apps/realtime.py
docker:   ; docker build -t tsr-api . && docker run -p 8000:8000 tsr-api
