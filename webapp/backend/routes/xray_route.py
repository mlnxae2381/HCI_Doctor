"""
X-ray 이미지 분석 API
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from models.xray_model import xray_model

xray_bp = Blueprint('xray', __name__)

UPLOAD_FOLDER = Path('/home/students/cs/doctor/AI_Doctor/webapp/data/uploads')

@xray_bp.route('/analyze', methods=['POST'])
def analyze_xray():
    """X-ray 이미지 분석"""
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / f"xray_{filename}"
        file.save(filepath)

        # 모델 예측
        result = xray_model.predict(str(filepath))

        # 파일 삭제 (선택적)
        # os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@xray_bp.route('/info', methods=['GET'])
def xray_info():
    """X-ray 모델 정보"""
    is_trained = xray_model.is_trained

    if is_trained:
        return jsonify({
            'model': 'MedGemma LoRA',
            'accuracy': 'TBD (testing)',
            'classes': ['Normal', 'Abnormal'],
            'description': '흉부 X-ray 이미지에서 정상/비정상을 분류합니다',
            'status': 'available',
            'training_info': {
                'checkpoint': 'step-3600',
                'val_loss': 0.0929,
                'epochs': '2.6/3.0'
            }
        })
    else:
        return jsonify({
            'model': 'MedGemma LoRA',
            'status': 'not_available',
            'error': 'Model not found',
            'classes': ['Normal', 'Abnormal'],
            'description': '흉부 X-ray 이미지에서 정상/비정상을 분류합니다',
            'message': 'xray_lora_adapter 디렉토리를 확인하세요'
        })
