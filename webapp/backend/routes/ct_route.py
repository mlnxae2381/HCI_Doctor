"""
CT 이미지 분석 API
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from models.ct_model import ct_model

ct_bp = Blueprint('ct', __name__)

UPLOAD_FOLDER = Path('/home/students/cs/doctor/AI_Doctor/webapp/data/uploads')

@ct_bp.route('/analyze', methods=['POST'])
def analyze_ct():
    """CT 이미지 분석"""
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / f"ct_{filename}"
        file.save(filepath)

        # 모델 예측
        result = ct_model.predict(str(filepath))

        # 파일 삭제 (선택적)
        # os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@ct_bp.route('/info', methods=['GET'])
def ct_info():
    """CT 모델 정보"""
    return jsonify({
        'model': 'MedGemma LoRA',
        'accuracy': '87.35%',
        'classes': ['Normal', 'Benign', 'Malignant'],
        'description': 'CT 이미지에서 정상, 양성 종양, 악성 종양을 분류합니다',
        'status': 'available'
    })
