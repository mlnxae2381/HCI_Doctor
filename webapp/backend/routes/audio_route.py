"""
오디오 (기침음/호흡음) 분석 API
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from models.audio_model import audio_model

audio_bp = Blueprint('audio', __name__)

UPLOAD_FOLDER = Path('/home/students/cs/doctor/AI_Doctor/webapp/data/uploads')

@audio_bp.route('/analyze', methods=['POST'])
def analyze_audio():
    """오디오 분석 (기침음/호흡음)"""
    try:
        # 파일 검증
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400

        # 분석 유형 (cough or breathing)
        audio_type = request.form.get('type', 'cough')

        # 파일 저장
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / f"{audio_type}_{filename}"
        file.save(filepath)

        # 모델 예측
        result = audio_model.predict(str(filepath))
        result['audio_type'] = audio_type

        # 파일 삭제 (선택적)
        # os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@audio_bp.route('/info', methods=['GET'])
def audio_info():
    """오디오 모델 정보"""
    return jsonify({
        'model': 'HeAR (Health Acoustic Representations)',
        'accuracy': '96.74%',
        'classes': ['Normal', 'Abnormal'],
        'types': ['기침음 (Cough)', '호흡음 (Breathing)'],
        'description': '기침 소리와 호흡 소리를 분석하여 호흡기 질환 여부를 판단합니다',
        'status': 'available'
    })
