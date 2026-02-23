"""
HCI DocTor - AI 의료 진단 보조 웹앱 백엔드
Port: 3003
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging

# 라우트 임포트
from routes.ct_route import ct_bp
from routes.xray_route import xray_bp
from routes.audio_route import audio_bp
from routes.community_route import community_bp

# 설정
app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# 업로드 설정
UPLOAD_FOLDER = '/home/students/cs/doctor/AI_Doctor/webapp/data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'wav', 'mp3', 'ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 블루프린트 등록
app.register_blueprint(ct_bp, url_prefix='/api/ct')
app.register_blueprint(xray_bp, url_prefix='/api/xray')
app.register_blueprint(audio_bp, url_prefix='/api/audio')
app.register_blueprint(community_bp, url_prefix='/api/community')

@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('../frontend', 'index.html')

# 정적 파일 서빙 (CSS, JS, 이미지 등)
@app.route('/css/<path:filename>')
def serve_css(filename):
    """CSS 파일 서빙"""
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    """JavaScript 파일 서빙"""
    return send_from_directory('../frontend/js', filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename):
    """Asset 파일 서빙"""
    return send_from_directory('../frontend/assets', filename)

@app.route('/health')
def health():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'service': 'HCI DocTor',
        'version': '1.0.0',
        'models': {
            'ct': 'MedGemma LoRA (87.35% acc)',
            'xray': 'MedGemma LoRA (67.21% acc)',
            'cough': 'HeAR (96.74% acc)',
            'breathing': 'HeAR (96.74% acc)'
        }
    })

@app.errorhandler(413)
def request_entity_too_large(error):
    """파일 크기 초과 에러"""
    return jsonify({'error': '파일 크기가 너무 큽니다 (최대 50MB)'}), 413

@app.errorhandler(404)
def not_found(error):
    """404 에러 - SPA 라우팅 지원"""
    # API 요청인 경우 JSON 에러 반환
    if request.path.startswith('/api/'):
        return jsonify({'error': '요청한 리소스를 찾을 수 없습니다'}), 404
    # SPA 라우팅을 위해 index.html 반환
    return send_from_directory('../frontend', 'index.html')

@app.errorhandler(500)
def internal_error(error):
    """500 에러"""
    logger.error(f"Internal error: {error}")
    return jsonify({'error': '서버 내부 오류가 발생했습니다'}), 500

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("HCI DocTor - AI Medical Diagnosis Assistant")
    logger.info("=" * 60)
    logger.info("Starting server on http://0.0.0.0:3003")
    logger.info("Models:")
    logger.info("  - CT Analysis: MedGemma LoRA (87.35% accuracy)")
    logger.info("  - X-ray Analysis: MedGemma LoRA (67.21% accuracy)")
    logger.info("  - Cough Analysis: HeAR (96.74% accuracy)")
    logger.info("  - Breathing Analysis: HeAR (96.74% accuracy)")
    logger.info("=" * 60)

    app.run(host='0.0.0.0', port=3003, debug=True)
