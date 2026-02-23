"""
커뮤니티 API
"""

from flask import Blueprint, request, jsonify
import database as db

community_bp = Blueprint('community', __name__)

# 게시글 관련
@community_bp.route('/posts', methods=['GET'])
def get_posts():
    """게시글 목록 조회"""
    try:
        category = request.args.get('category')
        author = request.args.get('author')
        limit = int(request.args.get('limit', 20))
        offset = int(request.args.get('offset', 0))

        posts = db.get_posts(category=category, author=author, limit=limit, offset=offset)
        return jsonify({'posts': posts, 'count': len(posts)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@community_bp.route('/posts', methods=['POST'])
def create_post():
    """게시글 생성"""
    try:
        data = request.get_json()

        title = data.get('title')
        content = data.get('content')
        author = data.get('author', 'Anonymous')
        category = data.get('category', 'general')

        if not title or not content:
            return jsonify({'error': '제목과 내용을 입력해주세요'}), 400

        post_id = db.create_post(title, content, author, category)

        return jsonify({'post_id': post_id, 'message': '게시글이 작성되었습니다'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@community_bp.route('/posts/<int:post_id>', methods=['GET'])
def get_post(post_id):
    """게시글 상세 조회"""
    try:
        post = db.get_post(post_id)

        if not post:
            return jsonify({'error': '게시글을 찾을 수 없습니다'}), 404

        # 댓글도 함께 조회
        comments = db.get_comments(post_id)
        post['comments'] = comments

        return jsonify(post)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@community_bp.route('/posts/<int:post_id>/like', methods=['POST'])
def like_post(post_id):
    """게시글 좋아요"""
    try:
        db.like_post(post_id)
        return jsonify({'message': '좋아요가 추가되었습니다'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 댓글 관련
@community_bp.route('/posts/<int:post_id>/comments', methods=['POST'])
def create_comment(post_id):
    """댓글 생성"""
    try:
        data = request.get_json()

        content = data.get('content')
        author = data.get('author', 'Anonymous')

        if not content:
            return jsonify({'error': '댓글 내용을 입력해주세요'}), 400

        comment_id = db.create_comment(post_id, content, author)

        return jsonify({'comment_id': comment_id, 'message': '댓글이 작성되었습니다'}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@community_bp.route('/posts/<int:post_id>/comments', methods=['GET'])
def get_comments(post_id):
    """댓글 목록 조회"""
    try:
        comments = db.get_comments(post_id)
        return jsonify({'comments': comments, 'count': len(comments)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@community_bp.route('/categories', methods=['GET'])
def get_categories():
    """카테고리 목록"""
    categories = [
        {'id': 'general', 'name': '자유게시판', 'icon': '💬'},
        {'id': 'health', 'name': '건강 정보', 'icon': '🏥'},
        {'id': 'question', 'name': '질문/답변', 'icon': '❓'},
        {'id': 'review', 'name': '진단 후기', 'icon': '⭐'}
    ]
    return jsonify({'categories': categories})
