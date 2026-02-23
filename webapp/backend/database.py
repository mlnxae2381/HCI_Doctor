"""
커뮤니티 기능을 위한 데이터베이스 설정
"""

import sqlite3
from datetime import datetime
import json

DB_PATH = '/home/students/cs/doctor/AI_Doctor/webapp/data/community.db'

def init_db():
    """데이터베이스 초기화"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 게시글 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            category TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            views INTEGER DEFAULT 0,
            likes INTEGER DEFAULT 0
        )
    ''')

    # 댓글 테이블
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            author TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts (id)
        )
    ''')

    conn.commit()
    conn.close()

def get_db():
    """데이터베이스 연결"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# 게시글 CRUD
def create_post(title, content, author, category):
    """게시글 생성"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO posts (title, content, author, category) VALUES (?, ?, ?, ?)',
        (title, content, author, category)
    )
    conn.commit()
    post_id = cursor.lastrowid
    conn.close()
    return post_id

def get_posts(category=None, author=None, limit=20, offset=0):
    """게시글 목록 조회"""
    conn = get_db()
    cursor = conn.cursor()

    conditions = []
    params = []
    if category:
        conditions.append('category = ?')
        params.append(category)
    if author:
        conditions.append('author = ?')
        params.append(author)

    where = ('WHERE ' + ' AND '.join(conditions)) if conditions else ''
    params += [limit, offset]
    cursor.execute(
        f'SELECT * FROM posts {where} ORDER BY created_at DESC LIMIT ? OFFSET ?',
        params
    )

    posts = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return posts

def get_post(post_id):
    """게시글 상세 조회"""
    conn = get_db()
    cursor = conn.cursor()

    # 조회수 증가
    cursor.execute('UPDATE posts SET views = views + 1 WHERE id = ?', (post_id,))

    cursor.execute('SELECT * FROM posts WHERE id = ?', (post_id,))
    post = cursor.fetchone()

    conn.commit()
    conn.close()

    return dict(post) if post else None

def like_post(post_id):
    """게시글 좋아요"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE posts SET likes = likes + 1 WHERE id = ?', (post_id,))
    conn.commit()
    conn.close()

# 댓글 CRUD
def create_comment(post_id, content, author):
    """댓글 생성"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO comments (post_id, content, author) VALUES (?, ?, ?)',
        (post_id, content, author)
    )
    conn.commit()
    comment_id = cursor.lastrowid
    conn.close()
    return comment_id

def get_comments(post_id):
    """댓글 목록 조회"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM comments WHERE post_id = ? ORDER BY created_at ASC',
        (post_id,)
    )
    comments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return comments

# 초기화
init_db()
