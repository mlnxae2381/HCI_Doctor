// HCI DocTor - JavaScript (with Router)

const API_BASE = '/api';
let currentService = null;

// ========== ROUTER SETUP ==========

// Register routes
router.addRoute('/', renderHomePage);
router.addRoute('/community', renderCommunityPage);
router.addRoute('/community/feed', renderCommunityFeedPage);
router.addRoute('/community/post/:id', renderPostDetailPage);
router.addRoute('/login', renderLoginPage);
router.addRoute('/signup', renderSignupPage);
router.addRoute('/about', renderAboutPage);
router.addRoute('/analysis/ct', renderCTAnalysisPage);
router.addRoute('/analysis/xray', renderXrayAnalysisPage);
router.addRoute('/analysis/acoustic', renderAcousticAnalysisPage);

// 404 handler
router.setNotFound(render404Page);

// Initialize router on page load
window.addEventListener('DOMContentLoaded', () => {
    router.init();
    fetchHealthMetrics();
});

// Legacy page navigation support (for backward compatibility)
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const href = link.getAttribute('href');
        if (href) {
            router.navigateTo(href);
        }
    });
});

// 분석 페이지로 이동
function openAnalysis(service) {
    if (service === 'ct') {
        router.navigateTo('/analysis/ct');
    } else if (service === 'xray') {
        router.navigateTo('/analysis/xray');
    } else if (service === 'breathing' || service === 'cough') {
        router.navigateTo('/analysis/acoustic');
    }
}

function closeModal() {
    const modal = document.getElementById('analysis-modal');
    modal.classList.remove('active');
    currentService = null;
    resetModal();
}

function resetModal() {
    document.getElementById('upload-area').style.display = 'block';
    document.getElementById('loading-area').style.display = 'none';
    document.getElementById('result-area').style.display = 'none';
    document.getElementById('result-area').innerHTML = '';
    document.getElementById('file-input').value = '';
}

// 파일 선택 처리
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    analyzeFile(file);
}

// 파일 분석
async function analyzeFile(file) {
    // UI 업데이트
    document.getElementById('upload-area').style.display = 'none';
    document.getElementById('loading-area').style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    let endpoint = '';
    if (currentService === 'ct') {
        endpoint = `${API_BASE}/ct/analyze`;
    } else if (currentService === 'xray') {
        endpoint = `${API_BASE}/xray/analyze`;
    } else if (currentService === 'cough' || currentService === 'breathing') {
        endpoint = `${API_BASE}/audio/analyze`;
        formData.append('type', currentService);
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        // 로딩 숨기기
        document.getElementById('loading-area').style.display = 'none';

        // 결과 표시
        displayResult(result);

    } catch (error) {
        document.getElementById('loading-area').style.display = 'none';
        alert('오류가 발생했습니다: ' + error.message);
        resetModal();
    }
}

// 결과 표시
function displayResult(result) {
    const resultArea = document.getElementById('result-area');
    resultArea.style.display = 'block';

    // 에러 처리
    if (result.error) {
        resultArea.innerHTML = `
            <div class="result-card">
                <h4 style="color: var(--warning-color);">
                    <i class="fas fa-exclamation-triangle"></i> ${result.error}
                </h4>
                <p>${result.message || ''}</p>
                ${result.status === 'training' ? `
                    <p>Training progress: ${result.progress}</p>
                    <p>Estimated completion: ${result.eta}</p>
                ` : ''}
            </div>
        `;
        return;
    }

    // 정상 결과 표시
    let html = `
        <div class="result-card">
            <div class="result-label">
                <i class="fas fa-clipboard-check"></i> Diagnosis Result
            </div>
            <h3 style="color: ${getPredictionColor(result.prediction)}; margin-bottom: 1rem;">
                ${formatPrediction(result.prediction)}
            </h3>
    `;

    // 신뢰도 표시
    if (result.confidence) {
        html += '<div class="result-confidence">';
        html += '<h4 style="margin-bottom: 1rem;">Confidence</h4>';

        for (const [label, score] of Object.entries(result.confidence)) {
            const percentage = (score * 100).toFixed(1);
            html += `
                <div class="confidence-bar">
                    <span style="min-width: 100px;">${formatPrediction(label)}</span>
                    <div class="bar">
                        <div class="bar-fill" style="width: ${percentage}%"></div>
                    </div>
                    <span style="font-weight: 600;">${percentage}%</span>
                </div>
            `;
        }
        html += '</div>';
    }

    // 설명 표시
    if (result.explanation) {
        html += `
            <div class="result-explanation">
                <h4 style="margin-bottom: 0.5rem;">
                    <i class="fas fa-info-circle"></i> Detailed Analysis
                </h4>
                <p>${result.explanation}</p>
            </div>
        `;
    }

    // 모델 정보
    html += `
        <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid var(--light);">
            <p style="color: var(--gray); font-size: 0.9rem;">
                <i class="fas fa-brain"></i> Model: ${result.model}
                ${result.accuracy ? ` | Accuracy: ${result.accuracy}` : ''}
            </p>
        </div>
    `;

    html += '</div>';

    // 다시 분석 버튼
    html += `
        <button class="btn btn-primary" onclick="resetModal()" style="width: 100%; margin-top: 1rem;">
            <i class="fas fa-redo"></i> Analyze Again
        </button>
    `;

    resultArea.innerHTML = html;
}

function formatPrediction(pred) {
    const labels = {
        'normal': 'Normal',
        'abnormal': 'Abnormal',
        'benign': 'Benign Tumor',
        'malignant': 'Malignant Tumor'
    };
    return labels[pred.toLowerCase()] || pred;
}

function getPredictionColor(pred) {
    const colors = {
        'normal': 'var(--success-color)',
        'abnormal': 'var(--warning-color)',
        'benign': 'var(--warning-color)',
        'malignant': 'var(--danger-color)'
    };
    return colors[pred.toLowerCase()] || 'var(--dark)';
}

// ========== 커뮤니티 기능 ==========

// 게시글 데이터 레지스트리 (북마크용 참조 저장)
let postRegistry = {};

let currentCategory = 'all';

// 커뮤니티 카테고리 데이터
const communityCategories = [
    {
        id: 'general',
        colorBar: 'teal',
        author: '커뮤니티',
        avatar: '💬',
        title: '자유게시판',
        subtitle: 'Free Discussion',
        description: 'AI 의료 진단 경험과 건강에 관한 자유로운 이야기를 나눠보세요.',
        tags: ['자유토론', '일상', '건강'],
        buttons: [
            { text: '게시글 보기', type: 'primary' }
        ]
    },
    {
        id: 'health',
        colorBar: 'teal',
        author: '건강 정보',
        avatar: '🏥',
        badge: { text: '✓', color: 'teal' },
        title: '건강 정보',
        subtitle: 'Medical Knowledge',
        description: '검증된 의료 정보와 AI 진단 결과 해석 가이드를 공유합니다.',
        tags: ['AI 검증', '의료 정보', '가이드'],
        buttons: [
            { text: '게시글 보기', type: 'primary' }
        ]
    },
    {
        id: 'question',
        colorBar: 'gray',
        author: '질문/답변',
        avatar: '❓',
        title: '질문/답변',
        subtitle: 'Q&A',
        description: '진단 결과 해석이나 건강 관련 궁금한 점을 전문가에게 질문해보세요.',
        tags: ['질문', '전문가 답변'],
        buttons: [
            { text: '질문하기', type: 'primary' }
        ]
    },
    {
        id: 'review',
        colorBar: 'blue',
        author: '진단 후기',
        avatar: '⭐',
        badge: { text: '✓', color: 'teal' },
        title: '진단 후기',
        subtitle: 'AI Diagnosis Reviews',
        description: 'HCI DocTor AI 진단 서비스를 이용한 실제 경험과 의사 소견을 공유합니다.',
        tags: ['AI 검증', '실제 후기', '진단'],
        buttons: [
            { text: '후기 보기', type: 'primary' }
        ]
    },
    {
        id: 'ct',
        colorBar: 'blue',
        author: 'CT 분석',
        avatar: '🫁',
        badge: { text: '✓', color: 'green' },
        title: 'CT 분석 케이스',
        subtitle: 'Lung CT Cases',
        description: '폐 CT 분석 결과 공유 및 전문가 피드백. MedGemma AI 판독 사례 모음.',
        tags: ['CT', '폐암', 'MedGemma'],
        buttons: [
            { text: '케이스 보기', type: 'primary' }
        ]
    },
    {
        id: 'audio',
        colorBar: 'gray',
        author: '음향 분석',
        avatar: '🎤',
        title: '음향 분석 케이스',
        subtitle: 'Audio Analysis Cases',
        description: '기침음·호흡음 분석 결과 공유. HeAR 모델 기반 호흡기 질환 스크리닝 사례.',
        tags: ['기침음', '호흡음', 'HeAR'],
        buttons: [
            { text: '케이스 보기', type: 'primary' }
        ]
    }
];

// 커뮤니티 카드 렌더링
function renderCommunityCategories() {
    const grid = document.getElementById('community-grid');
    if (!grid) return;

    grid.innerHTML = communityCategories.map(cat => `
        <div class="community-card" onclick="router.navigateTo('/community/feed?category=${cat.id}')">
            <div class="community-card-color-bar ${cat.colorBar}"></div>
            <div class="community-card-body">
                <div class="community-card-header">
                    <div class="community-card-avatar">${cat.avatar}</div>
                    <div class="community-card-author">${cat.author}</div>
                    ${cat.badge ? `<div class="community-card-badge ${cat.badge.color}">${cat.badge.text}</div>` : ''}
                </div>

                <h3 class="community-card-title">${cat.title}</h3>
                ${cat.subtitle ? `<div class="community-card-subtitle">${cat.subtitle}</div>` : ''}
                ${cat.description ? `<p class="community-card-description">${cat.description}</p>` : ''}

                ${cat.image ? `
                    <div style="width: 100%; height: 120px; background: linear-gradient(135deg, #E0F2FE, #BAE6FD); border-radius: 8px; margin: 12px 0; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-user-md" style="font-size: 48px; color: #0284C7; opacity: 0.3;"></i>
                    </div>
                ` : ''}

                ${cat.tags ? `
                    <div class="community-card-tags">
                        ${cat.tags.map(tag => `<span class="community-card-tag">${tag}</span>`).join('')}
                    </div>
                ` : ''}

                ${cat.buttons ? `
                    <div class="community-card-actions">
                        ${cat.buttons.map(btn => `
                            <button class="community-card-button ${btn.type}" onclick="event.stopPropagation()">
                                ${btn.text}
                            </button>
                        `).join('')}
                    </div>
                ` : ''}

                ${cat.redButton ? `
                    <button style="width: 100%; margin-top: 12px; padding: 8px; background: #EF4444; color: white; border: none; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;" onclick="event.stopPropagation()">
                        ${cat.redButton}
                    </button>
                ` : ''}

                ${cat.dots ? `
                    <div class="community-card-footer">
                        <div class="community-card-avatar-small"></div>
                        <span class="community-card-footer-text">Hover Ipela E theric Riekero Mey risoismind goon</span>
                        <div class="community-card-dots">
                            ${cat.dots.map(active => `<div class="community-card-dot ${active ? 'active' : ''}"></div>`).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `).join('');
}

// 게시글 로드 (레거시 호환)
async function loadPosts() {
    filterCommunity('all');
}

// ========== 커뮤니티 카테고리 필터링 ==========

let currentCommunityCategory = 'all';

/**
 * 카테고리별 게시글 로드 및 렌더링
 */
async function filterCommunity(category) {
    currentCommunityCategory = category;
    setSidebarActive('sidebar-all');

    // 탭 active 상태 업데이트
    document.querySelectorAll('#category-nav .nav-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === category);
    });

    const grid = document.getElementById('community-grid');
    if (!grid) return;

    // 피드 모드 전환 + 로딩 표시
    grid.className = 'community-grid community-posts-mode';
    grid.id = 'community-grid';
    grid.innerHTML = `
        <div class="community-loading">
            <i class="fas fa-spinner fa-spin"></i>
            <span>Loading...</span>
        </div>
    `;

    // API에서 게시글 fetch
    let posts = [];
    try {
        const url = category === 'all'
            ? '/api/community/posts?limit=30'
            : `/api/community/posts?category=${category}&limit=30`;
        const response = await fetch(url);
        if (response.ok) {
            const data = await response.json();
            posts = (data.posts || []).map(p => ({
                id: p.id,
                author: p.author || 'Anonymous',
                avatar: null,
                time: formatDate(p.created_at),
                category: getCategoryName(p.category),
                title: p.title,
                content: p.content,
                tags: [],
                likes: p.likes || 0,
                replies: 0,
                image: null
            }));
        }
    } catch (e) { /* 서버 미연결 시 빈 상태 표시 */ }

    // 게시글을 레지스트리에 저장
    posts.forEach(p => { postRegistry[p.id] = p; });

    // 빈 상태
    if (posts.length === 0) {
        grid.innerHTML = `
            <div class="community-empty">
                <i class="fas fa-file-alt"></i>
                <p>No posts yet</p>
                <button class="btn-write-post" style="margin-top:8px;" onclick="openPostForm()">
                    <i class="fas fa-plus"></i> Write the first post
                </button>
            </div>
        `;
        return;
    }

    // renderPostCard는 pages.js에서 정의
    if (typeof renderPostCard === 'function') {
        grid.innerHTML = posts.map(post => renderPostCard(post)).join('');
    }
}

// ========== 사이드바 네비게이션 ==========

function setSidebarActive(id) {
    document.querySelectorAll('.sidebar-link').forEach(el => el.classList.remove('active'));
    const target = document.getElementById(id);
    if (target) target.classList.add('active');
}

function showAllPosts() {
    setSidebarActive('sidebar-all');
    // 카테고리 탭 복원
    const nav = document.getElementById('category-nav');
    if (nav) nav.style.display = '';
    filterCommunity(currentCommunityCategory || 'all');
}

async function showMyActivity() {
    setSidebarActive('sidebar-my');
    const username = localStorage.getItem('doctor_username');
    const grid = document.getElementById('community-grid');
    if (!grid) return;

    // 카테고리 탭 숨기기
    const nav = document.getElementById('category-nav');
    if (nav) nav.style.display = 'none';

    grid.className = 'community-grid community-posts-mode';
    grid.innerHTML = `<div class="community-loading"><i class="fas fa-spinner fa-spin"></i><span>Loading...</span></div>`;

    if (!username) {
        grid.innerHTML = `
            <div class="community-empty">
                <i class="fas fa-user-circle"></i>
                <p>Set a nickname to view your posts</p>
                <button class="btn-write-post" style="margin-top:8px;" onclick="openSettingsModal()">
                    <i class="fas fa-cog"></i> Set Nickname
                </button>
            </div>`;
        return;
    }

    let posts = [];
    try {
        const response = await fetch(`/api/community/posts?author=${encodeURIComponent(username)}&limit=50`);
        if (response.ok) {
            const data = await response.json();
            posts = (data.posts || []).map(p => ({
                id: p.id,
                author: p.author || 'Anonymous',
                avatar: null,
                time: formatDate(p.created_at),
                category: getCategoryName(p.category),
                title: p.title,
                content: p.content,
                tags: [],
                likes: p.likes || 0,
                replies: 0,
                image: null
            }));
        }
    } catch (e) {}

    posts.forEach(p => { postRegistry[p.id] = p; });

    if (posts.length === 0) {
        grid.innerHTML = `
            <div class="community-empty">
                <i class="fas fa-file-alt"></i>
                <p>No posts by <strong>${escapeHtml(username)}</strong> yet</p>
                <button class="btn-write-post" style="margin-top:8px;" onclick="openPostForm()">
                    <i class="fas fa-plus"></i> Write a Post
                </button>
            </div>`;
        return;
    }

    if (typeof renderPostCard === 'function') {
        grid.innerHTML = `
            <div class="my-activity-header">
                <i class="fas fa-chart-line"></i> Posts by <strong>${escapeHtml(username)}</strong> (${posts.length})
            </div>
            ${posts.map(post => renderPostCard(post)).join('')}`;
    }
}

// ========== 북마크 기능 ==========

function getSavedPosts() {
    try {
        return JSON.parse(localStorage.getItem('doctor_saved_posts') || '[]');
    } catch (e) { return []; }
}

function isPostSaved(postId) {
    return getSavedPosts().includes(String(postId));
}

function bookmarkPost(postId, buttonEl) {
    postId = String(postId);
    const saved = getSavedPosts();
    const idx = saved.indexOf(postId);
    if (idx === -1) {
        saved.push(postId);
        if (buttonEl) {
            buttonEl.classList.add('bookmarked');
            buttonEl.title = 'Remove bookmark';
        }
    } else {
        saved.splice(idx, 1);
        if (buttonEl) {
            buttonEl.classList.remove('bookmarked');
            buttonEl.title = 'Save';
        }
    }
    localStorage.setItem('doctor_saved_posts', JSON.stringify(saved));
}

async function showSavedPosts() {
    setSidebarActive('sidebar-saved');
    const grid = document.getElementById('community-grid');
    if (!grid) return;

    const nav = document.getElementById('category-nav');
    if (nav) nav.style.display = 'none';

    grid.className = 'community-grid community-posts-mode';
    grid.innerHTML = `<div class="community-loading"><i class="fas fa-spinner fa-spin"></i><span>Loading...</span></div>`;

    const savedIds = getSavedPosts();
    if (savedIds.length === 0) {
        grid.innerHTML = `
            <div class="community-empty">
                <i class="fas fa-bookmark"></i>
                <p>No saved posts</p>
                <p style="font-size:0.85rem;color:var(--gray);">Click the <i class="fas fa-bookmark"></i> on any post to save it</p>
            </div>`;
        return;
    }

    // 레지스트리에서 먼저 조회, 없으면 API fetch
    let posts = [];
    for (const id of savedIds) {
        if (postRegistry[id]) {
            posts.push(postRegistry[id]);
        } else {
            try {
                const res = await fetch(`/api/community/posts/${id}`);
                if (res.ok) {
                    const p = await res.json();
                    const mapped = {
                        id: p.id,
                        author: p.author || 'Anonymous',
                        avatar: null,
                        time: formatDate(p.created_at),
                        category: getCategoryName(p.category),
                        title: p.title,
                        content: p.content,
                        tags: [],
                        likes: p.likes || 0,
                        replies: 0,
                        image: null
                    };
                    postRegistry[p.id] = mapped;
                    posts.push(mapped);
                }
            } catch (e) {}
        }
    }

    if (posts.length === 0) {
        grid.innerHTML = `
            <div class="community-empty">
                <i class="fas fa-bookmark"></i>
                <p>Could not load saved posts</p>
            </div>`;
        return;
    }

    if (typeof renderPostCard === 'function') {
        grid.innerHTML = `
            <div class="my-activity-header">
                <i class="fas fa-bookmark"></i> Saved Posts (${posts.length})
            </div>
            ${posts.map(post => renderPostCard(post)).join('')}`;
    }
}

function clearSavedPosts() {
    if (confirm('Clear all saved posts?')) {
        localStorage.removeItem('doctor_saved_posts');
        alert('Saved posts have been cleared.');
    }
}

// ========== 설정 모달 ==========

function openSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (!modal) return;
    const usernameInput = document.getElementById('settings-username');
    if (usernameInput) {
        usernameInput.value = localStorage.getItem('doctor_username') || '';
    }
    modal.classList.add('active');
}

function closeSettingsModal() {
    const modal = document.getElementById('settings-modal');
    if (modal) modal.classList.remove('active');
}

function saveSettings() {
    const usernameInput = document.getElementById('settings-username');
    const username = usernameInput?.value?.trim();
    if (username) {
        localStorage.setItem('doctor_username', username);
        // 게시글 작성 폼 author 필드도 동기화
        const authorEl = document.getElementById('post-author');
        if (authorEl && !authorEl.value) authorEl.value = username;
    } else {
        localStorage.removeItem('doctor_username');
    }
    closeSettingsModal();
    alert('Settings saved.');
}

/**
 * 댓글 작성 (API 연동)
 */
async function submitComment(postId) {
    const contentEl = document.getElementById(`comment-input-${postId}`);
    const authorEl = document.getElementById(`comment-author-${postId}`);
    const content = contentEl?.value?.trim();
    const author = authorEl?.value?.trim() || 'Anonymous';

    if (!content) {
        alert('Please enter a comment.');
        return;
    }

    try {
        const response = await fetch(`/api/community/posts/${postId}/comments`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content, author })
        });

        if (response.ok) {
            const data = await response.json();
            const commentsList = document.querySelector('.comments-list');
            const commentsTitle = document.querySelector('.comments-title');

            // 새 댓글 추가
            const newComment = {
                id: data.comment_id,
                author,
                avatar: null,
                time: 'just now',
                text: content,
                likes: 0
            };
            if (commentsList && typeof renderCommentItem === 'function') {
                commentsList.insertAdjacentHTML('afterbegin', renderCommentItem(newComment));
            }
            // 카운트 업데이트
            if (commentsTitle) {
                const current = parseInt(commentsTitle.textContent) || 0;
                commentsTitle.textContent = `${current + 1} Comments`;
            }
            if (contentEl) contentEl.value = '';
        } else {
            alert('Failed to post comment.');
        }
    } catch (e) {
        alert('Could not connect to server.');
    }
}

/**
 * 게시글 좋아요 (API 연동)
 */
async function likePost(postId, buttonEl) {
    try {
        const response = await fetch(`/api/community/posts/${postId}/like`, { method: 'POST' });
        if (response.ok) {
            const data = await response.json();
            const countSpan = buttonEl.querySelector('span');
            if (countSpan) countSpan.textContent = data.likes ?? (parseInt(countSpan.textContent) + 1);
            buttonEl.classList.add('liked');
            buttonEl.style.color = 'var(--primary-color)';
        }
    } catch (e) {
        // MOCK 게시글은 로컬 카운트만 증가
        const countSpan = buttonEl.querySelector('span');
        if (countSpan) countSpan.textContent = parseInt(countSpan.textContent || 0) + 1;
        buttonEl.classList.add('liked');
        buttonEl.style.color = 'var(--primary-color)';
    }
}

function getCategoryName(category) {
    const names = {
        'general': '💬 Free Board',
        'health': '🏥 Health Info',
        'question': '❓ Q&A',
        'review': '⭐ Diagnosis Reviews'
    };
    return names[category] || category;
}

function formatDate(dateStr) {
    const date = new Date(dateStr);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;

    return date.toLocaleDateString('en-US');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 게시글 보기
async function viewPost(postId) {
    // 커뮤니티 피드 상세 페이지로 이동
    router.navigateTo(`/community/feed?id=${postId}`);
}

// 게시글 작성 폼
function openPostForm() {
    const modal = document.getElementById('post-modal');
    modal.classList.add('active');
    // 설정된 닉네임 자동 입력
    const authorEl = document.getElementById('post-author');
    if (authorEl && !authorEl.value) {
        const saved = localStorage.getItem('doctor_username');
        if (saved) authorEl.value = saved;
    }
}

function closePostForm() {
    document.getElementById('post-modal').classList.remove('active');
    document.getElementById('post-form').reset();
}

// 게시글 제출
document.getElementById('post-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const title = document.getElementById('post-title').value.trim();
    const content = document.getElementById('post-content').value.trim();
    const author = document.getElementById('post-author').value.trim() || 'Anonymous';
    const category = document.getElementById('post-category').value;

    if (!title || !content) {
        alert('Please enter a title and content.');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/community/posts`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ title, content, author, category })
        });

        const result = await response.json();

        if (response.ok) {
            alert('Post submitted!');
            closePostForm();
            filterCommunity(category);
        } else {
            alert('Error: ' + result.error);
        }

    } catch (error) {
        alert('An error occurred while submitting the post.');
    }
});

// 모달 외부 클릭시 닫기
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
});

// ========== 로그인/회원가입 ==========

function openLoginModal() {
    document.getElementById('login-modal').classList.add('active');
    document.getElementById('register-modal').classList.remove('active');
}

function closeLoginModal() {
    document.getElementById('login-modal').classList.remove('active');
}

function openRegisterModal() {
    document.getElementById('register-modal').classList.add('active');
    document.getElementById('login-modal').classList.remove('active');
}

function closeRegisterModal() {
    document.getElementById('register-modal').classList.remove('active');
}

// 로그인 폼 제출
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    // TODO: 실제 로그인 API 연동
    console.log('Login attempt:', { email, password });
    alert('Login requires backend integration.');
});

// 회원가입 폼 제출
document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = document.getElementById('register-name').value;
    const email = document.getElementById('register-email').value;
    const type = document.getElementById('register-type').value;
    const password = document.getElementById('register-password').value;

    // TODO: 실제 회원가입 API 연동
    console.log('Register attempt:', { name, email, type, password });
    alert('Registration requires backend integration.');
});

// ========== 유틸리티 함수 ==========

/**
 * 준비 중 알림
 */
function showComingSoon(feature) {
    alert(`${feature} is coming soon!`);
}

/**
 * /api/health 에서 모델 메트릭을 가져와 홈페이지 카드 업데이트
 */
async function fetchHealthMetrics() {
    try {
        const response = await fetch('/api/health');
        if (!response.ok) return;
        const data = await response.json();

        // CT 정확도 추출 (예: "MedGemma LoRA (87.35% acc)")
        const ctAcc = data.models?.ct?.match(/([\d.]+%)/)?.[1];
        if (ctAcc) {
            const el = document.getElementById('metric-accuracy');
            if (el) el.textContent = ctAcc;
            const cardEl = document.getElementById('card-acc-ct');
            if (cardEl) cardEl.textContent = `${ctAcc} ACC`;
        }

        // X-ray 정확도 (테스트 완료 후 업데이트됨)
        const xrayAcc = data.models?.xray?.match(/([\d.]+%)/)?.[1];
        if (xrayAcc) {
            const cardEl = document.getElementById('card-acc-xray');
            if (cardEl) cardEl.textContent = `${xrayAcc} ACC`;
        }

        // 음향 정확도
        const audioAcc = data.models?.cough?.match(/([\d.]+%)/)?.[1];
        if (audioAcc) {
            const el = document.getElementById('metric-audio');
            if (el) el.textContent = audioAcc;
        }

        // 활성 모델 수
        const modelCount = Object.keys(data.models || {}).length;
        const modelsEl = document.getElementById('metric-models');
        if (modelsEl) modelsEl.textContent = `0${modelCount}`;

    } catch (e) {
        // 서버 미연결 시 기본값 유지
    }
}

/**
 * 커뮤니티 네비게이션 탭 업데이트
 */
function updateCommunityNavTabs() {
    const currentPath = window.location.pathname;

    // 상단 네비게이션 탭
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.classList.remove('active');
        const href = tab.getAttribute('href');
        if (href && (href === currentPath || (href.startsWith('/analysis') && currentPath.startsWith('/analysis')))) {
            tab.classList.add('active');
        }
    });

    // 사이드바 링크
    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.classList.remove('active');
        const href = link.getAttribute('href');
        if (href && href === currentPath) {
            link.classList.add('active');
        }
    });
}

// 페이지 로드 시 네비게이션 업데이트
window.addEventListener('DOMContentLoaded', () => {
    updateCommunityNavTabs();
});

// 라우터 네비게이션 후 탭 업데이트
const originalNavigateTo = router.navigateTo;
router.navigateTo = function(path) {
    originalNavigateTo.call(this, path);
    setTimeout(updateCommunityNavTabs, 100);
};

// 초기 로드
console.log('HCI DocTor Web App Loaded');
