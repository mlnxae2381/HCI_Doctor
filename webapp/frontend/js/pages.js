// HCI DocTor - Page Components

/**
 * Page 렌더링 헬퍼
 */
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === window.location.pathname) {
            link.classList.add('active');
        }
    });
}

/**
 * 동적 페이지를 위한 공통 레이아웃 생성
 * @param {string} title - 페이지 제목
 * @param {string} content - 페이지 본문 HTML
 * @param {boolean} showBackButton - 뒤로가기 버튼 표시 여부
 * @param {string} backUrl - 뒤로가기 URL (기본: 브라우저 히스토리)
 */
function createPageLayout(title, content, showBackButton = true, backUrl = null) {
    const backButtonHtml = showBackButton ? `
        <button class="btn btn-secondary" onclick="${backUrl ? `router.navigateTo('${backUrl}')` : 'history.back()'}" style="margin-bottom: 1.5rem; display: inline-flex; align-items: center; gap: 8px;">
            <i class="fas fa-arrow-left"></i> Back
        </button>
    ` : '';

    return `
        <div class="page active" style="animation: fadeIn 0.3s ease;">
            <div class="container" style="padding-top: 2rem;">
                ${backButtonHtml}
                <div style="background: var(--white); padding: 3rem; border-radius: 16px; box-shadow: var(--shadow-md);">
                    <h2 style="font-size: 32px; font-weight: 900; color: var(--gray-900); margin-bottom: 2rem;">
                        ${title}
                    </h2>
                    ${content}
                </div>
            </div>
        </div>
    `;
}

/**
 * 동적 페이지 렌더링 (헤더 유지)
 */
function renderDynamicPage(html) {
    // 기존 페이지 모두 숨기기
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));

    // 동적 페이지 컨테이너 찾기 또는 생성
    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    // 페이지 렌더링
    dynamicContainer.innerHTML = html;
}

/**
 * CT 풀스크린 모드 진입/해제
 */
function enterCTFullscreenMode() {
    document.body.classList.add('ct-fullscreen-mode');
}

function exitCTFullscreenMode() {
    document.body.classList.remove('ct-fullscreen-mode');
}

/**
 * 홈 페이지 (메인 대시보드)
 */
function renderHomePage() {
    exitCTFullscreenMode();
    showPage('home');
}

/**
 * 커뮤니티 페이지 — 카테고리 탭 + 게시글 피드
 */
function renderCommunityPage() {
    exitCTFullscreenMode();
    showPage('community');
    // 카테고리 탭 복원
    const nav = document.getElementById('category-nav');
    if (nav) nav.style.display = '';
    // 카테고리 탭 초기화 + 전체 게시글 로드
    if (typeof filterCommunity === 'function') {
        filterCommunity(typeof currentCommunityCategory !== 'undefined' ? currentCommunityCategory : 'all');
    }
    if (typeof updateCommunityNavTabs === 'function') {
        setTimeout(updateCommunityNavTabs, 50);
    }
}

// ── Mock Community Data (module-level) ─────────────────────────
const MOCK_POSTS = [
    {
        id: 1,
        author: 'Sarah Miller',
        avatar: 'https://i.pravatar.cc/150?img=1',
        verified: true,
        aiFeedback: true,
        time: '2 hours ago',
        category: 'Diagnosis Review',
        title: "Comparing HCI DocTor's skin scan results with my dermatologist's clinic biopsy",
        content: "I used the HCI DocTor AI scanner on a suspicious mole last week. The AI suggested it was likely benign but recommended a professional checkup due to irregular borders. Just came back from the dermatologist, and the results were identical...",
        fullContent: `I've been a patient advocate and health tech enthusiast for the past three years, and I recently put HCI DocTor's skin analysis to the test in a real clinical scenario.\n\nThree weeks ago, I noticed a new mole on my shoulder that had irregular borders — one of the classic ABCDE warning signs. I first used the HCI DocTor AI scanner to get an initial read.\n\n<strong>What the AI found:</strong> The system analyzed the mole's ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution) and classified it as "likely benign with moderate confidence (78%)" but flagged it for professional review due to the irregular border pattern.\n\nI then visited my dermatologist at Stanford Medical Center, who performed a physical exam and ordered a biopsy. The result: <em>benign compound nevus with mild atypia</em> — essentially the same conclusion the AI reached.\n\nWhat struck me was the specific note about "border irregularity" appearing in both the AI report and my dermatologist's notes verbatim. The AI was capturing the same clinical concern a trained specialist was seeing — that's genuinely impressive.`,
        tags: ['#AIAccuracy', '#SkinCare', '#DermDiagnosis'],
        likes: 142,
        replies: 24,
        image: null
    },
    {
        id: 2,
        author: 'Dr. James Chen',
        avatar: 'https://i.pravatar.cc/150?img=12',
        doctorConsulted: true,
        time: '5 hours ago',
        category: 'Health Info',
        title: "How to interpret your AI-generated ECG report: A beginner's guide",
        content: "AI diagnostics are becoming incredibly accurate for cardiac rhythms. In this post, I break down the 3 key indicators you should look for in your DocTor summary...",
        fullContent: `As a cardiologist who has been using AI-assisted diagnostics for four years, I've noticed that many patients struggle to interpret their AI-generated ECG reports. Let me walk you through the three most important indicators.\n\n<strong>1. Heart Rate (HR)</strong>\nA normal resting heart rate is 60–100 BPM. Athletes may normally sit as low as 40 BPM. The AI flags anything outside the normal range and contextualizes it against your age group.\n\n<strong>2. QRS Duration</strong>\nThis measures how long it takes electrical impulses to travel through the ventricles. Normal range: 0.06–0.10 seconds. A prolonged QRS (>0.12s) can indicate bundle branch block — the AI highlights this in yellow.\n\n<strong>3. ST Segment Analysis</strong>\nThis is where HCI DocTor's AI truly shines. ST elevation or depression can be early indicators of myocardial ischemia. The AI analyzes all 12 leads simultaneously and flags anomalies with confidence scores.\n\nRemember: an AI report is a screening tool, not a diagnosis. Always consult your cardiologist for interpretation.`,
        tags: ['#HeartHealth', '#ECG', '#CardiacCare'],
        likes: 385,
        replies: 56,
        image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=600'
    },
    {
        id: 3,
        author: 'Marcus Thompson',
        avatar: 'https://i.pravatar.cc/150?img=33',
        time: '8 hours ago',
        category: 'Question',
        title: "Anyone tried the new 'Mental Wellness' AI module yet?",
        content: "I've been using the physical diagnostic tools for a while, but I'm curious about the new mood tracking and anxiety analysis features. How personalized is the feedback?",
        fullContent: `Long-time user of HCI DocTor here (about 18 months). I've mainly used the CT and X-ray analysis features for monitoring my ongoing respiratory situation, but I just noticed there's a new "Mental Wellness" module in beta.\n\nHas anyone tried it? I'm particularly curious about:\n\n1. How does it assess mood patterns — self-reporting or actual physiological markers?\n2. The anxiety analysis — does it use voice patterns like the audio analysis tools?\n3. Is the feedback actually personalized to your health history, or is it generic wellness advice?\n4. Privacy concerns — does this data stay separate from medical records?\n\nI've been dealing with some health anxiety related to my respiratory condition (which is apparently totally normal), and I'm wondering if this could be a helpful supplementary tool alongside therapy.\n\nWould love to hear from anyone who's been in the beta!`,
        tags: ['#MentalHealth', '#WellnessTech', '#HealthAnxiety'],
        likes: 29,
        replies: 12,
        image: null
    }
];

const MOCK_COMMENTS = {
    1: [
        {
            id: 1,
            author: 'Dr. James Chen',
            avatar: 'https://i.pravatar.cc/150?img=12',
            doctorConsulted: true,
            time: '3 hours ago',
            text: "Fantastic validation case! The borderline pigmentation patterns are often the most challenging for both AI and dermatologists alike. The AI's flagging of border irregularity as the key concern is exactly the right clinical decision-making process — this is precisely how ABCDE criteria should be applied.",
            likes: 38
        },
        {
            id: 2,
            author: 'Amina K.',
            avatar: 'https://i.pravatar.cc/150?img=9',
            verified: true,
            time: '4 hours ago',
            text: "I had a very similar experience 6 months ago! The AI flagged a lesion I was worried about, and my dermatologist confirmed it was benign. I was impressed by how the AI explained its reasoning — the ABCDE breakdown was really educational.",
            likes: 24
        },
        {
            id: 3,
            author: 'Marcus Thompson',
            avatar: 'https://i.pravatar.cc/150?img=33',
            time: '5 hours ago',
            text: "Question for the author — how long did the AI analysis take? And did the app give you a confidence percentage? I've found that cases with confidence below 70% really need the professional follow-up regardless.",
            likes: 11
        }
    ],
    2: [
        {
            id: 1,
            author: 'Sarah Miller',
            avatar: 'https://i.pravatar.cc/150?img=1',
            verified: true,
            aiFeedback: true,
            time: '6 hours ago',
            text: "Incredibly helpful guide! I've been confused about the QRS duration metric for months. The explanation about athletes having naturally lower heart rates is something I didn't know either. Bookmarking this for reference.",
            likes: 45
        },
        {
            id: 2,
            author: 'James Wilson',
            avatar: 'https://i.pravatar.cc/150?img=13',
            time: '7 hours ago',
            text: "Dr. Chen, would you say the AI's ST-segment analysis is reliable enough to rule out ischemia without a follow-up? I had an AI flag that my cardiologist later confirmed was a false positive.",
            likes: 19
        }
    ],
    3: [
        {
            id: 1,
            author: 'Dr. Elena Rossi',
            avatar: 'https://i.pravatar.cc/150?img=5',
            doctorConsulted: true,
            time: '9 hours ago',
            text: "I've been in the Mental Wellness beta. The mood assessment uses a combination of self-reporting and voice pattern analysis. The feedback is reasonably personalized based on your health history — more nuanced than generic wellness apps, but still early days.",
            likes: 18
        }
    ]
};

/**
 * 커뮤니티 피드 페이지 (04번)
 */
async function renderCommunityFeedPage() {
    exitCTFullscreenMode();
    // 기존 페이지 모두 숨기기
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));

    // 동적 페이지 컨테이너 찾기 또는 생성
    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    // 실제 API에서 게시글 로드 (실패 시 MOCK_POSTS 사용)
    let posts = [];
    try {
        const params = new URLSearchParams(window.location.search);
        const category = params.get('category') || '';
        const url = category
            ? `/api/community/posts?category=${category}&limit=20`
            : '/api/community/posts?limit=20';
        const response = await fetch(url);
        if (response.ok) {
            const data = await response.json();
            const apiPosts = data.posts || [];
            if (apiPosts.length > 0) {
                posts = apiPosts.map(p => ({
                    id: p.id,
                    author: p.author || 'Anonymous',
                    avatar: null,
                    time: typeof formatDate === 'function' ? formatDate(p.created_at) : p.created_at,
                    category: p.category,
                    title: p.title,
                    content: p.content,
                    tags: [],
                    likes: p.likes || 0,
                    replies: 0,
                    image: null
                }));
            }
        }
    } catch (e) {
        // 서버 미연결 시 무시
    }

    // API 결과가 없으면 MOCK_POSTS 사용
    if (posts.length === 0) {
        posts = MOCK_POSTS;
    }

    const html = `
        <div class="page active" style="animation: fadeIn 0.3s ease;">
            <div class="community-feed-layout">
                <!-- Left Sidebar -->
                <aside class="feed-sidebar-left">
                    ${renderFeedSidebar()}
                </aside>

                <!-- Main Feed -->
                <main class="feed-main">
                    <!-- Tabs -->
                    <div class="feed-tabs">
                        <button class="feed-tab active">Newest</button>
                        <button class="feed-tab">Most Helpful</button>
                        <button class="feed-tab">Verified Only</button>
                    </div>

                    <!-- Posts -->
                    ${posts.map(post => renderPostCard(post)).join('')}
                </main>

                <!-- Right Sidebar -->
                <aside class="feed-sidebar-right">
                    ${renderFeedWidgets()}
                </aside>
            </div>
        </div>
    `;

    dynamicContainer.innerHTML = html;
}

/**
 * Feed 좌측 사이드바 렌더링
 */
function renderFeedSidebar() {
    return `
        <div class="sidebar-brand">
            <h3>CATEGORIES</h3>
        </div>
        <nav class="sidebar-nav">
            <a href="/community/feed" class="sidebar-link active" data-link>
                <i class="fas fa-th-large"></i>
                All Posts
            </a>
            <a href="#" class="sidebar-link" onclick="showComingSoon('Diagnosis Reviews'); return false;">
                <i class="fas fa-clipboard-check"></i>
                Diagnosis Reviews
            </a>
            <a href="#" class="sidebar-link" onclick="showComingSoon('Health Info'); return false;">
                <i class="fas fa-book-medical"></i>
                Health Info
            </a>
            <a href="#" class="sidebar-link" onclick="showComingSoon('Saved Guides'); return false;">
                <i class="fas fa-bookmark"></i>
                Saved Guides
            </a>
        </nav>

        <div class="sidebar-brand" style="margin-top: 32px;">
            <h3>TRENDING TAGS</h3>
        </div>
        <div style="padding: 0 var(--space-xl); display: flex; flex-direction: column; gap: 8px;">
            <button class="community-card-tag" onclick="showComingSoon('#AIAccuracy'); return false;" style="cursor: pointer; border: none; justify-content: flex-start;">#AIAccuracy</button>
            <button class="community-card-tag" onclick="showComingSoon('#Telemedicine'); return false;" style="cursor: pointer; border: none; justify-content: flex-start;">#Telemedicine</button>
            <button class="community-card-tag" onclick="showComingSoon('#SkinCare'); return false;" style="cursor: pointer; border: none; justify-content: flex-start;">#SkinCare</button>
            <button class="community-card-tag" onclick="showComingSoon('#HeartHealth'); return false;" style="cursor: pointer; border: none; justify-content: flex-start;">#HeartHealth</button>
            <button class="community-card-tag" onclick="showComingSoon('#MentalHealth'); return false;" style="cursor: pointer; border: none; justify-content: flex-start;">#MentalHealth</button>
        </div>
    `;
}

/**
 * 게시글 카드 렌더링
 */
function renderPostCard(post) {
    // 레지스트리에 저장
    if (typeof postRegistry !== 'undefined') postRegistry[post.id] = post;

    const badges = [];
    if (post.verified) badges.push('<span class="post-badge verified"><i class="fas fa-check-circle"></i> Verified</span>');
    if (post.aiFeedback) badges.push('<span class="post-badge ai-feedback"><i class="fas fa-robot"></i> AI Feedback</span>');
    if (post.doctorConsulted) badges.push('<span class="post-badge doctor-consulted"><i class="fas fa-user-md"></i> Doctor Consulted</span>');

    const avatarHtml = post.avatar
        ? `<img src="${post.avatar}" alt="${post.author}" class="post-avatar">`
        : `<div class="post-avatar-placeholder">${(post.author || 'A')[0].toUpperCase()}</div>`;

    const saved = typeof isPostSaved === 'function' && isPostSaved(post.id);

    return `
        <article class="post-card" onclick="router.navigateTo('/community/post/${post.id}')" style="cursor:pointer;">
            <div class="post-header">
                ${avatarHtml}
                <div class="post-info">
                    <div class="post-author-row">
                        <span class="post-author">${post.author}</span>
                        ${badges.join('')}
                    </div>
                    <div class="post-meta">
                        ${post.time}<span class="post-meta-dot">•</span>${post.category}
                    </div>
                </div>
                <button class="post-action post-bookmark ${saved ? 'bookmarked' : ''}" title="${saved ? 'Remove bookmark' : 'Save'}"
                    onclick="event.stopPropagation(); bookmarkPost(${post.id}, this);">
                    <i class="fas fa-bookmark"></i>
                </button>
            </div>
            <h3 class="post-title">${post.title}</h3>
            <p class="post-content">${post.content}</p>
            ${post.image ? `<img src="${post.image}" alt="Post image" class="post-image">` : ''}
            <div class="post-actions">
                <button class="post-action" onclick="event.stopPropagation(); likePost(${post.id}, this);">
                    <i class="fas fa-thumbs-up"></i>
                    <span>${post.likes}</span>
                </button>
                <button class="post-action" onclick="event.stopPropagation(); router.navigateTo('/community/post/${post.id}');">
                    <i class="fas fa-comment"></i>
                    <span>${post.replies}</span>
                </button>
                <button class="post-action post-action-share" onclick="event.stopPropagation(); showComingSoon('Share');">
                    <i class="fas fa-share"></i>
                </button>
            </div>
        </article>
    `;
}

/**
 * Feed 우측 위젯 렌더링
 */
function renderFeedWidgets() {
    return `
        <!-- Community Trust Widget -->
        <div class="feed-widget trust">
            <div class="widget-header">
                <i class="fas fa-info-circle"></i>
                <h3 class="widget-title">Community Trust</h3>
            </div>
            <div class="widget-content">
                <div class="trust-item">
                    <i class="fas fa-shield-alt"></i>
                    <span>HCI DocTor is a support platform. While our AI is advanced, always consult licensed professional for medical emergencies.</span>
                </div>
                <div class="trust-item">
                    <i class="fas fa-check-circle"></i>
                    <span>Posts with 'Verified' labels are cross-referenced with medical reports.</span>
                </div>
                <div class="trust-item">
                    <i class="fas fa-user-md"></i>
                    <span>Professional medical advice is provided by certified doctors.</span>
                </div>
                <button class="btn-guidelines" onclick="showComingSoon('Community Guidelines');">
                    Read Guidelines
                </button>
            </div>
        </div>

        <!-- Top Contributors Widget -->
        <div class="feed-widget">
            <div class="widget-header">
                <i class="fas fa-trophy"></i>
                <h3 class="widget-title">TOP CONTRIBUTORS</h3>
            </div>
            <div class="contributors-list">
                <div class="contributor-item">
                    <img src="https://i.pravatar.cc/150?img=5" alt="Dr. Elena Rossi" class="contributor-avatar">
                    <div class="contributor-info">
                        <div class="contributor-name">Dr. Elena Rossi</div>
                        <div class="contributor-role">EXPERT MODERATOR</div>
                    </div>
                    <div class="contributor-points">2.4k pts</div>
                </div>
                <div class="contributor-item">
                    <img src="https://i.pravatar.cc/150?img=13" alt="James Wilson" class="contributor-avatar">
                    <div class="contributor-info">
                        <div class="contributor-name">James Wilson</div>
                        <div class="contributor-role helpful">HELPFUL PEER</div>
                    </div>
                    <div class="contributor-points">1.8k pts</div>
                </div>
                <div class="contributor-item">
                    <img src="https://i.pravatar.cc/150?img=9" alt="Amina K." class="contributor-avatar">
                    <div class="contributor-info">
                        <div class="contributor-name">Amina K.</div>
                        <div class="contributor-role verified">VERIFIED PATIENT</div>
                    </div>
                    <div class="contributor-points">950 pts</div>
                </div>
            </div>
        </div>
    `;
}

/**
 * 로그인 페이지 (05번)
 */
function renderLoginPage() {
    exitCTFullscreenMode();
    openLoginModal();
    showPage('home');
}

/**
 * 회원가입 페이지 (06번)
 */
function renderSignupPage() {
    exitCTFullscreenMode();
    openRegisterModal();
    showPage('home');
}

/**
 * 소개 페이지 (07번)
 */
function renderAboutPage() {
    exitCTFullscreenMode();
    showPage('about');
}

/**
 * CT 분석 상세 페이지 (08번)
 */
function renderCTAnalysisPage() {
    enterCTFullscreenMode();
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    const html = `
        <div class="page active" style="animation:none; min-height:100vh; background:#0f172a; color:#f1f5f9; display:flex; flex-direction:column; font-family:inherit;">

            <!-- Header -->
            <header style="display:flex; align-items:center; gap:1rem; padding:1rem 1.5rem; border-bottom:1px solid rgba(255,255,255,0.08); flex-shrink:0;">
                <button onclick="router.navigateTo('/')" title="Back to Home" style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); color:#f1f5f9; width:36px; height:36px; border-radius:8px; cursor:pointer; display:flex; align-items:center; justify-content:center;">
                    <i class="fas fa-arrow-left"></i>
                </button>
                <div>
                    <div style="font-size:1.05rem; font-weight:700; letter-spacing:-0.01em;">CT Lung Analysis</div>
                    <div style="font-size:0.72rem; color:#64748b;">MedGemma 1.5 4B LoRA &nbsp;·&nbsp; Accuracy 87.35% &nbsp;·&nbsp; Normal / Benign / Malignant</div>
                </div>
            </header>

            <!-- Content -->
            <div id="ct-main-content" style="flex:1; display:flex; align-items:center; justify-content:center; padding:2rem;">

                <!-- Upload Zone -->
                <div style="width:100%; max-width:520px; text-align:center;">
                    <div id="ct-drop-zone"
                         onclick="document.getElementById('ct-file-input').click()"
                         ondragover="event.preventDefault(); document.getElementById('ct-drop-zone').style.borderColor='#38bdf8'; document.getElementById('ct-drop-zone').style.background='rgba(56,189,248,0.05)';"
                         ondragleave="document.getElementById('ct-drop-zone').style.borderColor='rgba(255,255,255,0.15)'; document.getElementById('ct-drop-zone').style.background='rgba(255,255,255,0.02)';"
                         ondrop="event.preventDefault(); document.getElementById('ct-drop-zone').style.borderColor='rgba(255,255,255,0.15)'; document.getElementById('ct-drop-zone').style.background='rgba(255,255,255,0.02)'; const f=event.dataTransfer.files[0]; if(f) ctAnalyzeFile(f);"
                         style="border:2px dashed rgba(255,255,255,0.15); border-radius:16px; padding:3.5rem 2rem; cursor:pointer; transition:all 0.2s; background:rgba(255,255,255,0.02);">
                        <i class="fas fa-cloud-upload-alt" style="font-size:3rem; color:#38bdf8; display:block; margin-bottom:1rem;"></i>
                        <div style="font-size:1rem; font-weight:600; margin-bottom:0.4rem;">Upload CT Image</div>
                        <div style="font-size:0.82rem; color:#64748b; margin-bottom:1.2rem;">Click or drag &amp; drop to upload</div>
                        <div style="font-size:0.72rem; color:#334155; background:rgba(255,255,255,0.05); border-radius:6px; padding:0.4rem 0.8rem; display:inline-block;">PNG · JPG · JPEG</div>
                    </div>
                    <input type="file" id="ct-file-input" accept="image/*" style="display:none;" onchange="const f=this.files[0]; this.value=''; if(f) ctAnalyzeFile(f);">
                </div>

            </div>
        </div>
    `;

    dynamicContainer.innerHTML = html;

    window.ctAnalyzeFile = async function(file) {
        const mainContent = document.getElementById('ct-main-content');
        if (!mainContent) return;

        // 1. 이미지 미리보기 + 로딩 상태
        const reader = new FileReader();
        reader.onload = e => {
            mainContent.innerHTML = `
                <div style="width:100%; max-width:900px; display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">
                    <img src="${e.target.result}" style="width:300px; min-width:200px; flex-shrink:0; border-radius:12px; border:1px solid rgba(255,255,255,0.1); object-fit:contain; max-height:400px;">
                    <div id="ct-result-area" style="flex:1; min-width:260px; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:2rem 0;">
                        <i class="fas fa-spinner fa-spin" style="font-size:2.5rem; color:#38bdf8; margin-bottom:1rem;"></i>
                        <div style="color:#94a3b8; font-size:0.9rem;">Analyzing CT scan...</div>
                    </div>
                </div>
            `;

            // 2. API 호출
            const formData = new FormData();
            formData.append('file', file);
            fetch('/api/ct/analyze', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(result => {
                    const ra = document.getElementById('ct-result-area');
                    if (!ra) return;
                    if (result.error) {
                        ra.innerHTML = `<div style="color:#f87171; font-size:0.9rem;"><i class="fas fa-exclamation-triangle"></i> ${result.error}</div><button onclick="router.navigateTo('/analysis/ct')" style="margin-top:1rem; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#f1f5f9; padding:0.5rem 1rem; border-radius:8px; cursor:pointer; font-size:0.82rem;"><i class="fas fa-redo"></i> Retry</button>`;
                        return;
                    }
                    const pred = (result.prediction || '').toLowerCase();
                    const color = pred === 'normal' ? '#10b981' : pred === 'malignant' ? '#ef4444' : '#f59e0b';
                    const icon  = pred === 'normal' ? 'fa-check-circle' : pred === 'malignant' ? 'fa-times-circle' : 'fa-exclamation-circle';

                    let confHtml = '';
                    if (result.confidence && typeof result.confidence === 'object') {
                        confHtml += '<div style="margin-top:1.5rem; width:100%;">';
                        confHtml += '<div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem; font-weight:600;">Confidence</div>';
                        for (const [k, v] of Object.entries(result.confidence)) {
                            const pct = typeof v === 'number' ? (v * 100).toFixed(1) : v;
                            const w   = typeof v === 'number' ? Math.min(v * 100, 100) : 50;
                            confHtml += `<div style="margin-bottom:0.65rem;"><div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:0.28rem; text-transform:capitalize;"><span style="color:#cbd5e1;">${k}</span><span style="font-weight:600; color:#f1f5f9;">${pct}${typeof v === 'number' ? '%' : ''}</span></div><div style="background:rgba(255,255,255,0.08); border-radius:4px; height:5px;"><div style="background:${color}; height:5px; border-radius:4px; width:${w}%;"></div></div></div>`;
                        }
                        confHtml += '</div>';
                    }

                    ra.innerHTML = `
                        <div style="width:100%;">
                            <div style="display:flex; align-items:center; gap:0.85rem; margin-bottom:0.25rem;">
                                <i class="fas ${icon}" style="font-size:2.2rem; color:${color};"></i>
                                <div>
                                    <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">Diagnosis</div>
                                    <div style="font-size:1.6rem; font-weight:800; color:${color}; text-transform:uppercase; letter-spacing:0.02em;">${result.prediction || 'Unknown'}</div>
                                </div>
                            </div>
                            ${confHtml}
                            ${result.explanation ? `<div style="margin-top:1.5rem; padding-top:1.25rem; border-top:1px solid rgba(255,255,255,0.08); font-size:0.8rem; color:#94a3b8; line-height:1.75;">${result.explanation}</div>` : ''}
                            <button onclick="router.navigateTo('/analysis/ct')" style="margin-top:1.5rem; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; padding:0.55rem 1.1rem; border-radius:8px; cursor:pointer; font-size:0.82rem; display:inline-flex; align-items:center; gap:0.4rem;">
                                <i class="fas fa-redo"></i> Analyze Again
                            </button>
                        </div>
                    `;
                })
                .catch(e => {
                    const ra = document.getElementById('ct-result-area');
                    if (ra) ra.innerHTML = `<div style="color:#f87171; font-size:0.9rem;"><i class="fas fa-exclamation-triangle"></i> Analysis failed: ${e.message}</div><button onclick="router.navigateTo('/analysis/ct')" style="margin-top:1rem; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#f1f5f9; padding:0.5rem 1rem; border-radius:8px; cursor:pointer; font-size:0.82rem;">Retry</button>`;
                });
        };
        reader.readAsDataURL(file);
    };
}

/**
 * 커뮤니티 게시글 상세 페이지 (Post Detail View)
 */
async function renderPostDetailPage(params) {
    enterCTFullscreenMode();

    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));

    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    const postId = params ? parseInt(params.id) : 1;

    // MOCK_POSTS에서 먼저 찾기
    let post = MOCK_POSTS.find(p => p.id === postId);
    let comments = post ? (MOCK_COMMENTS[post.id] || []) : [];

    // MOCK에 없으면 API에서 fetch
    if (!post) {
        dynamicContainer.innerHTML = `<div style="padding:3rem;text-align:center;color:var(--gray-400);"><i class="fas fa-spinner fa-spin" style="font-size:32px;"></i><p style="margin-top:1rem;">Loading...</p></div>`;
        try {
            const [postRes, commentsRes] = await Promise.all([
                fetch(`/api/community/posts/${postId}`),
                fetch(`/api/community/posts/${postId}/comments`)
            ]);
            if (postRes.ok) {
                const data = await postRes.json();
                post = {
                    id: data.id,
                    title: data.title,
                    content: data.content,
                    fullContent: data.content,
                    author: data.author || 'Anonymous',
                    avatar: null,
                    time: typeof formatDate === 'function' ? formatDate(data.created_at) : data.created_at,
                    category: data.category,
                    tags: [],
                    likes: data.likes || 0,
                    replies: 0
                };
            }
            if (commentsRes.ok) {
                const cData = await commentsRes.json();
                comments = (cData.comments || []).map(c => ({
                    id: c.id,
                    author: c.author || 'Anonymous',
                    avatar: null,
                    time: typeof formatDate === 'function' ? formatDate(c.created_at) : c.created_at,
                    text: c.content,
                    likes: 0
                }));
            }
        } catch (e) {}

        if (!post) {
            dynamicContainer.innerHTML = `<div style="padding:3rem;text-align:center;"><p>Post not found.</p><button class="btn btn-primary" style="margin-top:1rem;" onclick="router.navigateTo('/community')">Back to Community</button></div>`;
            return;
        }
    }

    const badges = [];
    if (post.verified)       badges.push('<span class="post-badge verified"><i class="fas fa-check-circle"></i> Verified</span>');
    if (post.aiFeedback)     badges.push('<span class="post-badge ai-feedback"><i class="fas fa-robot"></i> AI Feedback</span>');
    if (post.doctorConsulted)badges.push('<span class="post-badge doctor-consulted"><i class="fas fa-user-md"></i> Doctor Consulted</span>');

    const html = `
        <div class="page active" style="animation: fadeIn 0.3s ease;">

          <!-- ── Community App Header ── -->
          <header class="community-app-header">
            <div class="community-app-logo">
              <div class="community-app-logo-icon"><i class="fas fa-shield-alt"></i></div>
              <span class="community-app-logo-text">HCI DocTor</span>
            </div>
            <div class="community-app-search">
              <i class="fas fa-search"></i>
              <input type="text" placeholder="Search diagnosis reviews, health info...">
            </div>
            <div class="community-app-actions">
              <button class="community-app-link" onclick="router.navigateTo('/about')">About HCI DocTor</button>
              <button class="community-app-write" onclick="showComingSoon('Write Post')">
                <i class="fas fa-plus"></i> Write Post
              </button>
              <div class="community-app-avatar">
                <img src="https://i.pravatar.cc/32?img=1" alt="User">
              </div>
            </div>
          </header>

          <!-- ── 3-Column Layout ── -->
          <div class="post-detail-layout">

            <!-- LEFT SIDEBAR -->
            <aside class="post-detail-sidebar-left">
              ${renderFeedSidebar()}
            </aside>

            <!-- CENTER: Post Detail + Comments -->
            <main class="post-detail-main">

              <!-- Back button -->
              <button class="post-back-link" onclick="router.navigateTo('/community')">
                <i class="fas fa-arrow-left"></i> Back to Feed
              </button>

              <!-- Post Card -->
              <article class="post-detail-card">

                <!-- Author Header -->
                <div class="post-detail-head">
                  ${post.avatar
                    ? `<img src="${post.avatar}" alt="${post.author}" class="post-detail-avatar">`
                    : `<div class="post-avatar-placeholder post-detail-avatar" style="width:48px;height:48px;font-size:20px;">${(post.author||'A')[0].toUpperCase()}</div>`
                  }
                  <div class="post-detail-author-info">
                    <div class="post-detail-author-row">
                      <span class="post-detail-author-name">${post.author}</span>
                      ${badges.join('')}
                    </div>
                    <div class="post-detail-meta">
                      Posted ${post.time}<span class="post-meta-dot">•</span>${post.category}
                    </div>
                  </div>
                  <button class="post-follow-btn" onclick="showComingSoon('Follow')">
                    <i class="fas fa-user-plus"></i> Follow
                  </button>
                </div>

                <!-- Title -->
                <h1 class="post-detail-title">${post.title}</h1>

                <!-- Full Body -->
                <div class="post-detail-body">
                  ${(post.fullContent || post.content).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}
                </div>

                ${post.image ? `<img src="${post.image}" alt="Post image" class="post-detail-img">` : ''}

                <!-- AI Verified Box -->
                <div class="post-ai-box">
                  <div class="post-ai-box-header">
                    <i class="fas fa-robot"></i>
                    <span class="post-ai-box-label">AI VERIFIED ANALYSIS</span>
                    <span class="post-ai-confidence">Confidence: 94.2%</span>
                  </div>
                  <p class="post-ai-box-text">Based on the diagnostic comparison provided, the correlation between HCI DocTor AI and clinical findings is consistent with documented accuracy benchmarks (92–96%). The AI's detection methodology aligns with established clinical criteria. This case supports the platform's efficacy as a primary screening tool.</p>
                </div>

                <!-- Tags -->
                <div class="post-detail-tags">
                  ${(post.tags || []).map(tag => `<button class="post-detail-tag" onclick="showComingSoon('${tag}')">${tag}</button>`).join('')}
                </div>

                <!-- Reaction Bar -->
                <div class="post-reactions-bar">
                  <button class="reaction-btn" id="detail-like-btn" onclick="likePost(${post.id}, this)">
                    <i class="fas fa-thumbs-up"></i> <span>${post.likes}</span>
                  </button>
                  <button class="reaction-btn" onclick="document.querySelector('.comment-compose-input')?.focus()">
                    <i class="fas fa-comment"></i> <span>${comments.length} Comments</span>
                  </button>
                  <button class="reaction-btn" onclick="showComingSoon('Share')">
                    <i class="fas fa-share-alt"></i> <span>Share</span>
                  </button>
                  <button class="reaction-btn bookmark-btn" onclick="showComingSoon('Bookmark')" title="Bookmark">
                    <i class="fas fa-bookmark"></i>
                  </button>
                </div>
              </article>

              <!-- Comments Section -->
              <section class="comments-section">
                <h3 class="comments-title">${comments.length} Comments</h3>

                <!-- Comment Compose -->
                <div class="comment-compose">
                  <div class="post-avatar-placeholder comment-compose-avatar" style="width:36px;height:36px;font-size:14px;">Me</div>
                  <div class="comment-compose-right">
                    <textarea id="comment-input-${post.id}" class="comment-compose-input" placeholder="Leave a comment..."></textarea>
                    <div class="comment-compose-footer">
                      <div class="comment-compose-opts">
                        <input type="text" id="comment-author-${post.id}" class="compose-opt-btn" placeholder="Nickname (optional)" style="font-size:12px;border:1px solid var(--gray-200);border-radius:4px;padding:4px 8px;width:120px;">
                      </div>
                      <button class="comment-submit-btn" onclick="submitComment(${post.id})">Post Comment</button>
                    </div>
                  </div>
                </div>

                <!-- Comment List -->
                <div class="comments-list">
                  ${comments.map(c => renderCommentItem(c)).join('')}
                </div>
              </section>
            </main>

            <!-- RIGHT SIDEBAR -->
            <aside class="post-detail-sidebar-right">
              ${renderFeedWidgets()}
            </aside>
          </div>
        </div>
    `;

    dynamicContainer.innerHTML = html;
}

/**
 * 댓글 아이템 렌더링 헬퍼
 */
function renderCommentItem(comment) {
    const badges = [];
    if (comment.verified)       badges.push('<span class="post-badge verified"><i class="fas fa-check-circle"></i> Verified</span>');
    if (comment.aiFeedback)     badges.push('<span class="post-badge ai-feedback"><i class="fas fa-robot"></i> AI Feedback</span>');
    if (comment.doctorConsulted)badges.push('<span class="post-badge doctor-consulted"><i class="fas fa-user-md"></i> Doctor Consulted</span>');

    const commentAvatarHtml = comment.avatar
        ? `<img src="${comment.avatar}" alt="${comment.author}" class="comment-avatar">`
        : `<div class="post-avatar-placeholder comment-avatar" style="width:36px;height:36px;font-size:14px;">${(comment.author||'A')[0].toUpperCase()}</div>`;

    return `
        <div class="comment-item">
          ${commentAvatarHtml}
          <div class="comment-body">
            <div class="comment-head">
              <span class="comment-author">${comment.author}</span>
              ${badges.join('')}
              <span class="comment-time">${comment.time}</span>
            </div>
            <p class="comment-text">${comment.text}</p>
            <div class="comment-actions">
              <button class="comment-action-btn" onclick="showComingSoon('Like comment')">
                <i class="fas fa-thumbs-up"></i> ${comment.likes}
              </button>
              <button class="comment-action-btn" onclick="showComingSoon('Reply')">
                <i class="fas fa-reply"></i> Reply
              </button>
            </div>
          </div>
        </div>
    `;
}

/**
 * AI Heatmap 토글 (실제 오버레이 표시/숨기기)
 */
function toggleHeatmap(element) {
    element.classList.toggle('active');
    const overlay = document.getElementById('ct-heatmap-overlay');
    if (overlay) {
        if (element.classList.contains('active')) {
            overlay.classList.remove('hidden');
        } else {
            overlay.classList.add('hidden');
        }
    }
}

/**
 * X-ray 분석 상세 페이지 (09번)
 */
function renderXrayAnalysisPage() {
    enterCTFullscreenMode();
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    const html = `
        <div class="page active" style="animation:none; min-height:100vh; background:#0f172a; color:#f1f5f9; display:flex; flex-direction:column; font-family:inherit;">

            <!-- Header -->
            <header style="display:flex; align-items:center; gap:1rem; padding:1rem 1.5rem; border-bottom:1px solid rgba(255,255,255,0.08); flex-shrink:0;">
                <button onclick="router.navigateTo('/')" title="Back to Home" style="background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); color:#f1f5f9; width:36px; height:36px; border-radius:8px; cursor:pointer; display:flex; align-items:center; justify-content:center;">
                    <i class="fas fa-arrow-left"></i>
                </button>
                <div>
                    <div style="font-size:1.05rem; font-weight:700; letter-spacing:-0.01em;">Chest X-ray Analysis</div>
                    <div style="font-size:0.72rem; color:#64748b;">MedGemma 1.5 4B LoRA &nbsp;·&nbsp; Accuracy 67.21% &nbsp;·&nbsp; Normal / Abnormal</div>
                </div>
            </header>

            <!-- Content -->
            <div id="xray-main-content" style="flex:1; display:flex; align-items:center; justify-content:center; padding:2rem;">

                <!-- Upload Zone -->
                <div style="width:100%; max-width:520px; text-align:center;">
                    <div id="xray-drop-zone"
                         onclick="document.getElementById('xray-file-input').click()"
                         ondragover="event.preventDefault(); document.getElementById('xray-drop-zone').style.borderColor='#38bdf8'; document.getElementById('xray-drop-zone').style.background='rgba(56,189,248,0.05)';"
                         ondragleave="document.getElementById('xray-drop-zone').style.borderColor='rgba(255,255,255,0.15)'; document.getElementById('xray-drop-zone').style.background='rgba(255,255,255,0.02)';"
                         ondrop="event.preventDefault(); document.getElementById('xray-drop-zone').style.borderColor='rgba(255,255,255,0.15)'; document.getElementById('xray-drop-zone').style.background='rgba(255,255,255,0.02)'; const f=event.dataTransfer.files[0]; if(f) xrayAnalyzeFile(f);"
                         style="border:2px dashed rgba(255,255,255,0.15); border-radius:16px; padding:3.5rem 2rem; cursor:pointer; transition:all 0.2s; background:rgba(255,255,255,0.02);">
                        <i class="fas fa-lungs" style="font-size:3rem; color:#38bdf8; display:block; margin-bottom:1rem;"></i>
                        <div style="font-size:1rem; font-weight:600; margin-bottom:0.4rem;">Upload X-ray Image</div>
                        <div style="font-size:0.82rem; color:#64748b; margin-bottom:1.2rem;">Click or drag &amp; drop to upload</div>
                        <div style="font-size:0.72rem; color:#334155; background:rgba(255,255,255,0.05); border-radius:6px; padding:0.4rem 0.8rem; display:inline-block;">PNG · JPG · JPEG</div>
                    </div>
                    <input type="file" id="xray-file-input" accept="image/*" style="display:none;" onchange="const f=this.files[0]; this.value=''; if(f) xrayAnalyzeFile(f);">
                </div>

            </div>
        </div>
    `;

    dynamicContainer.innerHTML = html;

    window.xrayAnalyzeFile = async function(file) {
        const mainContent = document.getElementById('xray-main-content');
        if (!mainContent) return;

        const reader = new FileReader();
        reader.onload = e => {
            mainContent.innerHTML = `
                <div style="width:100%; max-width:900px; display:flex; gap:2rem; align-items:flex-start; flex-wrap:wrap;">
                    <img src="${e.target.result}" style="width:300px; min-width:200px; flex-shrink:0; border-radius:12px; border:1px solid rgba(255,255,255,0.1); object-fit:contain; max-height:400px;">
                    <div id="xray-result-area" style="flex:1; min-width:260px; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:2rem 0;">
                        <i class="fas fa-spinner fa-spin" style="font-size:2.5rem; color:#38bdf8; margin-bottom:1rem;"></i>
                        <div style="color:#94a3b8; font-size:0.9rem;">Analyzing X-ray...</div>
                    </div>
                </div>
            `;

            const formData = new FormData();
            formData.append('file', file);
            fetch('/api/xray/analyze', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(result => {
                    const ra = document.getElementById('xray-result-area');
                    if (!ra) return;
                    if (result.error) {
                        ra.innerHTML = `<div style="color:#f87171; font-size:0.9rem;"><i class="fas fa-exclamation-triangle"></i> ${result.error}</div><button onclick="router.navigateTo('/analysis/xray')" style="margin-top:1rem; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#f1f5f9; padding:0.5rem 1rem; border-radius:8px; cursor:pointer; font-size:0.82rem;"><i class="fas fa-redo"></i> Retry</button>`;
                        return;
                    }
                    const pred = (result.prediction || '').toLowerCase();
                    const isNormal = pred === 'normal';
                    const color = isNormal ? '#10b981' : '#ef4444';
                    const icon  = isNormal ? 'fa-check-circle' : 'fa-times-circle';

                    let confHtml = '';
                    if (result.confidence && typeof result.confidence === 'object') {
                        confHtml += '<div style="margin-top:1.5rem; width:100%;">';
                        confHtml += '<div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.75rem; font-weight:600;">Confidence</div>';
                        for (const [k, v] of Object.entries(result.confidence)) {
                            const pct = typeof v === 'number' ? (v * 100).toFixed(1) : v;
                            const w   = typeof v === 'number' ? Math.min(v * 100, 100) : 50;
                            confHtml += `<div style="margin-bottom:0.65rem;"><div style="display:flex; justify-content:space-between; font-size:0.78rem; margin-bottom:0.28rem; text-transform:capitalize;"><span style="color:#cbd5e1;">${k}</span><span style="font-weight:600; color:#f1f5f9;">${pct}${typeof v === 'number' ? '%' : ''}</span></div><div style="background:rgba(255,255,255,0.08); border-radius:4px; height:5px;"><div style="background:${color}; height:5px; border-radius:4px; width:${w}%;"></div></div></div>`;
                        }
                        confHtml += '</div>';
                    }

                    ra.innerHTML = `
                        <div style="width:100%;">
                            <div style="display:flex; align-items:center; gap:0.85rem; margin-bottom:0.25rem;">
                                <i class="fas ${icon}" style="font-size:2.2rem; color:${color};"></i>
                                <div>
                                    <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.08em;">Diagnosis</div>
                                    <div style="font-size:1.6rem; font-weight:800; color:${color}; text-transform:uppercase; letter-spacing:0.02em;">${result.prediction || 'Unknown'}</div>
                                </div>
                            </div>
                            ${confHtml}
                            ${result.explanation ? `<div style="margin-top:1.5rem; padding-top:1.25rem; border-top:1px solid rgba(255,255,255,0.08); font-size:0.8rem; color:#94a3b8; line-height:1.75;">${result.explanation}</div>` : ''}
                            <button onclick="router.navigateTo('/analysis/xray')" style="margin-top:1.5rem; background:rgba(255,255,255,0.07); border:1px solid rgba(255,255,255,0.12); color:#cbd5e1; padding:0.55rem 1.1rem; border-radius:8px; cursor:pointer; font-size:0.82rem; display:inline-flex; align-items:center; gap:0.4rem;">
                                <i class="fas fa-redo"></i> Analyze Again
                            </button>
                        </div>
                    `;
                })
                .catch(e => {
                    const ra = document.getElementById('xray-result-area');
                    if (ra) ra.innerHTML = `<div style="color:#f87171; font-size:0.9rem;"><i class="fas fa-exclamation-triangle"></i> Analysis failed: ${e.message}</div><button onclick="router.navigateTo('/analysis/xray')" style="margin-top:1rem; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#f1f5f9; padding:0.5rem 1rem; border-radius:8px; cursor:pointer; font-size:0.82rem;">Retry</button>`;
                });
        };
        reader.readAsDataURL(file);
    };
}

/**
 * 음향 분석 상세 페이지 (10번 - 임시) - 호흡음 + 기침음
 */
function renderAcousticAnalysisPage() {
    exitCTFullscreenMode();
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    let dynamicContainer = document.getElementById('dynamic-page-container');
    if (!dynamicContainer) {
        const appContainer = document.getElementById('app-container');
        dynamicContainer = document.createElement('div');
        dynamicContainer.id = 'dynamic-page-container';
        appContainer.appendChild(dynamicContainer);
    }

    const html = `
        <div class="page active" style="animation:fadeIn 0.3s ease; padding: 2rem 0; min-height:60vh;">
            <div class="container">

                <!-- Back + Title -->
                <button onclick="router.navigateTo('/')" style="display:inline-flex; align-items:center; gap:0.5rem; background:none; border:none; color:var(--gray-500); cursor:pointer; font-size:0.9rem; margin-bottom:1.5rem; padding:0;">
                    <i class="fas fa-arrow-left"></i> Home
                </button>
                <h2 style="font-size:1.6rem; font-weight:800; color:var(--gray-900); margin-bottom:0.4rem;">Acoustic Analysis</h2>
                <p style="color:var(--gray-500); font-size:0.88rem; margin-bottom:2rem;">HeAR Model &nbsp;·&nbsp; Accuracy 96.74% &nbsp;·&nbsp; Normal / Abnormal</p>

                <!-- Two analysis cards -->
                <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:1.5rem;">

                    <!-- Breathing -->
                    <div style="background:var(--white); border-radius:16px; padding:2rem; box-shadow:var(--shadow-md); border:1px solid var(--gray-100);">
                        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.5rem;">
                            <div style="width:42px; height:42px; background:linear-gradient(135deg, #d1fae5, #6ee7b7); border-radius:10px; display:flex; align-items:center; justify-content:center;">
                                <i class="fas fa-wind" style="color:#059669;"></i>
                            </div>
                            <div>
                                <div style="font-weight:700; font-size:0.95rem;">Breathing Sound Analysis</div>
                                <div style="font-size:0.72rem; color:var(--gray-400);">HeAR Model</div>
                            </div>
                        </div>

                        <div id="breathing-upload-zone"
                             onclick="document.getElementById('breathing-file-input').click()"
                             ondragover="event.preventDefault(); document.getElementById('breathing-upload-zone').style.borderColor='#059669';"
                             ondragleave="document.getElementById('breathing-upload-zone').style.borderColor='var(--gray-200)';"
                             ondrop="event.preventDefault(); document.getElementById('breathing-upload-zone').style.borderColor='var(--gray-200)'; const f=event.dataTransfer.files[0]; if(f) audioAnalyzeFile(f,'breathing');"
                             style="border:2px dashed var(--gray-200); border-radius:12px; padding:2.5rem 1.5rem; text-align:center; cursor:pointer; transition:border-color 0.2s; margin-bottom:1rem;">
                            <i class="fas fa-microphone" style="font-size:2rem; color:#059669; display:block; margin-bottom:0.75rem;"></i>
                            <div style="font-size:0.88rem; font-weight:600; margin-bottom:0.35rem;">Upload Audio File</div>
                            <div style="font-size:0.75rem; color:var(--gray-400);">WAV · MP3 · OGG</div>
                        </div>
                        <input type="file" id="breathing-file-input" accept="audio/*" style="display:none;" onchange="const f=this.files[0]; this.value=''; if(f) audioAnalyzeFile(f,'breathing');">
                        <div id="breathing-result"></div>
                    </div>

                    <!-- Cough -->
                    <div style="background:var(--white); border-radius:16px; padding:2rem; box-shadow:var(--shadow-md); border:1px solid var(--gray-100);">
                        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.5rem;">
                            <div style="width:42px; height:42px; background:linear-gradient(135deg, #ede9fe, #c4b5fd); border-radius:10px; display:flex; align-items:center; justify-content:center;">
                                <i class="fas fa-lungs-virus" style="color:#7c3aed;"></i>
                            </div>
                            <div>
                                <div style="font-weight:700; font-size:0.95rem;">Cough Sound Analysis</div>
                                <div style="font-size:0.72rem; color:var(--gray-400);">HeAR Model</div>
                            </div>
                        </div>

                        <div id="cough-upload-zone"
                             onclick="document.getElementById('cough-file-input').click()"
                             ondragover="event.preventDefault(); document.getElementById('cough-upload-zone').style.borderColor='#7c3aed';"
                             ondragleave="document.getElementById('cough-upload-zone').style.borderColor='var(--gray-200)';"
                             ondrop="event.preventDefault(); document.getElementById('cough-upload-zone').style.borderColor='var(--gray-200)'; const f=event.dataTransfer.files[0]; if(f) audioAnalyzeFile(f,'cough');"
                             style="border:2px dashed var(--gray-200); border-radius:12px; padding:2.5rem 1.5rem; text-align:center; cursor:pointer; transition:border-color 0.2s; margin-bottom:1rem;">
                            <i class="fas fa-microphone-alt" style="font-size:2rem; color:#7c3aed; display:block; margin-bottom:0.75rem;"></i>
                            <div style="font-size:0.88rem; font-weight:600; margin-bottom:0.35rem;">Upload Audio File</div>
                            <div style="font-size:0.75rem; color:var(--gray-400);">WAV · MP3 · OGG</div>
                        </div>
                        <input type="file" id="cough-file-input" accept="audio/*" style="display:none;" onchange="const f=this.files[0]; this.value=''; if(f) audioAnalyzeFile(f,'cough');">
                        <div id="cough-result"></div>
                    </div>

                </div>
            </div>
        </div>
    `;

    dynamicContainer.innerHTML = html;

    window.audioAnalyzeFile = async function(file, type) {
        const resultEl = document.getElementById(type + '-result');
        const uploadZone = document.getElementById(type + '-upload-zone');
        if (!resultEl) return;

        // 로딩
        resultEl.innerHTML = `<div style="text-align:center; padding:1rem; color:var(--gray-400); font-size:0.85rem;"><i class="fas fa-spinner fa-spin" style="margin-right:0.4rem;"></i>Analyzing...</div>`;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        try {
            const res = await fetch('/api/audio/analyze', { method: 'POST', body: formData });
            const result = await res.json();

            if (result.error) {
                resultEl.innerHTML = `<div style="color:#ef4444; font-size:0.82rem; padding:0.75rem; background:#fef2f2; border-radius:8px;"><i class="fas fa-exclamation-triangle"></i> ${result.error}</div>`;
                return;
            }

            const pred = (result.prediction || '').toLowerCase();
            const isNormal = pred === 'normal';
            const color  = isNormal ? '#059669' : '#dc2626';
            const bg     = isNormal ? '#f0fdf4' : '#fef2f2';
            const border = isNormal ? '#bbf7d0' : '#fecaca';
            const icon   = isNormal ? 'fa-check-circle' : 'fa-times-circle';

            resultEl.innerHTML = `
                <div style="background:${bg}; border:1px solid ${border}; border-radius:10px; padding:1rem 1.25rem;">
                    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:${result.explanation ? '0.75rem' : '0'};">
                        <i class="fas ${icon}" style="color:${color}; font-size:1.2rem;"></i>
                        <span style="font-weight:700; color:${color}; font-size:1rem; text-transform:uppercase;">${result.prediction || 'Unknown'}</span>
                    </div>
                    ${result.explanation ? `<div style="font-size:0.78rem; color:var(--gray-600); line-height:1.6;">${result.explanation}</div>` : ''}
                </div>
            `;
        } catch (e) {
            resultEl.innerHTML = `<div style="color:#ef4444; font-size:0.82rem; padding:0.75rem; background:#fef2f2; border-radius:8px;"><i class="fas fa-exclamation-triangle"></i> Analysis failed: ${e.message}</div>`;
        }
    };
}

/**
 * 404 페이지
 */
function render404Page() {
    exitCTFullscreenMode();
    const content = `
        <div style="text-align: center; padding: 2rem 0;">
            <div style="font-size: 120px; color: var(--primary-color); margin-bottom: 1rem; font-weight: 900;">404</div>
            <h3 style="font-size: 24px; font-weight: 700; color: var(--gray-900); margin-bottom: 1rem;">
                Page Not Found
            </h3>
            <p style="color: var(--gray-500); margin-bottom: 2rem;">
                The page you requested does not exist or has been moved.
            </p>

            <button class="btn btn-primary" onclick="router.navigateTo('/')">
                <i class="fas fa-home"></i> Back to Home
            </button>
        </div>
    `;

    const html = createPageLayout('', content, false);
    renderDynamicPage(html);
}
