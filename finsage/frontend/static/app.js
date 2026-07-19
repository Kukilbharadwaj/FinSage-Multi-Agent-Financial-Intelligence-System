/**
 * FinSage AI — Frontend Application
 * Chat logic
 */

// ============================================================
// Configuration
// ============================================================
const API_BASE = window.location.origin;
const MAX_QUERIES = 5;
const EXAMPLE_QUERIES = [
    { icon: "📊", label: "Stock Analysis", text: "Should I buy Reliance stock today? How are its fundamentals?" },
    { icon: "💰", label: "Salary & Budget", text: "My salary is ₹80,000. How to manage it properly?" },
    { icon: "📉", label: "Tax Calculation", text: "I made 50k profit in TCS after 8 months. What is my tax?" },
    { icon: "📈", label: "Intraday & Options", text: "Nifty option chain max pain and PCR today?" },
    { icon: "🛡️", label: "Term Insurance", text: "Should I buy a term plan or ULIP?" },
    { icon: "🏦", label: "Home Loan EMI", text: "Home loan of 50 lakhs for 15 years, what is EMI and tax benefit?" },
    { icon: "💼", label: "Mutual Fund SIP", text: "Is Parag Parikh Flexi Cap a good fund for monthly SIP?" },
    { icon: "🥇", label: "Gold Investment", text: "Should I buy physical gold or SGB for investment?" },
    { icon: "👴", label: "Retirement (NPS)", text: "What are the tax benefits of NPS under 80CCD?" },
];

// ============================================================
// State
// ============================================================
let state = {
    history: [],
    queryCount: 0,
    userId: "web_" + Math.random().toString(36).substring(2, 14),
    isProcessing: false,
};

// ============================================================
// Initialization
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    renderExampleQueries();
    renderWelcomeChips();
    checkHealth();
    // Auto-focus input
    document.getElementById("queryInput").focus();
});

function renderExampleQueries() {
    const container = document.getElementById("exampleQueries");
    container.innerHTML = EXAMPLE_QUERIES.map((q, i) => `
        <div class="example-card" onclick="useExample(${i})">
            <div class="example-card-label">
                <span class="example-card-icon">${q.icon}</span>${q.label}
            </div>
            <div class="example-card-text">${q.text}</div>
        </div>
    `).join("");
}

function renderWelcomeChips() {
    const container = document.getElementById("welcomeChips");
    container.innerHTML = EXAMPLE_QUERIES.map((q, i) => `
        <div class="welcome-chip" onclick="useExample(${i})">${q.icon} ${q.label}</div>
    `).join("");
}

function useExample(index) {
    const q = EXAMPLE_QUERIES[index];
    const input = document.getElementById("queryInput");
    input.value = q.text;
    autoResize(input);
    input.focus();
    // Close sidebar on mobile
    if (window.innerWidth <= 768) toggleSidebar();
}

// ============================================================
// Health Check
// ============================================================
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        
        const dot = document.getElementById("statusDot");
        const text = document.getElementById("statusText");
        const mcpDot = document.getElementById("mcpStatusDot");
        const mcpText = document.getElementById("mcpStatusText");
        const mcpTools = document.getElementById("mcpToolsText");

        if (data.status === "ok") {
            dot.classList.remove("offline");
            text.textContent = "Online";
            
            if (data.mcp_connected) {
                mcpDot.classList.remove("offline");
                mcpText.textContent = "MCP Connected";
                mcpTools.textContent = `Tools: ${(data.mcp_tools || []).join(", ")}`;
            } else {
                mcpDot.classList.add("offline");
                mcpText.textContent = "MCP Disconnected";
                mcpTools.textContent = "";
            }
        }
    } catch (e) {
        document.getElementById("statusDot").classList.add("offline");
        document.getElementById("statusText").textContent = "Offline";
        document.getElementById("mcpStatusDot").classList.add("offline");
        document.getElementById("mcpStatusText").textContent = "Backend unreachable";
    }
}

// ============================================================
// Sidebar Toggle
// ============================================================
function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const backdrop = document.getElementById("sidebarBackdrop");
    sidebar.classList.toggle("open");
    backdrop.classList.toggle("active");
}

// ============================================================
// Input Handling
// ============================================================
function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoResize(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
}

// ============================================================
// Session Update
// ============================================================
function updateSession() {
    const pct = (state.queryCount / MAX_QUERIES) * 100;
    document.getElementById("sessionBarFill").style.width = pct + "%";
    document.getElementById("sessionCount").textContent = `${state.queryCount} / ${MAX_QUERIES} queries used`;
}

// ============================================================
// Send Message
// ============================================================
async function sendMessage() {
    const input = document.getElementById("queryInput");
    const query = input.value.trim();
    
    if (!query || state.isProcessing) return;
    
    if (state.queryCount >= MAX_QUERIES) {
        addAssistantMessage("🛑 You have reached the maximum limit of 5 queries per session. Refresh the page to start a new session.", {});
        return;
    }
    
    // Hide welcome screen
    const welcome = document.getElementById("welcomeScreen");
    if (welcome) welcome.classList.add("hidden");
    
    // Add user message
    addUserMessage(query);
    
    // Clear input
    input.value = "";
    input.style.height = "auto";
    
    // Show typing indicator
    state.isProcessing = true;
    updateSendButton();
    showTypingIndicator();
    
    const startTime = Date.now();
    
    try {
        const res = await fetch(`${API_BASE}/api/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: state.userId, query: query }),
        });
        
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        
        removeTypingIndicator();
        
        if (res.ok) {
            const data = await res.json();
            state.queryCount++;
            updateSession();
            
            state.history.unshift({
                query,
                answer: data.answer,
                confidence: data.confidence || 0,
                intent: data.intent || "general",
                trace: data.trace || [],
                time: elapsed,
            });
            if (state.history.length > 5) state.history.pop();
            
            addAssistantMessage(data.answer, {
                confidence: data.confidence || 0,
                intent: data.intent || "general",
                trace: data.trace || [],
                elapsed,
            });
        } else if (res.status === 400) {
            const err = await res.json();
            addAssistantMessage(`❌ Bad request: ${err.detail || "Unknown error"}`, {});
        } else {
            const text = await res.text();
            addAssistantMessage(`❌ Server error (${res.status}): ${text.substring(0, 300)}`, {});
        }
    } catch (e) {
        removeTypingIndicator();
        if (e.name === "TypeError" && e.message.includes("fetch")) {
            addAssistantMessage("❌ Backend not running. Please start it with: `python main.py`", {});
        } else {
            addAssistantMessage(`❌ Error: ${e.message || "Unknown error"}`, {});
        }
    }
    
    state.isProcessing = false;
    updateSendButton();
}

function updateSendButton() {
    const btn = document.getElementById("sendBtn");
    btn.disabled = state.isProcessing;
}

// ============================================================
// Message Rendering
// ============================================================
function addUserMessage(text) {
    const container = document.getElementById("chatContainer");
    const div = document.createElement("div");
    div.className = "message user";
    div.innerHTML = `
        <div class="message-avatar">👤</div>
        <div class="message-content">
            <div class="message-bubble">${escapeHtml(text)}</div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function addAssistantMessage(text, meta) {
    const container = document.getElementById("chatContainer");
    const div = document.createElement("div");
    div.className = "message assistant";
    
    let metaHtml = "";
    if (meta && meta.confidence !== undefined) {
        const confClass = meta.confidence >= 70 ? "high" : meta.confidence >= 40 ? "medium" : "low";
        const confColor = meta.confidence >= 70 ? "#10b981" : meta.confidence >= 40 ? "#f59e0b" : "#ef4444";
        
        metaHtml = `
            <div class="message-meta">
                <span class="meta-badge intent">🎯 ${meta.intent}</span>
                <span class="meta-badge confidence ${confClass}">
                    <span class="confidence-bar-inline">
                        <span class="confidence-bar-track">
                            <span class="confidence-bar-fill" style="width:${meta.confidence}%; background:${confColor};"></span>
                        </span>
                        ${meta.confidence}%
                    </span>
                </span>
                <span class="meta-badge time">⏱️ ${meta.elapsed}s</span>
                ${meta.trace && meta.trace.length > 0 ? `
                    <span class="trace-toggle" onclick="toggleTrace(this)">🔍 Trace (${meta.trace.length})</span>
                ` : ""}
            </div>
            ${meta.trace && meta.trace.length > 0 ? `
                <div class="trace-content">
                    <ul class="trace-list">
                        ${meta.trace.map(t => `<li>${escapeHtml(String(t))}</li>`).join("")}
                    </ul>
                </div>
            ` : ""}
        `;
    }
    
    div.innerHTML = `
        <div class="message-avatar" style="background: rgba(99, 102, 241, 0.2); border: 1px solid rgba(99, 102, 241, 0.3);">🤖</div>
        <div class="message-content">
            <div class="message-bubble">${renderMarkdown(text)}</div>
            ${metaHtml}
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function showTypingIndicator() {
    const container = document.getElementById("chatContainer");
    const div = document.createElement("div");
    div.className = "typing-indicator";
    div.id = "typingIndicator";
    div.innerHTML = `
        <div class="message-avatar" style="width:36px;height:36px;border-radius:10px;background:rgba(99,102,241,0.2);border:1px solid rgba(99,102,241,0.3);display:flex;align-items:center;justify-content:center;font-size:16px;">🤖</div>
        <div class="typing-dots">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    container.appendChild(div);
    scrollToBottom();
}

function removeTypingIndicator() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
}

function toggleTrace(el) {
    const content = el.closest(".message-content").querySelector(".trace-content");
    if (content) content.classList.toggle("open");
}

function scrollToBottom() {
    const container = document.getElementById("chatContainer");
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ============================================================
// Markdown Renderer (Lightweight)
// ============================================================
function renderMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    
    // Headers
    html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
    html = html.replace(/^## (.+)$/gm, "<h2>$1</h2>");
    html = html.replace(/^# (.+)$/gm, "<h1>$1</h1>");
    
    // Bold and italic
    html = html.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
    
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>');
    html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
    
    // Unordered lists
    html = html.replace(/^[\s]*[-•] (.+)$/gm, "<li>$1</li>");
    html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");
    
    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");
    
    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    // Line breaks → paragraphs
    html = html.replace(/\n\n/g, "</p><p>");
    html = html.replace(/\n/g, "<br>");
    html = "<p>" + html + "</p>";
    
    // Clean up empty paragraphs
    html = html.replace(/<p>\s*<\/p>/g, "");
    html = html.replace(/<p>\s*(<h[1-3]>)/g, "$1");
    html = html.replace(/(<\/h[1-3]>)\s*<\/p>/g, "$1");
    html = html.replace(/<p>\s*(<ul>)/g, "$1");
    html = html.replace(/(<\/ul>)\s*<\/p>/g, "$1");
    html = html.replace(/<p>\s*(<pre>)/g, "$1");
    html = html.replace(/(<\/pre>)\s*<\/p>/g, "$1");
    
    return html;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
