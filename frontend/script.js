// Base API URL (same origin as the UI)
const API_URL = window.location.origin;

// DOM Elements
const repoUrl = document.getElementById('repoUrl');
const ingestBtn = document.getElementById('ingestBtn');
const ingestStatus = document.getElementById('ingestStatus');
const queryInput = document.getElementById('queryInput');
const askBtn = document.getElementById('askBtn');
const queryStatus = document.getElementById('queryStatus');
const resultsContainer = document.getElementById('results');

// State
let repoLoaded = false;

// Event Listeners
ingestBtn.addEventListener('click', handleIngest);
askBtn.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleQuery();
});
repoUrl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') handleIngest();
});

// Handle Repository Ingestion
async function handleIngest() {
    const url = repoUrl.value.trim();
    
    if (!url) {
        showStatus(ingestStatus, 'Please enter a repository URL', 'error');
        return;
    }

    showStatus(ingestStatus, 'Loading repository... <span class="loading-spinner"></span>', 'info');
    ingestBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/ingest?repo_url=${encodeURIComponent(url)}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        const data = await response.json();

        if (data.status === 'success') {
            repoLoaded = true;
            showStatus(
                ingestStatus,
                `✓ Repository loaded! Indexed ${data.chunks_indexed} chunks.`,
                'success'
            );
            resultsContainer.innerHTML = `
                <div class="result-item">
                    <h3>✓ Repository Ready</h3>
                    <p>You can now ask questions about the codebase.</p>
                </div>
            `;
        } else {
            throw new Error(data.message || 'Failed to load repository');
        }
    } catch (error) {
        showStatus(ingestStatus, `Error: ${error.message}`, 'error');
        repoLoaded = false;
    } finally {
        ingestBtn.disabled = false;
    }
}

// Handle Query
async function handleQuery() {
    const query = queryInput.value.trim();
    const url = repoUrl.value.trim();

    if (!query) {
        showStatus(queryStatus, 'Please enter a question', 'error');
        return;
    }

    if (!repoLoaded) {
        showStatus(queryStatus, 'Please load a repository first', 'error');
        return;
    }

    showStatus(queryStatus, 'Searching... <span class="loading-spinner"></span>', 'info');
    askBtn.disabled = true;

    try {
        const response = await fetch(`${API_URL}/ask?repo_url=${encodeURIComponent(url)}&query=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.status === 'success') {
            showStatus(queryStatus, '✓ Search complete', 'success');
            displayResults(query, data.results);
        } else {
            throw new Error(data.message || 'Query failed');
        }
    } catch (error) {
        showStatus(queryStatus, `Error: ${error.message}`, 'error');
    } finally {
        askBtn.disabled = false;
    }
}

// Display Results
function displayResults(query, results) {
    let html = '';

    // Query Section
    html += `
        <div class="result-item">
            <h3>📝 Your Question</h3>
            <p><code>${escapeHtml(query)}</code></p>
        </div>
    `;

    // Answer Section
    if (results.explanation) {
        html += `
            <div class="result-item answer">
                <h3>💡 AI Response</h3>
                <pre style="white-space: pre-wrap; line-height: 1.45;">${escapeHtml(results.explanation)}</pre>
            </div>
        `;
    }

    // Sources Section
    if (results.sources && results.sources.length > 0) {
        html += `
            <div class="result-item">
                <h3>📚 Source Files (${results.sources.length})</h3>
        `;
        results.sources.forEach((source, idx) => {
            html += `
                <p><strong>Source ${idx + 1}:</strong></p>
                <p>
                  <code>${escapeHtml(source.source_id || `S${idx + 1}`)}</code>
                  File: <code>${escapeHtml(source.path)}</code>
                  ${source.start_line ? ` (lines ${escapeHtml(String(source.start_line))}-${escapeHtml(String(source.end_line))})` : ''}
                </p>
                <pre style="margin-top: 8px; font-size: 0.95em; color: #718096; white-space: pre-wrap;">${escapeHtml((source.snippet || '').substring(0, 600))}</pre>
            `;
        });
        html += `</div>`;
    }

    resultsContainer.innerHTML = html;
}

// Show Status Message
function showStatus(element, message, type) {
    element.innerHTML = message;
    element.className = `status-message ${type}`;
}

// Escape HTML Special Characters
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize
window.addEventListener('load', () => {
    console.log('Codebase RAG Agent ready!');
    showStatus(ingestStatus, 'Ready to load repository', 'info');
});
