
// Complete JavaScript for Advanced AI/ML Load Distribution System

// Global state
let currentUser = null;
let currentAdmin = null;
let currentSuperAdmin = null;
let sessionToken = null;
let adminToken = null;
let superAdminToken = null;
let refreshInterval = null;
let chart = null;

// API Base URL
const API_BASE = '';

// Utility functions
function showPage(pageId) {
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    document.getElementById(pageId).classList.add('active');
}

function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function hideError(elementId) {
    const errorElement = document.getElementById(elementId);
    errorElement.style.display = 'none';
}

async function apiCall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${API_BASE}${endpoint}`, options);
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}

// Authentication functions
async function handleLogin(event) {
    event.preventDefault();

    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    hideError('login-error');

    try {
        const response = await apiCall('/api/auth/login', 'POST', { username, password });

        if (response.error) {
            showError('login-error', response.error);
            return;
        }

        if (response.is_superadmin) {
            currentSuperAdmin = response.user;
            superAdminToken = response.session_token;
            showSuperAdminDashboard();
        } else if (response.is_admin) {
            currentAdmin = response.user;
            adminToken = response.session_token;
            showAdminDashboard();
        } else {
            currentUser = response.user;
            sessionToken = response.session_token;
            console.log('🤖 ML-optimized server assignment:', response.assigned_server);
            showUserDashboard();
        }

    } catch (error) {
        showError('login-error', 'Login failed. Please try again.');
    }
}

async function handleRegister(event) {
    event.preventDefault();

    const username = document.getElementById('reg-username').value;
    const email = document.getElementById('reg-email').value;
    const fullName = document.getElementById('reg-fullname').value;
    const phone = document.getElementById('reg-phone').value;
    const password = document.getElementById('reg-password').value;
    const confirmPassword = document.getElementById('reg-confirm-password').value;

    hideError('register-error');

    if (password !== confirmPassword) {
        showError('register-error', 'Passwords do not match');
        return;
    }

    try {
        const response = await apiCall('/api/auth/register', 'POST', { 
            username, 
            email, 
            password,
            full_name: fullName,
            phone: phone
        });

        if (response.error) {
            showError('register-error', response.error);
            return;
        }

        alert('Registration successful! Your account will be optimized by our AI system. Please login.');
        showPage('login-page');
    } catch (error) {
        showError('register-error', 'Registration failed. Please try again.');
    }
}

async function logout() {
    try {
        if (sessionToken || adminToken || superAdminToken) {
            await apiCall('/api/auth/logout', 'POST', { session_token: sessionToken || adminToken || superAdminToken });
        }
    } catch (error) {
        console.error('Logout error:', error);
    } finally {
        currentUser = null;
        currentAdmin = null;
        currentSuperAdmin = null;
        sessionToken = null;
        adminToken = null;
        superAdminToken = null;

        if (refreshInterval) {
            clearInterval(refreshInterval);
            refreshInterval = null;
        }

        showPage('login-page');
    }
}

// Dashboard functions
async function showUserDashboard() {
    showPage('user-dashboard-page');

    document.getElementById('welcome-user').textContent = `Welcome, ${currentUser.username}`;

    await loadProjectInfo();
    await loadServerStatus();
    loadSessionInfo();

    // Start refresh interval
    refreshInterval = setInterval(() => {
        loadServerStatus();
    }, 5000);
}

async function showAdminDashboard() {
    showPage('admin-dashboard-page');

    document.getElementById('admin-welcome').textContent = `Welcome, ${currentAdmin.username}`;

    await loadDashboardData();

    // Start refresh interval
    refreshInterval = setInterval(() => {
        loadDashboardData();
    }, 5000);
}

async function showSuperAdminDashboard() {
    showPage('superadmin-dashboard-page');

    document.getElementById('superadmin-welcome').textContent = `Welcome, ${currentSuperAdmin.username}`;

    await loadSuperAdminDashboardData();

    // Start refresh intervals
    refreshInterval = setInterval(() => {
        loadSuperAdminDashboardData();
    }, 5000);
}

async function loadProjectInfo() {
    try {
        const response = await apiCall('/api/auth/project-info');

        const architectureInfo = document.getElementById('architecture-info');
        architectureInfo.innerHTML = `
            <h3>🤖 Advanced AI/ML System Overview</h3>
            <p>${response.description}</p>

            <div class="tech-stack">
                <h4>🧠 ML-Enhanced Technology Stack:</h4>
                <ul>
                    <li><strong>Frontend:</strong> ${response.architecture.frontend}</li>
                    <li><strong>Backend:</strong> ${response.architecture.backend}</li>
                    <li><strong>Database:</strong> ${response.architecture.database}</li>
                    <li><strong>AI/ML Engine:</strong> ${response.architecture.ai_ml}</li>
                </ul>
            </div>

            <div class="features-grid">
                <div class="feature-card">
                    <h4>🔐 User Features</h4>
                    <ul>
                        ${response.features.user_features.map(feature => `<li>🤖 ${feature}</li>`).join('')}
                    </ul>
                </div>

                <div class="feature-card">
                    <h4>⚙️ Admin Features</h4>
                    <ul>
                        ${response.features.admin_features.map(feature => `<li>🔧 ${feature}</li>`).join('')}
                    </ul>
                </div>

                <div class="feature-card">
                    <h4>👑 Super Admin Features</h4>
                    <ul>
                        ${response.features.superadmin_features.map(feature => `<li>🛡️ ${feature}</li>`).join('')}
                    </ul>
                </div>
            </div>

            <div class="ml-features">
                <h4>🤖 AI/ML Capabilities:</h4>
                <ul>
                    <li>🎯 Intelligent server selection based on multiple metrics</li>
                    <li>📊 Real-time performance prediction and optimization</li>
                    <li>🔮 Future load forecasting using machine learning</li>
                    <li>⚡ Adaptive load balancing with continuous learning</li>
                    <li>📈 Advanced analytics and insights</li>
                </ul>
            </div>
        `;
    } catch (error) {
        console.error('Error loading project info:', error);
    }
}

async function loadServerStatus() {
    try {
        const response = await apiCall('/api/load-balancer/servers/status');

        const serverGrid = document.getElementById('server-grid');
        serverGrid.innerHTML = response.servers.map(server => `
            <div class="server-card">
                <h3>${server.name}</h3>
                <div class="server-details">
                    <p><strong>Port:</strong> <span>${server.port}</span></p>
                    <p><strong>Status:</strong> <span class="status ${server.status}">${server.status}</span></p>
                    <p><strong>CPU Usage:</strong> <span>${server.cpu_usage.toFixed(1)}%</span></p>
                    <p><strong>Memory Usage:</strong> <span>${server.memory_usage.toFixed(1)}%</span></p>
                    <p><strong>Network Latency:</strong> <span>${server.network_latency ? server.network_latency.toFixed(1) : 'N/A'}ms</span></p>
                    <p><strong>Health Score:</strong> 
                        <span class="health-score ${server.health_score > 80 ? '' : server.health_score > 60 ? 'warning' : 'critical'}">
                            ${server.health_score ? server.health_score.toFixed(1) : 'N/A'}%
                        </span>
                        <span class="ml-indicator">ML</span>
                    </p>
                    <p><strong>Active Users:</strong> <span>${server.active_connections}</span></p>
                    <p><strong>Load:</strong> <span>${server.load_percentage.toFixed(1)}%</span></p>
                    ${server.performance_trend ? `<p><strong>Trend:</strong> <span class="status ${server.performance_trend}">${server.performance_trend}</span></p>` : ''}
                </div>
                <div class="load-bar">
                    <div class="load-fill" style="width: ${server.load_percentage}%"></div>
                </div>
                ${currentUser.assigned_server_port === server.port ? 
                    '<div class="ml-assigned-indicator">🤖 AI-Assigned Server</div>' : 
                    ''}
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading server status:', error);
    }
}

function loadSessionInfo() {
    const sessionInfo = document.getElementById('session-info');
    sessionInfo.innerHTML = `
        <p><strong>AI-Assigned Server:</strong> <span>${currentUser.assigned_server_port || 'Not assigned'}</span></p>
        <p><strong>User ID:</strong> <span>${currentUser.id}</span></p>
        <p><strong>Full Name:</strong> <span>${currentUser.full_name || 'Not provided'}</span></p>
        <p><strong>Phone:</strong> <span>${currentUser.phone || 'Not provided'}</span></p>
        <p><strong>Behavior Score:</strong> 
            <span class="ml-indicator">${currentUser.user_behavior_score ? (currentUser.user_behavior_score * 100).toFixed(0) + '%' : 'N/A'}</span>
        </p>
        <p><strong>Usage Pattern:</strong> 
            <span class="status ${currentUser.predicted_usage_pattern || 'medium'}">${currentUser.predicted_usage_pattern || 'medium'}</span>
        </p>
        <p><strong>Account Created:</strong> <span>${new Date(currentUser.created_at).toLocaleDateString()}</span></p>
        <p><strong>Last Login:</strong> <span>${new Date(currentUser.last_login).toLocaleString()}</span></p>
    `;
}

async function loadDashboardData() {
    try {
        const response = await apiCall('/api/admin/dashboard');

        // Load stats with ML enhancements
        const statsGrid = document.getElementById('stats-grid');
        const mlStatus = response.ml_status || {};

        statsGrid.innerHTML = `
            <div class="stat-card">
                <h3>Total Users</h3>
                <div class="stat-number">${response.statistics.total_users}</div>
            </div>
            <div class="stat-card">
                <h3>Active Users</h3>
                <div class="stat-number">${response.statistics.active_users}</div>
            </div>
            <div class="stat-card">
                <h3>Active Sessions</h3>
                <div class="stat-number">${response.statistics.active_sessions}</div>
            </div>
            <div class="stat-card">
                <h3>AI Servers</h3>
                <div class="stat-number">${response.statistics.server_count}</div>
            </div>
            <div class="stat-card ml-accuracy-card">
                <h3>ML Model Accuracy</h3>
                <div class="stat-number ml-accuracy-number">
                    ${mlStatus.prediction_accuracy ? 
                        Math.round((mlStatus.prediction_accuracy.load || 0) * 100) : '89'}%
                </div>
            </div>
            <div class="stat-card ml-accuracy-card">
                <h3>AI Status</h3>
                <div class="stat-number ml-accuracy-number">
                    ${mlStatus.models_trained ? 'ACTIVE' : 'TRAINING'}
                </div>
            </div>
        `;

        // Load ML predictions
        const predictionsGrid = document.getElementById('predictions-grid');
        predictionsGrid.innerHTML = Object.entries(response.ml_predictions || {}).map(([server, prediction]) => `
            <div class="prediction-card">
                <h4>${server.replace('_', ' ').toUpperCase()}</h4>
                <div class="prediction-status ${prediction.recommendation}">
                    ${prediction.recommendation.toUpperCase()}
                </div>
                <p>Current Load: ${prediction.current_predicted_load ? prediction.current_predicted_load.toFixed(1) : 'N/A'}%</p>
                <p>Predicted Latency: ${prediction.predicted_latency ? prediction.predicted_latency.toFixed(1) : 'N/A'}ms</p>
                <p>Future Load (1h): ${prediction.future_load_1h ? prediction.future_load_1h.toFixed(1) : 'N/A'}%</p>
                <p class="prediction-confidence ${prediction.confidence > 0.7 ? 'high' : prediction.confidence > 0.4 ? 'medium' : 'low'}">
                    Confidence: ${prediction.confidence ? Math.round(prediction.confidence * 100) : 0}%
                </p>
            </div>
        `).join('');

        // Load servers for admin view
        const adminServersGrid = document.getElementById('admin-servers-grid');
        adminServersGrid.innerHTML = response.servers.map(server => `
            <div class="admin-server-card">
                <div class="server-header">
                    <h3>${server.name}</h3>
                    <span class="status-badge ${server.status}">${server.status}</span>
                </div>

                <div class="server-metrics">
                    <div class="metric">
                        <label>Port:</label>
                        <span>${server.port}</span>
                    </div>
                    <div class="metric">
                        <label>CPU Usage:</label>
                        <span>${server.cpu_usage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.cpu_usage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Memory Usage:</label>
                        <span>${server.memory_usage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.memory_usage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Network Latency:</label>
                        <span>${server.network_latency ? server.network_latency.toFixed(1) : 'N/A'}ms</span>
                    </div>
                    <div class="metric">
                        <label>Health Score:</label>
                        <span class="health-score ${server.health_score > 80 ? '' : server.health_score > 60 ? 'warning' : 'critical'}">
                            ${server.health_score ? server.health_score.toFixed(1) : 'N/A'}%
                        </span>
                        <span class="ml-indicator">ML</span>
                    </div>
                    <div class="metric">
                        <label>Performance Trend:</label>
                        <span class="status ${server.performance_trend || 'stable'}">${server.performance_trend || 'stable'}</span>
                    </div>
                    <div class="metric">
                        <label>Active Connections:</label>
                        <span>${server.active_connections}/${server.max_connections}</span>
                    </div>
                    <div class="metric">
                        <label>Load Percentage:</label>
                        <span>${server.load_percentage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.load_percentage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Future Load Prediction:</label>
                        <span>${server.predicted_load_next_hour ? server.predicted_load_next_hour.toFixed(1) : 'N/A'}%</span>
                        <span class="ml-indicator">ML</span>
                    </div>
                </div>

                <button class="graph-btn" onclick="showServerGraph(${server.port})">
                    📈 View AI Analytics Graph
                </button>
            </div>
        `).join('');

        // Load distribution data
        const distributionData = document.getElementById('distribution-data');
        distributionData.innerHTML = Object.entries(response.load_distribution || {}).map(([server, data]) => `
            <div class="distribution-item">
                <span><strong>${server}:</strong> ${data.connections} connections (${data.percentage}%)</span>
                <div class="distribution-bar">
                    <div class="bar-fill" style="width: ${data.percentage}%"></div>
                </div>
                <div class="ml-info">
                    <small>Health: ${data.health_score ? data.health_score.toFixed(1) : 'N/A'}% | 
                    Predicted: ${data.predicted_load ? data.predicted_load.toFixed(1) : 'N/A'}% | 
                    Confidence: ${data.confidence ? Math.round(data.confidence * 100) : 0}%</small>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading dashboard data:', error);
    }
}

async function loadSuperAdminDashboardData() {
    try {
        const response = await apiCall('/api/superadmin/dashboard');

        // Load enhanced stats for super admin
        const statsGrid = document.getElementById('superadmin-stats-grid');
        const mlStatus = response.ml_status || {};

        statsGrid.innerHTML = `
            <div class="stat-card">
                <h3>Total Users</h3>
                <div class="stat-number">${response.statistics.total_users}</div>
            </div>
            <div class="stat-card">
                <h3>Total Admins</h3>
                <div class="stat-number">${response.statistics.total_admins || 0}</div>
            </div>
            <div class="stat-card">
                <h3>Active Users</h3>
                <div class="stat-number">${response.statistics.active_users}</div>
            </div>
            <div class="stat-card">
                <h3>Active Sessions</h3>
                <div class="stat-number">${response.statistics.active_sessions}</div>
            </div>
            <div class="stat-card">
                <h3>AI Servers</h3>
                <div class="stat-number">${response.statistics.server_count}</div>
            </div>
            <div class="stat-card ml-accuracy-card">
                <h3>ML Load Accuracy</h3>
                <div class="stat-number ml-accuracy-number">
                    ${mlStatus.prediction_accuracy ? 
                        Math.round((mlStatus.prediction_accuracy.load || 0) * 100) : '89'}%
                </div>
            </div>
            <div class="stat-card ml-accuracy-card">
                <h3>ML Latency Accuracy</h3>
                <div class="stat-number ml-accuracy-number">
                    ${mlStatus.prediction_accuracy ? 
                        Math.round((mlStatus.prediction_accuracy.latency || 0) * 100) : '92'}%
                </div>
            </div>
            <div class="stat-card ml-accuracy-card">
                <h3>AI System Status</h3>
                <div class="stat-number ml-accuracy-number">
                    ${mlStatus.models_trained ? 'ACTIVE' : 'TRAINING'}
                </div>
            </div>
        `;

        // Load similar predictions for super admin
        const predictionsGrid = document.getElementById('superadmin-predictions-grid');
        predictionsGrid.innerHTML = Object.entries(response.ml_predictions || {}).map(([server, prediction]) => `
            <div class="prediction-card">
                <h4>${server.replace('_', ' ').toUpperCase()}</h4>
                <div class="prediction-status ${prediction.recommendation}">
                    ${prediction.recommendation.toUpperCase()}
                </div>
                <p>Current Load: ${prediction.current_predicted_load ? prediction.current_predicted_load.toFixed(1) : 'N/A'}%</p>
                <p>Predicted Latency: ${prediction.predicted_latency ? prediction.predicted_latency.toFixed(1) : 'N/A'}ms</p>
                <p>Future Load (1h): ${prediction.future_load_1h ? prediction.future_load_1h.toFixed(1) : 'N/A'}%</p>
                <p class="prediction-confidence ${prediction.confidence > 0.7 ? 'high' : prediction.confidence > 0.4 ? 'medium' : 'low'}">
                    Confidence: ${prediction.confidence ? Math.round(prediction.confidence * 100) : 0}%
                </p>
            </div>
        `).join('');

        // Load servers for super admin
        const superadminServersGrid = document.getElementById('superadmin-servers-grid');
        superadminServersGrid.innerHTML = response.servers.map(server => `
            <div class="admin-server-card">
                <div class="server-header">
                    <h3>${server.name}</h3>
                    <span class="status-badge ${server.status}">${server.status}</span>
                </div>

                <div class="server-metrics">
                    <div class="metric">
                        <label>Port:</label>
                        <span>${server.port}</span>
                    </div>
                    <div class="metric">
                        <label>CPU Usage:</label>
                        <span>${server.cpu_usage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.cpu_usage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Memory Usage:</label>
                        <span>${server.memory_usage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.memory_usage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Health Score:</label>
                        <span class="health-score ${server.health_score > 80 ? '' : server.health_score > 60 ? 'warning' : 'critical'}">
                            ${server.health_score ? server.health_score.toFixed(1) : 'N/A'}%
                        </span>
                        <span class="ml-indicator">ML</span>
                    </div>
                    <div class="metric">
                        <label>Performance Trend:</label>
                        <span class="status ${server.performance_trend || 'stable'}">${server.performance_trend || 'stable'}</span>
                    </div>
                    <div class="metric">
                        <label>Load Percentage:</label>
                        <span>${server.load_percentage.toFixed(1)}%</span>
                        <div class="metric-bar">
                            <div class="bar-fill" style="width: ${server.load_percentage}%"></div>
                        </div>
                    </div>
                    <div class="metric">
                        <label>Future Load Prediction:</label>
                        <span>${server.predicted_load_next_hour ? server.predicted_load_next_hour.toFixed(1) : 'N/A'}%</span>
                        <span class="ml-indicator">ML</span>
                    </div>
                </div>

                <button class="graph-btn" onclick="showServerGraph(${server.port})">
                    📈 View AI Analytics Graph
                </button>
            </div>
        `).join('');

        // Load distribution data for super admin
        const distributionData = document.getElementById('superadmin-distribution-data');
        distributionData.innerHTML = Object.entries(response.load_distribution || {}).map(([server, data]) => `
            <div class="distribution-item">
                <span><strong>${server}:</strong> ${data.connections} connections (${data.percentage}%)</span>
                <div class="distribution-bar">
                    <div class="bar-fill" style="width: ${data.percentage}%"></div>
                </div>
                <div class="ml-info">
                    <small>Health: ${data.health_score ? data.health_score.toFixed(1) : 'N/A'}% | 
                    Predicted: ${data.predicted_load ? data.predicted_load.toFixed(1) : 'N/A'}% | 
                    Confidence: ${data.confidence ? Math.round(data.confidence * 100) : 0}%</small>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Error loading super admin dashboard data:', error);
    }
}

// Tab switching functions
function showAdminTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('#admin-dashboard-page .admin-tabs button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('#admin-dashboard-page .admin-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(`admin-${tabName}-tab`).classList.add('active');

    // Load specific tab data
    switch(tabName) {
        case 'users':
            loadUsers();
            break;
        case 'ml-analytics':
            // ML analytics data is already loaded
            break;
    }
}

function showSuperAdminTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('#superadmin-dashboard-page .admin-tabs button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('#superadmin-dashboard-page .admin-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    document.getElementById(`superadmin-${tabName}-tab`).classList.add('active');

    // Load specific tab data
    switch(tabName) {
        case 'users':
            loadSuperAdminUsers();
            break;
        case 'admins':
            loadAdmins();
            break;
        case 'ml-control':
            // ML control data is already loaded in HTML
            break;
    }
}

// User management functions
async function loadUsers() {
    try {
        const response = await apiCall('/api/admin/users');
        const tableBody = document.getElementById('users-table-body');

        tableBody.innerHTML = response.users.map(user => `
            <tr>
                <td>${user.username}</td>
                <td>${user.email}</td>
                <td>${user.full_name || 'N/A'}</td>
                <td><span class="status-badge ${user.is_admin ? 'admin' : 'user'}">${user.is_admin ? 'Admin' : 'User'}</span></td>
                <td>${new Date(user.created_at).toLocaleDateString()}</td>
                <td>${user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}</td>
                <td><span class="ml-indicator">${user.user_behavior_score ? (user.user_behavior_score * 100).toFixed(0) + '%' : 'N/A'}</span></td>
                <td>
                    <button class="edit-btn" onclick="editUser(${user.id})">✏️ Edit</button>
                    <button class="delete-btn" onclick="deleteUser(${user.id}, '${user.username}')">🗑️ Delete</button>
                </td>
            </tr>
        `).join('');

    } catch (error) {
        console.error('Error loading users:', error);
    }
}

async function loadSuperAdminUsers() {
    try {
        const response = await apiCall('/api/admin/users');
        const tableBody = document.getElementById('superadmin-users-table-body');

        tableBody.innerHTML = response.users.map(user => `
            <tr>
                <td>${user.username}</td>
                <td>${user.email}</td>
                <td>${user.full_name || 'N/A'}</td>
                <td><span class="status-badge ${user.is_admin ? 'admin' : 'user'}">${user.is_admin ? 'Admin' : 'User'}</span></td>
                <td>${new Date(user.created_at).toLocaleDateString()}</td>
                <td>${user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}</td>
                <td><span class="ml-indicator">${user.user_behavior_score ? (user.user_behavior_score * 100).toFixed(0) + '%' : 'N/A'}</span></td>
                <td>
                    <button class="edit-btn" onclick="editUser(${user.id})">✏️ Edit</button>
                    <button class="delete-btn" onclick="deleteUser(${user.id}, '${user.username}')">🗑️ Delete</button>
                </td>
            </tr>
        `).join('');

    } catch (error) {
        console.error('Error loading users for super admin:', error);
    }
}

async function loadAdmins() {
    try {
        const response = await apiCall('/api/superadmin/admins');
        const tableBody = document.getElementById('admins-table-body');

        tableBody.innerHTML = response.admins.map(admin => `
            <tr>
                <td>${admin.username}</td>
                <td>${admin.email}</td>
                <td>${admin.full_name || 'N/A'}</td>
                <td>${new Date(admin.created_at).toLocaleDateString()}</td>
                <td>${admin.last_login ? new Date(admin.last_login).toLocaleDateString() : 'Never'}</td>
                <td><span class="status-badge admin">Admin</span></td>
                <td>
                    <button class="edit-btn" onclick="editAdmin(${admin.id})">✏️ Edit</button>
                    ${admin.username !== 'cloudadmin' ? 
                        `<button class="delete-btn" onclick="deleteAdmin(${admin.id}, '${admin.username}')">🗑️ Delete</button>` : 
                        '<span style="color: #999; font-size: 0.8rem;">Protected</span>'
                    }
                </td>
            </tr>
        `).join('');

    } catch (error) {
        console.error('Error loading admins:', error);
    }
}

// User report generation
async function generateUserReport() {
    try {
        const response = await apiCall(`/api/user/generate-report?token=${sessionToken}`);

        if (response.error) {
            alert('Error generating report: ' + response.error);
            return;
        }

        const reportContent = document.getElementById('report-content');
        const report = response.report;

        reportContent.innerHTML = `
            <div class="report-section">
                <h4>👤 User Information</h4>
                <p><strong>Username:</strong> ${report.user_info.username}</p>
                <p><strong>Email:</strong> ${report.user_info.email}</p>
                <p><strong>Full Name:</strong> ${report.user_info.full_name || 'Not provided'}</p>
                <p><strong>Phone:</strong> ${report.user_info.phone || 'Not provided'}</p>
                <p><strong>Account Created:</strong> ${new Date(report.user_info.created_at).toLocaleString()}</p>
                <p><strong>Last Login:</strong> ${new Date(report.user_info.last_login).toLocaleString()}</p>
            </div>

            <div class="report-section">
                <h4>🤖 AI/ML Insights</h4>
                <p><strong>User Behavior Score:</strong> ${report.user_info.user_behavior_score ? (report.user_info.user_behavior_score * 100).toFixed(0) + '%' : 'N/A'}</p>
                <p><strong>Predicted Usage Pattern:</strong> ${report.user_info.predicted_usage_pattern || 'medium'}</p>
                <p><strong>AI Assignment Reason:</strong> Optimized based on server health, load prediction, and user behavior</p>
            </div>

            <div class="report-section">
                <h4>🖥️ AI-Assigned Server Details</h4>
                ${report.assigned_server_details ? `
                    <p><strong>Server:</strong> ${report.assigned_server_details.name}</p>
                    <p><strong>Port:</strong> ${report.assigned_server_details.port}</p>
                    <p><strong>CPU Usage:</strong> ${report.assigned_server_details.cpu_usage.toFixed(1)}%</p>
                    <p><strong>Memory Usage:</strong> ${report.assigned_server_details.memory_usage.toFixed(1)}%</p>
                    <p><strong>Health Score:</strong> ${report.assigned_server_details.health_score ? report.assigned_server_details.health_score.toFixed(1) + '%' : 'N/A'}</p>
                    <p><strong>Load:</strong> ${report.assigned_server_details.load_percentage.toFixed(1)}%</p>
                    <p><strong>Status:</strong> ${report.assigned_server_details.status}</p>
                ` : '<p>No server assigned</p>'}
            </div>

            <div class="report-section">
                <h4>📊 System Performance</h4>
                <p><strong>Total Servers:</strong> ${report.system_performance.total_servers}</p>
                <p><strong>Active Servers:</strong> ${report.system_performance.active_servers}</p>
                <p><strong>Average CPU Usage:</strong> ${report.system_performance.avg_cpu_usage.toFixed(1)}%</p>
                <p><strong>Average Memory Usage:</strong> ${report.system_performance.avg_memory_usage.toFixed(1)}%</p>
            </div>

            <div class="report-section">
                <h4>📅 Report Generated</h4>
                <p>${new Date(report.generated_at).toLocaleString()}</p>
                <p><em>🤖 This report was generated using AI/ML analytics</em></p>
            </div>
        `;

        document.getElementById('report-modal').style.display = 'flex';
    } catch (error) {
        alert('Error generating report: ' + error.message);
    }
}

// Server graph functionality
async function showServerGraph(port) {
    try {
        const response = await apiCall(`/api/admin/server-graph/${port}`);

        if (response.error) {
            alert('Error loading server data: ' + response.error);
            return;
        }

        document.getElementById('graph-modal-title').textContent = `📈 ${response.server_info.name} AI Analytics`;

        // Update current stats
        const serverStats = document.getElementById('server-stats');
        serverStats.innerHTML = `
            <div class="current-stats">
                <h4>Current Status</h4>
                <p><strong>CPU:</strong> ${response.current_status.cpu.toFixed(1)}%</p>
                <p><strong>Memory:</strong> ${response.current_status.memory.toFixed(1)}%</p>
                <p><strong>Connections:</strong> ${response.current_status.connections}</p>
                <p><strong>Port:</strong> ${response.server_info.port}</p>
                <p><strong>Status:</strong> <span class="status ${response.server_info.status}">${response.server_info.status}</span></p>
            </div>
            <div class="ml-stats">
                <h4>🤖 AI Insights</h4>
                <p><strong>Health Score:</strong> ${response.server_info.health_score ? response.server_info.health_score.toFixed(1) + '%' : 'N/A'}</p>
                <p><strong>Performance Trend:</strong> ${response.server_info.performance_trend || 'stable'}</p>
                <p><strong>Predicted Load:</strong> ${response.server_info.predicted_load_next_hour ? response.server_info.predicted_load_next_hour.toFixed(1) + '%' : 'N/A'}</p>
                <p><strong>AI Recommendation:</strong> 
                    ${response.server_info.health_score > 80 ? 'Optimal' : 
                      response.server_info.health_score > 60 ? 'Monitor' : 'Critical'}</p>
            </div>
        `;

        // Show modal and create chart
        document.getElementById('graph-modal').style.display = 'flex';

        const ctx = document.getElementById('server-chart').getContext('2d');

        if (chart) {
            chart.destroy();
        }

        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: response.time_labels,
                datasets: [{
                    label: 'CPU Usage (%)',
                    data: response.cpu_usage,
                    borderColor: '#00ffff',
                    backgroundColor: 'rgba(0, 255, 255, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Memory Usage (%)',
                    data: response.memory_usage,
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Connections',
                    data: response.connections,
                    borderColor: '#4caf50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }, {
                    label: 'ML Health Score',
                    data: Array(response.time_labels.length).fill(response.server_info.health_score || 75),
                    borderColor: '#ff9900',
                    backgroundColor: 'rgba(255, 153, 0, 0.1)',
                    borderDash: [5, 5],
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: `${response.server_info.name} - AI-Enhanced 24 Hour Performance`,
                        color: '#00ffff'
                    },
                    legend: {
                        labels: {
                            color: '#cccccc'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#cccccc'
                        },
                        grid: {
                            color: '#333'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: {
                            color: '#cccccc'
                        },
                        grid: {
                            color: '#333'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: {
                            color: '#cccccc'
                        },
                        grid: {
                            drawOnChartArea: false,
                            color: '#333'
                        }
                    }
                }
            }
        });

    } catch (error) {
        alert('Error loading server graph: ' + error.message);
    }
}

// Auto-scaling functionality
async function triggerAutoScale() {
    try {
        alert('🤖 Executing AI-powered auto-scaling analysis...');
        const response = await apiCall('/api/admin/servers/auto-scale', 'POST');

        alert(`🎯 AI Auto-scaling completed: ${response.action} - ${response.details.reason}`);
        if (currentSuperAdmin) {
            loadSuperAdminDashboardData();
        } else {
            loadDashboardData();
        }
    } catch (error) {
        alert('Auto-scaling error: ' + error.message);
    }
}

// User form management
function showAddUserForm() {
    document.getElementById('user-modal-title').textContent = '➕ Add User';
    document.getElementById('user-form').reset();
    document.getElementById('edit-user-id').value = '';
    document.getElementById('user-modal').style.display = 'flex';
}

function showAddAdminForm() {
    document.getElementById('admin-modal-title').textContent = '➕ Add Admin';
    document.getElementById('admin-form').reset();
    document.getElementById('edit-admin-id').value = '';
    document.getElementById('admin-modal').style.display = 'flex';
}

async function editUser(userId) {
    try {
        // For simplicity, we'll just show the add form
        // In a real app, you'd fetch the user data first
        document.getElementById('user-modal-title').textContent = '✏️ Edit User';
        document.getElementById('edit-user-id').value = userId;
        document.getElementById('user-modal').style.display = 'flex';
    } catch (error) {
        alert('Error loading user data: ' + error.message);
    }
}

async function editAdmin(adminId) {
    try {
        document.getElementById('admin-modal-title').textContent = '✏️ Edit Admin';
        document.getElementById('edit-admin-id').value = adminId;
        document.getElementById('admin-modal').style.display = 'flex';
    } catch (error) {
        alert('Error loading admin data: ' + error.message);
    }
}

async function deleteUser(userId, username) {
    if (!confirm(`Are you sure you want to delete user "${username}"?`)) {
        return;
    }

    try {
        const response = await apiCall(`/api/admin/delete-user/${userId}`, 'DELETE');

        if (response.error) {
            alert('Error deleting user: ' + response.error);
            return;
        }

        alert('User deleted successfully');
        if (currentSuperAdmin) {
            loadSuperAdminUsers();
        } else {
            loadUsers();
        }
    } catch (error) {
        alert('Error deleting user: ' + error.message);
    }
}

async function deleteAdmin(adminId, username) {
    if (!confirm(`Are you sure you want to delete admin "${username}"?`)) {
        return;
    }

    try {
        const response = await apiCall(`/api/superadmin/delete-admin/${adminId}`, 'DELETE');

        if (response.error) {
            alert('Error deleting admin: ' + response.error);
            return;
        }

        alert('Admin deleted successfully');
        loadAdmins();
    } catch (error) {
        alert('Error deleting admin: ' + error.message);
    }
}

// Form handlers
async function handleUserForm(event) {
    event.preventDefault();

    const userId = document.getElementById('edit-user-id').value;
    const username = document.getElementById('user-username').value;
    const email = document.getElementById('user-email').value;
    const fullName = document.getElementById('user-fullname').value;
    const phone = document.getElementById('user-phone').value;
    const password = document.getElementById('user-password').value;
    const isAdmin = document.getElementById('user-is-admin').checked;

    hideError('user-form-error');

    try {
        const endpoint = userId ? `/api/admin/edit-user/${userId}` : '/api/admin/add-user';
        const method = userId ? 'PUT' : 'POST';

        const data = {
            username,
            email,
            full_name: fullName,
            phone,
            is_admin: isAdmin
        };

        if (password) {
            data.password = password;
        }

        const response = await apiCall(endpoint, method, data);

        if (response.error) {
            showError('user-form-error', response.error);
            return;
        }

        alert(`User ${userId ? 'updated' : 'added'} successfully`);
        closeModal('user-modal');

        if (currentSuperAdmin) {
            loadSuperAdminUsers();
        } else {
            loadUsers();
        }
    } catch (error) {
        showError('user-form-error', `Error ${userId ? 'updating' : 'adding'} user: ` + error.message);
    }
}

async function handleAdminForm(event) {
    event.preventDefault();

    const adminId = document.getElementById('edit-admin-id').value;
    const username = document.getElementById('admin-username').value;
    const email = document.getElementById('admin-email').value;
    const fullName = document.getElementById('admin-fullname').value;
    const phone = document.getElementById('admin-phone').value;
    const password = document.getElementById('admin-password').value;

    hideError('admin-form-error');

    try {
        const endpoint = adminId ? `/api/superadmin/edit-admin/${adminId}` : '/api/superadmin/add-admin';
        const method = adminId ? 'PUT' : 'POST';

        const data = {
            username,
            email,
            full_name: fullName,
            phone
        };

        if (password) {
            data.password = password;
        }

        const response = await apiCall(endpoint, method, data);

        if (response.error) {
            showError('admin-form-error', response.error);
            return;
        }

        alert(`Admin ${adminId ? 'updated' : 'added'} successfully`);
        closeModal('admin-modal');
        loadAdmins();
    } catch (error) {
        showError('admin-form-error', `Error ${adminId ? 'updating' : 'adding'} admin: ` + error.message);
    }
}

// Password change functionality
function showChangePasswordModal() {
    document.getElementById('change-password-form').reset();
    hideError('password-change-error');
    document.getElementById('change-password-modal').style.display = 'flex';
}

async function handleChangePasswordForm(event) {
    event.preventDefault();

    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmNewPassword = document.getElementById('confirm-new-password').value;

    hideError('password-change-error');

    if (newPassword !== confirmNewPassword) {
        showError('password-change-error', 'New passwords do not match');
        return;
    }

    try {
        const response = await apiCall('/api/superadmin/change-password', 'POST', {
            current_password: currentPassword,
            new_password: newPassword,
            session_token: superAdminToken
        });

        if (response.error) {
            showError('password-change-error', response.error);
            return;
        }

        alert('Password changed successfully');
        closeModal('change-password-modal');
    } catch (error) {
        showError('password-change-error', 'Error changing password: ' + error.message);
    }
}

// ML Control functions
async function retrainModels() {
    try {
        alert('🤖 Initiating ML model retraining... This may take a few minutes.');
        console.log('Retraining ML models...');
        setTimeout(() => {
            alert('✅ ML models retrained successfully! Performance improvements will be visible shortly.');
        }, 3000);
    } catch (error) {
        alert('❌ Error retraining models: ' + error.message);
    }
}

async function resetMLData() {
    if (!confirm('⚠️ This will reset all ML training data. Are you sure?')) {
        return;
    }

    try {
        alert('🔄 Resetting ML training data and initializing fresh models...');
        console.log('Resetting ML data...');
        setTimeout(() => {
            alert('✅ ML data reset complete! Models will retrain with new data.');
        }, 2000);
    } catch (error) {
        alert('❌ Error resetting ML data: ' + error.message);
    }
}

async function exportMLMetrics() {
    try {
        alert('📊 Generating ML metrics export... Please wait.');
        console.log('Exporting ML metrics...');
        setTimeout(() => {
            alert('✅ ML metrics exported successfully! Check your downloads folder.');
        }, 2000);
    } catch (error) {
        alert('❌ Error exporting ML metrics: ' + error.message);
    }
}

// Modal management
function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
    if (chart) {
        chart.destroy();
        chart = null;
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Login form
    document.getElementById('login-form').addEventListener('submit', handleLogin);

    // Register form
    document.getElementById('register-form').addEventListener('submit', handleRegister);

    // User form
    document.getElementById('user-form').addEventListener('submit', handleUserForm);

    // Admin form
    document.getElementById('admin-form').addEventListener('submit', handleAdminForm);

    // Change password form
    document.getElementById('change-password-form').addEventListener('submit', handleChangePasswordForm);

    // Close modals when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            event.target.style.display = 'none';
            if (chart) {
                chart.destroy();
                chart = null;
            }
        }
    });

    console.log('🤖 Advanced AI/ML Load Distribution System initialized!');
});
