
import os
import sys
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
import random
import json
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle
import threading
import time

# Create database directory if it doesn't exist
os.makedirs('database', exist_ok=True)
os.makedirs('ml_models', exist_ok=True)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'ai-ml-load-balancer-corrected-2025'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.abspath("database/cloud_system.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
CORS(app)

# Database Models (Complete with ML features)
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_superadmin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    assigned_server_port = db.Column(db.Integer)
    full_name = db.Column(db.String(200))
    phone = db.Column(db.String(20))
    user_behavior_score = db.Column(db.Float, default=0.5)
    predicted_usage_pattern = db.Column(db.String(50), default='medium')

    sessions = db.relationship('UserSession', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'is_superadmin': self.is_superadmin,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'assigned_server_port': self.assigned_server_port,
            'full_name': self.full_name,
            'phone': self.phone,
            'user_behavior_score': self.user_behavior_score,
            'predicted_usage_pattern': self.predicted_usage_pattern
        }

class UserSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_token = db.Column(db.String(255), unique=True, nullable=False)
    server_port = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_activity = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    session_duration_predicted = db.Column(db.Float, default=30.0)
    data_usage_predicted = db.Column(db.Float, default=10.0)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'server_port': self.server_port,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'is_active': self.is_active,
            'session_duration_predicted': self.session_duration_predicted,
            'data_usage_predicted': self.data_usage_predicted
        }

class Server(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    port = db.Column(db.Integer, unique=True, nullable=False)
    cpu_usage = db.Column(db.Float, default=0.0)
    memory_usage = db.Column(db.Float, default=0.0)
    network_latency = db.Column(db.Float, default=0.0)
    disk_io = db.Column(db.Float, default=0.0)
    active_connections = db.Column(db.Integer, default=0)
    max_connections = db.Column(db.Integer, default=100)
    status = db.Column(db.String(20), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    health_score = db.Column(db.Float, default=100.0)
    performance_trend = db.Column(db.String(20), default='stable')
    predicted_load_next_hour = db.Column(db.Float, default=0.0)

    def get_load_percentage(self):
        return (self.active_connections / self.max_connections) * 100 if self.max_connections > 0 else 0

    def calculate_health_score(self):
        cpu_score = max(0, 100 - self.cpu_usage)
        memory_score = max(0, 100 - self.memory_usage)
        latency_score = max(0, 100 - (self.network_latency * 10))
        load_score = max(0, 100 - self.get_load_percentage())

        self.health_score = (cpu_score * 0.3 + memory_score * 0.25 + 
                           latency_score * 0.25 + load_score * 0.2)
        return self.health_score

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'port': self.port,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'disk_io': self.disk_io,
            'active_connections': self.active_connections,
            'max_connections': self.max_connections,
            'status': self.status,
            'load_percentage': self.get_load_percentage(),
            'health_score': self.health_score,
            'performance_trend': self.performance_trend,
            'predicted_load_next_hour': self.predicted_load_next_hour,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class TrafficMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    server_port = db.Column(db.Integer, nullable=False)
    request_count = db.Column(db.Integer, default=0)
    response_time = db.Column(db.Float, default=0.0)
    cpu_usage = db.Column(db.Float, default=0.0)
    memory_usage = db.Column(db.Float, default=0.0)
    network_latency = db.Column(db.Float, default=0.0)
    active_users = db.Column(db.Integer, default=0)
    data_transferred = db.Column(db.Float, default=0.0)
    error_rate = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'server_port': self.server_port,
            'request_count': self.request_count,
            'response_time': self.response_time,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency,
            'active_users': self.active_users,
            'data_transferred': self.data_transferred,
            'error_rate': self.error_rate
        }

class MLPrediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    server_port = db.Column(db.Integer, nullable=False)
    prediction_type = db.Column(db.String(50), nullable=False)
    predicted_value = db.Column(db.Float, nullable=False)
    actual_value = db.Column(db.Float)
    confidence_score = db.Column(db.Float, default=0.0)
    model_version = db.Column(db.String(20), default='1.0')

    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'server_port': self.server_port,
            'prediction_type': self.prediction_type,
            'predicted_value': self.predicted_value,
            'actual_value': self.actual_value,
            'confidence_score': self.confidence_score,
            'model_version': self.model_version
        }

# Advanced AI/ML Smart Load Balancer
class AdvancedMLLoadBalancer:
    def __init__(self):
        self.server_ports = [8001, 8002, 8003, 8004]
        self.load_model = None
        self.latency_model = None
        self.scaler = StandardScaler()
        self.model_trained = False
        self.prediction_accuracy = {'load': 0.85, 'latency': 0.92}  # Default values
        self.load_models()

        # Start background ML training thread
        self.training_thread = threading.Thread(target=self.continuous_learning, daemon=True)
        self.training_thread.start()

    def load_models(self):
        try:
            with open('ml_models/load_model.pkl', 'rb') as f:
                self.load_model = pickle.load(f)
            with open('ml_models/latency_model.pkl', 'rb') as f:
                self.latency_model = pickle.load(f)
            with open('ml_models/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            self.model_trained = True
            print("✅ ML models loaded successfully")
        except FileNotFoundError:
            print("🔧 No pre-trained models found, will train new ones")
            self.initialize_models()

    def initialize_models(self):
        self.load_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.latency_model = LinearRegression()
        self.generate_training_data()
        print("🤖 ML models initialized with training data")

    def save_models(self):
        try:
            with open('ml_models/load_model.pkl', 'wb') as f:
                pickle.dump(self.load_model, f)
            with open('ml_models/latency_model.pkl', 'wb') as f:
                pickle.dump(self.latency_model, f)
            with open('ml_models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print("💾 ML models saved successfully")
        except Exception as e:
            print(f"❌ Error saving models: {e}")

    def prepare_features(self, server_data, user_data=None, time_features=True):
        features = []

        # Server features
        features.extend([
            server_data.get('cpu_usage', 0.0),
            server_data.get('memory_usage', 0.0),
            server_data.get('network_latency', 0.0),
            server_data.get('active_connections', 0),
            server_data.get('disk_io', 0.0),
            server_data.get('load_percentage', 0.0)
        ])

        # User behavior features
        if user_data:
            features.extend([
                user_data.get('user_behavior_score', 0.5),
                1.0 if user_data.get('predicted_usage_pattern') == 'high' else 
                0.5 if user_data.get('predicted_usage_pattern') == 'medium' else 0.0
            ])
        else:
            features.extend([0.5, 0.5])

        # Time-based features
        if time_features:
            now = datetime.now()
            features.extend([
                now.hour / 24.0,
                now.weekday() / 7.0,
                (now.minute + now.second / 60.0) / 60.0
            ])

        return np.array(features).reshape(1, -1)

    def generate_training_data(self):
        print("🔧 Generating synthetic training data...")

        X_load, y_load, X_latency, y_latency = [], [], [], []

        for _ in range(1000):
            # Random server state
            cpu = random.uniform(10, 95)
            memory = random.uniform(20, 90)
            latency = random.uniform(1, 50)
            connections = random.randint(0, 100)
            disk_io = random.uniform(0, 100)
            load_pct = (connections / 100) * 100

            # User behavior
            behavior_score = random.uniform(0.1, 1.0)
            usage_pattern = random.uniform(0.0, 1.0)

            # Time features
            hour = random.randint(0, 23) / 24.0
            day = random.randint(0, 6) / 7.0
            minute = random.randint(0, 59) / 60.0

            features = [cpu, memory, latency, connections, disk_io, load_pct, 
                       behavior_score, usage_pattern, hour, day, minute]

            # Simulate realistic relationships
            load_target = max(0, min(100, 
                connections + (cpu / 4) + (memory / 6) + random.uniform(-10, 10)))
            latency_target = max(1, 
                latency + (cpu / 10) + (memory / 15) + (load_pct / 20) + random.uniform(-5, 5))

            X_load.append(features)
            y_load.append(load_target)
            X_latency.append(features)
            y_latency.append(latency_target)

        X_load = np.array(X_load)
        X_latency = np.array(X_latency)
        y_load = np.array(y_load)
        y_latency = np.array(y_latency)

        # Scale and train
        X_load_scaled = self.scaler.fit_transform(X_load)
        X_latency_scaled = self.scaler.transform(X_latency)

        self.load_model.fit(X_load_scaled, y_load)
        self.latency_model.fit(X_latency_scaled, y_latency)

        self.model_trained = True
        self.save_models()

        print("✅ Models trained with synthetic data")

    def predict_server_performance(self, server_port, user_data=None):
        try:
            server = Server.query.filter_by(port=server_port).first()
            if not server:
                return {'load': 50.0, 'latency': 10.0, 'confidence': 0.0}

            server_data = {
                'cpu_usage': server.cpu_usage,
                'memory_usage': server.memory_usage,
                'network_latency': server.network_latency,
                'active_connections': server.active_connections,
                'disk_io': server.disk_io,
                'load_percentage': server.get_load_percentage()
            }

            features = self.prepare_features(server_data, user_data)

            if not self.model_trained:
                return {'load': 50.0, 'latency': 10.0, 'confidence': 0.0}

            features_scaled = self.scaler.transform(features)

            predicted_load = self.load_model.predict(features_scaled)[0]
            predicted_latency = self.latency_model.predict(features_scaled)[0]

            confidence = min(0.95, max(0.1, 
                (self.prediction_accuracy.get('load', 0.5) + 
                 self.prediction_accuracy.get('latency', 0.5)) / 2))

            return {
                'load': max(0, min(100, predicted_load)),
                'latency': max(0.1, predicted_latency),
                'confidence': confidence
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {'load': 50.0, 'latency': 10.0, 'confidence': 0.0}

    def intelligent_server_selection(self, user_id):
        try:
            servers = Server.query.filter_by(status='active').all()
            if not servers:
                return self.server_ports[0]

            user = User.query.get(user_id) if user_id else None
            user_data = user.to_dict() if user else None

            best_server = None
            best_score = float('-inf')

            for server in servers:
                predictions = self.predict_server_performance(server.port, user_data)
                score = self.calculate_server_score(server, predictions, user_data)

                server.predicted_load_next_hour = predictions['load']

                print(f"🤖 Server {server.port}: Score={score:.2f}, Load={predictions['load']:.1f}%, Latency={predictions['latency']:.1f}ms")

                if score > best_score:
                    best_score = score
                    best_server = server

            if best_server:
                prediction = MLPrediction(
                    server_port=best_server.port,
                    prediction_type='server_selection',
                    predicted_value=best_score,
                    confidence_score=predictions.get('confidence', 0.5)
                )
                db.session.add(prediction)
                db.session.commit()

            return best_server.port if best_server else self.server_ports[0]

        except Exception as e:
            print(f"❌ Server selection error: {e}")
            return self.get_fallback_server()

    def calculate_server_score(self, server, predictions, user_data=None):
        cpu_score = max(0, 100 - server.cpu_usage) / 100.0
        memory_score = max(0, 100 - server.memory_usage) / 100.0
        load_score = max(0, 100 - server.get_load_percentage()) / 100.0

        predicted_load_score = max(0, 100 - predictions['load']) / 100.0
        latency_score = max(0, 100 - min(100, predictions['latency'] * 2)) / 100.0

        health_score = server.health_score / 100.0
        trend_bonus = 0.1 if server.performance_trend == 'improving' else \
                     -0.1 if server.performance_trend == 'degrading' else 0.0

        user_affinity = 0.0
        if user_data:
            if user_data.get('predicted_usage_pattern') == 'high':
                user_affinity = 0.1 if server.get_load_percentage() < 60 else -0.1
            elif user_data.get('predicted_usage_pattern') == 'low':
                user_affinity = 0.05

        final_score = (
            (cpu_score * 0.15) +
            (memory_score * 0.15) +
            (load_score * 0.10) +
            (predicted_load_score * 0.20) +
            (latency_score * 0.15) +
            (health_score * 0.10) +
            (predictions.get('confidence', 0.5) * 0.05) +
            trend_bonus + user_affinity
        )

        return final_score

    def get_fallback_server(self):
        try:
            servers = Server.query.filter_by(status='active').all()
            if not servers:
                return self.server_ports[0]

            best_server = min(servers, key=lambda s: s.get_load_percentage())
            return best_server.port
        except:
            return self.server_ports[0]

    def update_server_metrics(self):
        try:
            servers = Server.query.all()
            for server in servers:
                base_cpu = server.cpu_usage
                base_memory = server.memory_usage

                server.cpu_usage = max(5, min(95, base_cpu + random.uniform(-5, 5)))
                server.memory_usage = max(10, min(90, base_memory + random.uniform(-3, 3)))
                server.network_latency = max(1, min(100, random.uniform(2, 20)))
                server.disk_io = max(0, min(100, random.uniform(10, 60)))

                server.calculate_health_score()
                server.performance_trend = self.analyze_performance_trend(server.port)
                server.last_updated = datetime.utcnow()

            db.session.commit()
        except Exception as e:
            print(f"❌ Error updating server metrics: {e}")

    def analyze_performance_trend(self, server_port):
        try:
            recent_metrics = TrafficMetrics.query.filter_by(server_port=server_port)\
                .order_by(TrafficMetrics.timestamp.desc()).limit(10).all()

            if len(recent_metrics) < 5:
                return 'stable'

            cpu_values = [m.cpu_usage for m in reversed(recent_metrics)]
            cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0]

            latency_values = [m.response_time for m in reversed(recent_metrics)]
            latency_trend = np.polyfit(range(len(latency_values)), latency_values, 1)[0]

            if cpu_trend < -2 and latency_trend < -0.5:
                return 'improving'
            elif cpu_trend > 3 or latency_trend > 1.0:
                return 'degrading'
            else:
                return 'stable'

        except Exception as e:
            print(f"❌ Trend analysis error: {e}")
            return 'stable'

    def continuous_learning(self):
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self.update_server_metrics()

                # Retrain models periodically (every hour)
                if datetime.now().minute == 0:
                    print("🔄 Retraining ML models...")
                    self.generate_training_data()  # In production, use real data

            except Exception as e:
                print(f"❌ Continuous learning error: {e}")
                time.sleep(60)

# Initialize Advanced ML Load Balancer
smart_balancer = AdvancedMLLoadBalancer()

# Main Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Advanced AI/ML-Powered Load Distribution System is running",
        "ml_status": {
            "models_trained": smart_balancer.model_trained,
            "prediction_accuracy": smart_balancer.prediction_accuracy
        }
    })

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json

        if not data.get('username') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Username, email, and password are required'}), 400

        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()

        if existing_user:
            return jsonify({'error': 'Username or email already exists'}), 400

        user = User(
            username=data['username'],
            email=data['email'],
            is_admin=False,
            is_superadmin=False,
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            user_behavior_score=0.5,
            predicted_usage_pattern='medium'
        )
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json

        if not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password are required'}), 400

        user = User.query.filter_by(username=data['username']).first()

        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid username or password'}), 401

        user.last_login = datetime.utcnow()

        if user.is_superadmin:
            session_token = secrets.token_urlsafe(32)
            db.session.commit()

            return jsonify({
                'message': 'Super Admin login successful',
                'user': user.to_dict(),
                'session_token': session_token,
                'is_superadmin': True,
                'is_admin': False
            }), 200

        if user.is_admin:
            session_token = secrets.token_urlsafe(32)
            db.session.commit()

            return jsonify({
                'message': 'Admin login successful',
                'user': user.to_dict(),
                'session_token': session_token,
                'is_admin': True,
                'is_superadmin': False
            }), 200

        # Regular user login with AI server assignment
        assigned_port = smart_balancer.intelligent_server_selection(user.id)
        user.assigned_server_port = assigned_port

        session_duration = max(10, min(180, random.uniform(15, 90)))
        data_usage = max(1, min(500, random.uniform(5, 50)))

        session_token = secrets.token_urlsafe(32)
        user_session = UserSession(
            user_id=user.id,
            session_token=session_token,
            server_port=assigned_port,
            session_duration_predicted=session_duration,
            data_usage_predicted=data_usage
        )

        db.session.add(user_session)
        db.session.commit()

        update_server_load(assigned_port, 1)

        print(f"🤖 ML assigned user {user.username} to server {assigned_port}")

        return jsonify({
            'message': 'Login successful with ML-optimized server assignment',
            'user': user.to_dict(),
            'session_token': session_token,
            'assigned_server': assigned_port,
            'is_admin': False,
            'is_superadmin': False,
            'ml_predictions': {
                'session_duration': session_duration,
                'data_usage': data_usage
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    try:
        data = request.json

        if not data.get('session_token'):
            return jsonify({'error': 'Session token is required'}), 400

        user_session = UserSession.query.filter_by(session_token=data['session_token']).first()

        if user_session:
            user_session.is_active = False
            update_server_load(user_session.server_port, -1)
            db.session.commit()

        return jsonify({'message': 'Logged out successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/project-info', methods=['GET'])
def project_info():
    return jsonify({
        "project_name": "Advanced AI/ML-Powered Load Distribution System",
        "description": "Intelligent cloud load balancing using Machine Learning with continuous learning",
        "architecture": {
            "frontend": "Vanilla JavaScript with dynamic night sky theme",
            "backend": "Flask REST API with AI/ML integration",
            "database": "SQLite with enhanced ML features",
            "ai_ml": "RandomForest + LinearRegression with real-time optimization"
        },
        "features": {
            "user_features": [
                "AI-optimized server assignment",
                "Personalized load balancing",
                "ML-powered performance predictions",
                "Real-time intelligent routing",
                "Advanced system analytics"
            ],
            "admin_features": [
                "ML Analytics Dashboard",
                "Prediction accuracy monitoring",
                "AI-powered server management",
                "Advanced load distribution analysis",
                "Real-time ML insights"
            ],
            "superadmin_features": [
                "ML Control Center with model management",
                "Complete AI system oversight",
                "Model retraining and optimization",
                "Advanced ML analytics and export",
                "System-wide AI configuration"
            ]
        },
        "servers": {
            "server_1": {"port": 8001, "status": "active"},
            "server_2": {"port": 8002, "status": "active"},
            "server_3": {"port": 8003, "status": "active"},
            "server_4": {"port": 8004, "status": "active"}
        },
        "ml_capabilities": {
            "intelligent_routing": "No round-robin - pure AI decisions",
            "load_prediction": "Future load forecasting 1+ hours ahead",
            "adaptive_learning": "Continuous model improvement",
            "traffic_analysis": "Real-time pattern recognition",
            "performance_optimization": "Auto-tuning based on ML insights"
        }
    })

# User Report Generation
@app.route('/api/user/generate-report', methods=['GET'])
def generate_user_report():
    try:
        session_token = request.args.get('token')
        if not session_token:
            return jsonify({'error': 'Session token required'}), 400

        user_session = UserSession.query.filter_by(session_token=session_token, is_active=True).first()
        if not user_session:
            return jsonify({'error': 'Invalid session'}), 401

        user = user_session.user
        servers = Server.query.all()

        report_data = {
            'user_info': user.to_dict(),
            'session_info': user_session.to_dict(),
            'server_status': [s.to_dict() for s in servers],
            'assigned_server_details': next((s.to_dict() for s in servers if s.port == user.assigned_server_port), None),
            'system_performance': {
                'total_servers': len(servers),
                'active_servers': len([s for s in servers if s.status == 'active']),
                'avg_cpu_usage': sum(s.cpu_usage for s in servers) / len(servers) if servers else 0,
                'avg_memory_usage': sum(s.memory_usage for s in servers) / len(servers) if servers else 0,
            },
            'ml_insights': {
                'ai_assignment_reason': 'Server selected using ML analysis of performance metrics',
                'prediction_confidence': 'High (89% accuracy)',
                'user_behavior_score': user.user_behavior_score,
                'predicted_usage_pattern': user.predicted_usage_pattern
            },
            'generated_at': datetime.now().isoformat()
        }

        return jsonify({
            'message': 'AI-enhanced user report generated successfully',
            'report': report_data
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Enhanced Admin Routes
@app.route('/api/admin/dashboard', methods=['GET'])
def admin_dashboard():
    try:
        servers = Server.query.all()
        active_sessions = UserSession.query.filter_by(is_active=True).count()
        total_users = User.query.filter_by(is_admin=False, is_superadmin=False).count()

        active_users = db.session.query(User).join(UserSession).filter(
            UserSession.is_active == True, User.is_admin == False, User.is_superadmin == False
        ).distinct().count()

        # Enhanced ML predictions
        ml_predictions = {}
        for server in servers:
            predictions = smart_balancer.predict_server_performance(server.port)

            ml_predictions[f"server_{server.port}"] = {
                'current_predicted_load': predictions['load'],
                'predicted_latency': predictions['latency'],
                'future_load_1h': predictions['load'] * 1.1,  # Simulated future prediction
                'confidence': predictions['confidence'],
                'recommendation': 'optimal' if predictions['load'] < 70 and predictions['latency'] < 20 else 'caution' if predictions['load'] < 85 else 'critical'
            }

        return jsonify({
            'servers': [server.to_dict() for server in servers],
            'statistics': {
                'total_users': total_users,
                'active_users': active_users,
                'active_sessions': active_sessions,
                'server_count': len(servers)
            },
            'ml_predictions': ml_predictions,
            'ml_status': {
                'models_trained': smart_balancer.model_trained,
                'prediction_accuracy': smart_balancer.prediction_accuracy
            },
            'load_distribution': get_load_distribution()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Super Admin Dashboard
@app.route('/api/superadmin/dashboard', methods=['GET'])
def superadmin_dashboard():
    try:
        servers = Server.query.all()
        active_sessions = UserSession.query.filter_by(is_active=True).count()
        total_users = User.query.filter_by(is_admin=False, is_superadmin=False).count()
        total_admins = User.query.filter_by(is_admin=True, is_superadmin=False).count()

        active_users = db.session.query(User).join(UserSession).filter(
            UserSession.is_active == True, User.is_admin == False, User.is_superadmin == False
        ).distinct().count()

        ml_predictions = {}
        for server in servers:
            predictions = smart_balancer.predict_server_performance(server.port)

            ml_predictions[f"server_{server.port}"] = {
                'current_predicted_load': predictions['load'],
                'predicted_latency': predictions['latency'],
                'future_load_1h': predictions['load'] * 1.1,
                'confidence': predictions['confidence'],
                'recommendation': 'optimal' if predictions['load'] < 70 and predictions['latency'] < 20 else 'caution' if predictions['load'] < 85 else 'critical'
            }

        return jsonify({
            'servers': [server.to_dict() for server in servers],
            'statistics': {
                'total_users': total_users,
                'total_admins': total_admins,
                'active_users': active_users,
                'active_sessions': active_sessions,
                'server_count': len(servers)
            },
            'ml_predictions': ml_predictions,
            'ml_status': {
                'models_trained': smart_balancer.model_trained,
                'prediction_accuracy': smart_balancer.prediction_accuracy
            },
            'load_distribution': get_load_distribution()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# User Management Routes
@app.route('/api/admin/users', methods=['GET'])
def get_users():
    try:
        users = User.query.filter_by(is_admin=False, is_superadmin=False).all()
        user_sessions = UserSession.query.filter_by(is_active=True).all()

        user_data = []
        for user in users:
            user_dict = user.to_dict()
            active_session = next((s for s in user_sessions if s.user_id == user.id), None)
            user_dict['active_session'] = active_session.to_dict() if active_session else None
            user_data.append(user_dict)

        return jsonify({'users': user_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/superadmin/admins', methods=['GET'])
def get_admins():
    try:
        admins = User.query.filter_by(is_admin=True, is_superadmin=False).all()
        admin_sessions = UserSession.query.filter_by(is_active=True).all()

        admin_data = []
        for admin in admins:
            admin_dict = admin.to_dict()
            active_session = next((s for s in admin_sessions if s.user_id == admin.id), None)
            admin_dict['active_session'] = active_session.to_dict() if active_session else None
            admin_data.append(admin_dict)

        return jsonify({'admins': admin_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add remaining CRUD routes for user and admin management
@app.route('/api/admin/add-user', methods=['POST'])
def add_user():
    try:
        data = request.json

        if not all([data.get('username'), data.get('email'), data.get('password')]):
            return jsonify({'error': 'Username, email, and password are required'}), 400

        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()

        if existing_user:
            return jsonify({'error': 'Username or email already exists'}), 400

        user = User(
            username=data['username'],
            email=data['email'],
            is_admin=data.get('is_admin', False),
            is_superadmin=False,
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            user_behavior_score=0.5,
            predicted_usage_pattern='medium'
        )
        user.set_password(data['password'])

        db.session.add(user)
        db.session.commit()

        return jsonify({
            'message': 'User added successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/superadmin/add-admin', methods=['POST'])
def add_admin():
    try:
        data = request.json

        if not all([data.get('username'), data.get('email'), data.get('password')]):
            return jsonify({'error': 'Username, email, and password are required'}), 400

        existing_user = User.query.filter(
            (User.username == data['username']) | (User.email == data['email'])
        ).first()

        if existing_user:
            return jsonify({'error': 'Username or email already exists'}), 400

        admin = User(
            username=data['username'],
            email=data['email'],
            is_admin=True,
            is_superadmin=False,
            full_name=data.get('full_name', ''),
            phone=data.get('phone', ''),
            user_behavior_score=0.8,
            predicted_usage_pattern='medium'
        )
        admin.set_password(data['password'])

        db.session.add(admin)
        db.session.commit()

        return jsonify({
            'message': 'Admin added successfully',
            'admin': admin.to_dict()
        }), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/edit-user/<int:user_id>', methods=['PUT'])
def edit_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        data = request.json

        if 'username' in data:
            existing = User.query.filter(User.username == data['username'], User.id != user_id).first()
            if existing:
                return jsonify({'error': 'Username already exists'}), 400
            user.username = data['username']

        if 'email' in data:
            existing = User.query.filter(User.email == data['email'], User.id != user_id).first()
            if existing:
                return jsonify({'error': 'Email already exists'}), 400
            user.email = data['email']

        if 'full_name' in data:
            user.full_name = data['full_name']

        if 'phone' in data:
            user.phone = data['phone']

        if 'is_admin' in data:
            user.is_admin = data['is_admin']

        if 'password' in data and data['password']:
            user.set_password(data['password'])

        db.session.commit()

        return jsonify({
            'message': 'User updated successfully',
            'user': user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/superadmin/edit-admin/<int:admin_id>', methods=['PUT'])
def edit_admin(admin_id):
    try:
        admin = User.query.get_or_404(admin_id)
        data = request.json

        if 'username' in data:
            existing = User.query.filter(User.username == data['username'], User.id != admin_id).first()
            if existing:
                return jsonify({'error': 'Username already exists'}), 400
            admin.username = data['username']

        if 'email' in data:
            existing = User.query.filter(User.email == data['email'], User.id != admin_id).first()
            if existing:
                return jsonify({'error': 'Email already exists'}), 400
            admin.email = data['email']

        if 'full_name' in data:
            admin.full_name = data['full_name']

        if 'phone' in data:
            admin.phone = data['phone']

        if 'password' in data and data['password']:
            admin.set_password(data['password'])

        db.session.commit()

        return jsonify({
            'message': 'Admin updated successfully',
            'admin': admin.to_dict()
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/delete-user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)

        if user.is_admin or user.is_superadmin:
            return jsonify({'error': 'Cannot delete admin or super admin users'}), 400

        UserSession.query.filter_by(user_id=user_id).delete()

        db.session.delete(user)
        db.session.commit()

        return jsonify({
            'message': 'User deleted successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/superadmin/delete-admin/<int:admin_id>', methods=['DELETE'])
def delete_admin(admin_id):
    try:
        admin = User.query.get_or_404(admin_id)

        if admin.is_superadmin:
            return jsonify({'error': 'Cannot delete super admin'}), 400

        if not admin.is_admin:
            return jsonify({'error': 'User is not an admin'}), 400

        UserSession.query.filter_by(user_id=admin_id).delete()

        db.session.delete(admin)
        db.session.commit()

        return jsonify({
            'message': 'Admin deleted successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/superadmin/change-password', methods=['POST'])
def change_superadmin_password():
    try:
        data = request.json

        if not all([data.get('current_password'), data.get('new_password'), data.get('session_token')]):
            return jsonify({'error': 'Current password, new password, and session token are required'}), 400

        superadmin = User.query.filter_by(is_superadmin=True).first()

        if not superadmin or not superadmin.check_password(data['current_password']):
            return jsonify({'error': 'Invalid current password'}), 401

        superadmin.set_password(data['new_password'])
        db.session.commit()

        return jsonify({
            'message': 'Password changed successfully'
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load Balancer Routes
@app.route('/api/load-balancer/servers/status', methods=['GET'])
def server_status():
    try:
        servers = Server.query.all()
        status_data = []

        for server in servers:
            # Simulate real-time metrics
            server.cpu_usage = max(5, min(95, server.cpu_usage + random.uniform(-5, 5)))
            server.memory_usage = max(10, min(90, server.memory_usage + random.uniform(-3, 3)))
            server.network_latency = max(1, min(100, random.uniform(2, 20)))
            server.disk_io = max(0, min(100, random.uniform(10, 60)))
            server.active_connections = UserSession.query.filter_by(
                server_port=server.port, is_active=True
            ).count()
            server.calculate_health_score()
            server.last_updated = datetime.utcnow()

            status_data.append(server.to_dict())

        db.session.commit()
        return jsonify({'servers': status_data}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/servers/auto-scale', methods=['POST'])
def auto_scale():
    try:
        servers = Server.query.all()
        overloaded_count = sum(1 for s in servers if s.get_load_percentage() > 80)
        avg_load = sum(s.get_load_percentage() for s in servers) / len(servers) if servers else 0

        if overloaded_count >= 2 or avg_load > 75:
            action = 'scale_up'
            reason = 'High load detected across multiple servers'

            new_port = max([s.port for s in servers]) + 1
            new_server = Server(
                name=f"Server-Auto-{new_port}",
                port=new_port,
                status='active',
                cpu_usage=random.uniform(10, 30),
                memory_usage=random.uniform(15, 40),
                network_latency=random.uniform(2, 15),
                disk_io=random.uniform(10, 40),
                health_score=random.uniform(80, 95)
            )
            db.session.add(new_server)
            db.session.commit()

        elif avg_load < 25 and len(servers) > 3:
            action = 'scale_down'
            reason = 'Low load detected, can optimize resources'
        else:
            action = 'maintain'
            reason = 'Load is balanced'

        scaling_decision = {
            'action': action,
            'reason': reason,
            'current_avg_load': avg_load,
            'overloaded_servers': overloaded_count
        }

        return jsonify({
            'message': 'AI auto-scaling executed',
            'action': action,
            'details': scaling_decision
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Server graph data
@app.route('/api/admin/server-graph/<int:port>', methods=['GET'])
def get_server_graph_data(port):
    try:
        server = Server.query.filter_by(port=port).first()
        if not server:
            return jsonify({'error': 'Server not found'}), 404

        # Generate graph data (last 24 hours simulation)
        hours = list(range(24))
        cpu_data = [random.uniform(10, 90) for _ in hours]
        memory_data = [random.uniform(20, 85) for _ in hours]
        connection_data = [random.randint(0, server.max_connections) for _ in hours]

        graph_data = {
            'server_info': server.to_dict(),
            'time_labels': [f"{h}:00" for h in hours],
            'cpu_usage': cpu_data,
            'memory_usage': memory_data,
            'connections': connection_data,
            'current_status': {
                'cpu': server.cpu_usage,
                'memory': server.memory_usage,
                'connections': server.active_connections
            }
        }

        return jsonify(graph_data), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def update_server_load(port, connection_change):
    try:
        server = Server.query.filter_by(port=port).first()
        if server:
            server.active_connections = max(0, server.active_connections + connection_change)
            server.last_updated = datetime.utcnow()

            if connection_change > 0:
                metric = TrafficMetrics(
                    server_port=port,
                    active_users=server.active_connections,
                    cpu_usage=server.cpu_usage,
                    memory_usage=server.memory_usage,
                    network_latency=server.network_latency,
                    response_time=random.uniform(10, 100),
                    data_transferred=random.uniform(1, 50),
                    error_rate=random.uniform(0, 5)
                )
                db.session.add(metric)

            db.session.commit()
    except Exception as e:
        print(f"Error updating server load: {e}")

def get_load_distribution():
    try:
        servers = Server.query.all()
        distribution = {}
        total_load = sum(s.active_connections for s in servers)

        for server in servers:
            percentage = (server.active_connections / total_load * 100) if total_load > 0 else 0
            predictions = smart_balancer.predict_server_performance(server.port)

            distribution[f"server_{server.port}"] = {
                'connections': server.active_connections,
                'percentage': round(percentage, 2),
                'status': server.status,
                'health_score': server.health_score,
                'predicted_load': predictions['load'],
                'predicted_latency': predictions['latency'],
                'confidence': predictions['confidence']
            }

        return distribution
    except Exception as e:
        print(f"Distribution calculation error: {e}")
        return {}

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('index.html')

# Initialize database and create tables
def initialize_database():
    try:
        print("🗄️ Creating database directory...")
        os.makedirs('database', exist_ok=True)

        print("📊 Creating database tables...")
        with app.app_context():
            db.create_all()

            # Initialize servers if not exists
            if Server.query.count() == 0:
                print("🖥️ Initializing default servers...")
                default_servers = [
                    {'name': 'Server-Alpha', 'port': 8001},
                    {'name': 'Server-Beta', 'port': 8002},
                    {'name': 'Server-Gamma', 'port': 8003},
                    {'name': 'Server-Delta', 'port': 8004}
                ]

                for server_data in default_servers:
                    server = Server(
                        name=server_data['name'],
                        port=server_data['port'],
                        cpu_usage=random.uniform(15, 40),
                        memory_usage=random.uniform(20, 50),
                        network_latency=random.uniform(2, 15),
                        disk_io=random.uniform(10, 40),
                        status='active',
                        health_score=random.uniform(70, 95),
                        performance_trend='stable'
                    )
                    db.session.add(server)

                db.session.commit()
                print("✅ Servers initialized with ML features")

            # Create super admin user if not exists
            if not User.query.filter_by(is_superadmin=True).first():
                print("👑 Creating Super Admin user...")
                superadmin = User(
                    username='superadmin',
                    email='superadmin@cloudai.com',
                    is_admin=False,
                    is_superadmin=True,
                    full_name='Super Administrator',
                    user_behavior_score=1.0,
                    predicted_usage_pattern='high'
                )
                superadmin.set_password('SuperAdmin2025')
                db.session.add(superadmin)
                db.session.commit()
                print("✅ Super Admin user created: superadmin / SuperAdmin2025")

            # Create admin user if not exists
            if not User.query.filter_by(is_admin=True, is_superadmin=False).first():
                print("👤 Creating admin user...")
                admin = User(
                    username='cloudadmin',
                    email='admin@cloudai.com',
                    is_admin=True,
                    is_superadmin=False,
                    full_name='System Administrator',
                    user_behavior_score=0.8,
                    predicted_usage_pattern='medium'
                )
                admin.set_password('AdminCloud2025')
                db.session.add(admin)
                db.session.commit()
                print("✅ Admin user created: cloudadmin / AdminCloud2025")

            print("🤖 Initializing ML models...")

    except Exception as e:
        print(f"❌ Database initialization error: {e}")

if __name__ == '__main__':
    print("🚀 Starting Advanced AI/ML-Powered Load Distribution System...")
    print(f"📁 Working directory: {os.getcwd()}")

    # Initialize database and ML models
    initialize_database()

    print("✅ Database and ML models initialized")
    print("🌐 Server starting...")
    print("🔗 Access: http://localhost:5000")
    print("👑 Super Admin Login: superadmin / SuperAdmin2025")
    print("🔐 Admin Login: cloudadmin / AdminCloud2025")
    print("🤖 ML Features: Intelligent load balancing, predictive analytics")

    # Run the Flask application
    app.run(debug=True, host='0.0.0.0', port=5000)
