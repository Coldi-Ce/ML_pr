from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from datetime import datetime, timedelta
import json
import numpy as np
import yt_dlp
import time
import uuid
from categories import CATEGORIES

# ----------------- Configuration -----------------
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['AVATAR_FOLDER'] = 'static/avatars'
app.config['FACE_FOLDER'] = 'static/faces'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.permanent_session_lifetime = timedelta(days=7)

db = SQLAlchemy(app)
video_categories = db.Table(
    'video_categories',
    db.Column('video_id', db.Integer, db.ForeignKey('video.id'), primary_key=True),
    db.Column('category_id', db.Integer, db.ForeignKey('category.id'), primary_key=True)
)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    avatar = db.Column(db.String, nullable=True, default='default.png')
    face_image = db.Column(db.String(120), nullable=True)
    name = db.Column(db.String(100), nullable=True)
    videos = db.relationship('Video', backref='user', lazy=True)

    def check_password(self, pwd):
        return self.password == pwd

class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)
    title = db.Column(db.String(255), nullable=False)
    comment = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    categories = db.relationship(
        'Category', secondary=video_categories, lazy='subquery',
        backref=db.backref('videos', lazy=True)
    )

# ----------------- Helpers -----------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def compute_embedding(path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = models.resnet18(pretrained=True).to(device)
    feature_extractor = nn.Sequential(*list(base_model.children())[:-1]).to(device)
    feature_extractor.eval()
    tfms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(path).convert('RGB')
    inp = tfms(img).unsqueeze(0).to(device)
    with torch.no_grad(): feat = feature_extractor(inp)
    emb = feat.squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)

def seed_categories():
    if Category.query.count() == 0:
        for name in CATEGORIES:
            db.session.add(Category(name=name))
        db.session.commit()


@app.route('/')
def index():
    videos = Video.query.order_by(Video.timestamp.desc()).all()
    return render_template('index.html', videos=videos)


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method=='POST':
        name = request.form['name']; email = request.form['email']; password = request.form['password']
        if User.query.filter_by(email=email).first(): flash('Пользователь существует'); return redirect(url_for('register'))
        user = User(email=email,password=password,name=name)
        db.session.add(user); db.session.commit()
        # update label_map
        path = os.path.join('model','label_map.json'); lm={}
        if os.path.exists(path):
            with open(path,'r',encoding='utf-8') as f: lm=json.load(f)
        lm[str(user.id)] = name
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with open(path,'w',encoding='utf-8') as f: json.dump(lm,f,ensure_ascii=False,indent=4)
        flash('Регистрация успешна'); return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        fi = request.files.get('face_file')
        if fi and allowed_file(fi.filename):
            temp = os.path.join(app.config['FACE_FOLDER'], f"temp_{uuid.uuid4().hex}.jpg")
            fi.save(temp)
            emb = compute_embedding(temp)
            best,score=None,-1
            for u in User.query.filter(User.face_image!=None):
                p=os.path.join(app.config['FACE_FOLDER'],u.face_image)
                if os.path.exists(p): s=np.dot(emb,compute_embedding(p))
                if s>score: best,score=u,s
            os.remove(temp)
            if best and score>0.84:
                session['user_id']=best.id
                flash(f'Здравствуйте, {best.name}!')
                return redirect(url_for('profile'))
            flash('Лицо не распознано.'); return redirect(url_for('login'))
        # password login
        email=request.form.get('email'); pwd=request.form.get('password')
        if email and pwd:
            u=User.query.filter_by(email=email).first()
            if u and u.check_password(pwd): session['user_id']=u.id; flash('Успешный вход'); return redirect(url_for('profile'))
        flash('Неверные данные'); return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/profile', methods=['GET','POST'])
def profile():
    if 'user_id' not in session: return redirect(url_for('login'))
    seed_categories()
    user = db.session.get(User, session['user_id'])
    categories = Category.query.all()
    if request.method=='POST':
        video_url = request.form.get('video_url')
        comment = request.form.get('comment')
        if video_url:
            cats = request.form.getlist('categories')
            with yt_dlp.YoutubeDL({'quiet':True}) as ydl:
                info = ydl.extract_info(video_url,download=False)
            v=Video(url=video_url,title=info.get('title','Untitled'),comment=comment,user=user)
            for cid in cats:
                c=Category.query.get(int(cid));
                if c: v.categories.append(c)
            db.session.add(v); db.session.commit(); return redirect(url_for('profile'))

        av=request.files.get('avatar')
        if av and av.filename:
            ext = av.filename.rsplit('.',1)[1].lower()
            filename = f"avatar_{user.id}_{int(time.time())}.{ext}"
            path = os.path.join(app.config['AVATAR_FOLDER'], filename)
            os.makedirs(os.path.dirname(path),exist_ok=True); av.save(path); user.avatar=filename; db.session.commit(); return redirect(url_for('profile'))

        fi=request.files.get('face_file')
        if fi and allowed_file(fi.filename):
            ext = fi.filename.rsplit('.',1)[1].lower()
            filename = f"face_{user.id}_{int(time.time())}.{ext}"
            path = os.path.join(app.config['FACE_FOLDER'], filename)
            os.makedirs(os.path.dirname(path),exist_ok=True); fi.save(path); user.face_image=filename; db.session.commit(); return redirect(url_for('profile'))
    videos = Video.query.filter_by(user_id=user.id).order_by(Video.timestamp.desc()).all()
    return render_template('profile.html',user=user,categories=categories,videos=videos)

from urllib.parse import urlparse, parse_qs

from sqlalchemy import func

@app.route('/view_video/<int:video_id>')
def view_video(video_id):
    v = Video.query.get_or_404(video_id)

    parsed = urlparse(v.url)
    vid = None
    if parsed.hostname in ('youtu.be',):
        vid = parsed.path[1:]
    elif 'youtube' in parsed.hostname:
        qs = parse_qs(parsed.query)
        vid = qs.get('v', [None])[0]
    embed_url = f'https://www.youtube.com/embed/{vid or v.url}'
    cat_ids = [c.id for c in v.categories]
    if cat_ids:
        related = (
            Video.query
            .join(video_categories)
            .filter(video_categories.c.category_id.in_(cat_ids))
            .filter(Video.id != v.id)
            .order_by(Video.timestamp.desc())
            .distinct()
            .limit(10)
            .all()
        )
    else:
        related = []

    return render_template(
        'video_view.html',
        video=v,
        embed_url=embed_url,
        related_videos=related
    )



@app.route('/download_video/<int:video_id>')
def download_video(video_id):
    v=Video.query.get_or_404(video_id)
    with yt_dlp.YoutubeDL({'quiet':True,'format':'best'}) as ydl:
        info=ydl.extract_info(v.url,download=False)
        stream=info.get('url')
    return redirect(stream)

@app.route('/logout')
def logout():
    session.pop('user_id',None); flash('Вы вышли'); return redirect(url_for('index'))

if __name__=='__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
