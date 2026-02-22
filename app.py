import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import spacy
import re
import language_tool_python
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sqlalchemy.orm import Session
import requests 
import smtplib 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables from .env file
load_dotenv()

# --- AUTH & DB IMPORTS ---
from auth import UserDB, hash_password, verify_password, Base, engine, get_db
from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# ==========================================
# 1. DATABASE MODELS (Strict Sync with MySQL)
# ==========================================
class CandidateDB(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    upload_date = Column(DateTime, default=datetime.utcnow)
    resume_link = Column(Text)
    
    # Candidate Identity
    first_name = Column(String(100))
    last_name = Column(String(100))
    email = Column(String(150))
    
    # Deep Analysis Data (Keeping all verified columns)
    strengths = Column(Text)
    weaknesses = Column(Text)
    risk_factor = Column(Text)
    reward_factor = Column(Text)
    overall_fit = Column(Integer)
    justification = Column(Text)
    
    # AI Metrics (Matches your MySQL result grid)
    tone_label = Column(String(50))
    tone_score = Column(Float)
    soft_skills = Column(Text)
    job_stability_score = Column(Float)
    grammar_mistakes = Column(Integer)
    
    # Metadata & Tracking
    personal_details = Column(Text)
    behavior_prediction = Column(String(100))
    entities = Column(Text)
    vector_size = Column(Integer)
    nlp_timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="Pending")
    role = Column(String(150))

# --- NEW MODEL: JOB LISTINGS (Inserted here) ---
class JobListingDB(Base):
    __tablename__ = "job_listings"

    id = Column(Integer, primary_key=True, index=True)
    role_name = Column(String(150), unique=True, nullable=False)
    jd_link = Column(Text, nullable=False)
    keywords = Column(Text)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# ==========================================
# 2. HYBRID EMAIL/WEBHOOK CONFIGURATION
# ==========================================
USE_N8N_FOR_EMAIL = False  # Set to True for n8n Webhook, False for Direct Gmail

N8N_URL = os.getenv("N8N_URL", "http://localhost:5678/webhook-test/shortlist-email-trigger")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("EMAIL_USER", "pandugadharmateja05@gmail.com")
SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD", "jjir ayjt hycd eacj") 

def trigger_email_notification(email, name, status):
    """Triggers either the n8n webhook or Direct Gmail SMTP with status context."""
    if USE_N8N_FOR_EMAIL:
        try:
            # Added "status" to payload so n8n Switch node can route
            requests.post(N8N_URL, json={"email": email, "first_name": name, "status": status})
            print(f"n8n Webhook triggered for {email} with status {status}")
        except Exception as e:
            print(f"n8n trigger failed: {e}")
    else:
        try:
            msg = MIMEMultipart()
            msg['From'] = SENDER_EMAIL
            msg['To'] = email
            
            # Logic branch for Subject and Body based on status
            if status == "Shortlisted":
                msg['Subject'] = "Congratulations! You've been Shortlisted"
                body = f"Hi {name},\n\nCongratulations! We have reviewed your application for the Full Stack AI Developer role and would like to move forward with your candidacy.\n\nYou have been SHORTLISTED for the next round of interviews. Our team will contact you shortly with the next steps.\n\nBest regards,\nThe Recruitment Team"
            else:
                msg['Subject'] = "Update regarding your application"
                body = f"Hi {name},\n\nThank you for your interest in the Full Stack AI Developer role. After careful consideration, we have decided not to move forward with your application at this time.\n\nWe appreciate the time you took to apply and wish you the best in your search.\n\nBest regards,\nThe Recruitment Team"

            msg.attach(MIMEText(body, 'plain'))
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"Direct Gmail sent to {email} (Status: {status})")
        except Exception as e:
            print(f"SMTP failed: {e}")

# ==========================================
# 3. APP & AI SETUP
# ==========================================
app = FastAPI(title="AI Recruitment Insight Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer("all-MiniLM-L6-v2")
tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

try:
    tool = language_tool_python.LanguageTool('en-US')
except:
    tool = None

SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving","adaptability", "time management", "critical thinking","collaboration", "creativity", "decision making"]
TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative","analytical thinker", "adaptable", "detail-oriented", "proactive"]

class ResumePayload(BaseModel):
    candidate_id: str
    resume_text: str

# ==========================================
# 4. ENDPOINTS
# ==========================================

@app.post("/register")
def register(user: dict, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.email == user['email']).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = UserDB(full_name=user.get('full_name'), email=user['email'], 
                      password=hash_password(user['password']), role=user.get('role', 'applicant'))
    db.add(new_user); db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.email == data['email']).first()
    if not user or not verify_password(data['password'], user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"id": user.id, "name": user.full_name, "role": user.role, "email": user.email}

@app.get("/candidates/")
def get_candidates(db: Session = Depends(get_db)):
    return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

@app.patch("/candidates/{c_id}/status")
def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
    candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    new_status = status_update.get('status', 'Pending')
    candidate.status = new_status
    db.commit()

    # TRIGGER NOTIFICATION FOR BOTH SHORTLISTED AND REJECTED
    if new_status in ["Shortlisted", "Rejected"]:
        trigger_email_notification(candidate.email, candidate.first_name, new_status)

    return {"message": f"Updated to {candidate.status}"}

@app.get("/my-application/{email}")
def get_my_application_status(email: str, db: Session = Depends(get_db)):
    application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
    if not application:
        raise HTTPException(status_code=404, detail="No application found")
    return {"first_name": application.first_name, "status": application.status, 
            "date": application.upload_date, "fit_score": application.overall_fit,"role": application.role}

@app.post("/process")
def process_resume(data: ResumePayload):
    text = data.resume_text[:2000]
    tone_result = tone_classifier(text, TONE_LABELS)
    found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
    company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
    stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
    grammar_errors = 0
    if tool:
        grammar_matches = tool.check(text)
        grammar_errors = len(grammar_matches)
    doc = nlp(text)
    personal_details = {"names": [], "organizations": [], "locations": []}
    for ent in doc.ents:
        if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
        elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
        elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
    behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
    sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

    return {
        "candidate_id": data.candidate_id,
        "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
        "soft_skills": ", ".join(found_soft_skills),
        "job_stability": stability,
        "grammar": {"error_count": grammar_errors},
        "personal_details": personal_details,
        "behavior_profile": {"type": behavior_result["labels"][0]},
        "sentiment": sentiment,
        "processed_at": datetime.utcnow().isoformat()
    }

# --- NEW ENDPOINTS: JOBS (Inserted here) ---
@app.post("/jobs")
def add_job(job_data: dict, db: Session = Depends(get_db)):
    """Recruiter posts a new job listing to the database."""
    if db.query(JobListingDB).filter(JobListingDB.role_name == job_data['role_name']).first():
        raise HTTPException(status_code=400, detail="Role already exists")
    
    new_job = JobListingDB(
        role_name=job_data['role_name'],
        jd_link=job_data['jd_link'],
        keywords=job_data['keywords']
    )
    db.add(new_job)
    db.commit()
    return {"message": "Job posted successfully"}

@app.get("/jobs")
def get_jobs(db: Session = Depends(get_db)):
    """Fetches all active job listings for the Applicant Portal."""
    return db.query(JobListingDB).filter(JobListingDB.is_active == 1).all()



# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session
# import requests 
# import smtplib 
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Strict Sync with MySQL)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis Data (Keeping all verified columns)
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics (Matches your MySQL result grid)
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")
#     role = Column(String(150))

# # --- NEW MODEL: JOB LISTINGS (Inserted here) ---
# class JobListingDB(Base):
#     __tablename__ = "job_listings"

#     id = Column(Integer, primary_key=True, index=True)
#     role_name = Column(String(150), unique=True, nullable=False)
#     jd_link = Column(Text, nullable=False)
#     keywords = Column(Text)
#     is_active = Column(Integer, default=1)
#     created_at = Column(DateTime, default=datetime.utcnow)

# # Create tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. HYBRID EMAIL/WEBHOOK CONFIGURATION
# # ==========================================
# USE_N8N_FOR_EMAIL = False  # Set to True for n8n Webhook, False for Direct Gmail

# N8N_URL = "http://localhost:5678/webhook-test/shortlist-email-trigger"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SENDER_EMAIL = "pandugadharmateja05@gmail.com"
# SENDER_PASSWORD = "jjir ayjt hycd eacj" 

# def trigger_email_notification(email, name, status):
#     """Triggers either the n8n webhook or Direct Gmail SMTP with status context."""
#     if USE_N8N_FOR_EMAIL:
#         try:
#             # Added "status" to payload so n8n Switch node can route
#             requests.post(N8N_URL, json={"email": email, "first_name": name, "status": status})
#             print(f"n8n Webhook triggered for {email} with status {status}")
#         except Exception as e:
#             print(f"n8n trigger failed: {e}")
#     else:
#         try:
#             msg = MIMEMultipart()
#             msg['From'] = SENDER_EMAIL
#             msg['To'] = email
            
#             # Logic branch for Subject and Body based on status
#             if status == "Shortlisted":
#                 msg['Subject'] = "Congratulations! You've been Shortlisted"
#                 body = f"Hi {name},\n\nCongratulations! We have reviewed your application for the Full Stack AI Developer role and would like to move forward with your candidacy.\n\nYou have been SHORTLISTED for the next round of interviews. Our team will contact you shortly with the next steps.\n\nBest regards,\nThe Recruitment Team"
#             else:
#                 msg['Subject'] = "Update regarding your application"
#                 body = f"Hi {name},\n\nThank you for your interest in the Full Stack AI Developer role. After careful consideration, we have decided not to move forward with your application at this time.\n\nWe appreciate the time you took to apply and wish you the best in your search.\n\nBest regards,\nThe Recruitment Team"

#             msg.attach(MIMEText(body, 'plain'))
#             server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#             server.quit()
#             print(f"Direct Gmail sent to {email} (Status: {status})")
#         except Exception as e:
#             print(f"SMTP failed: {e}")

# # ==========================================
# # 3. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving","adaptability", "time management", "critical thinking","collaboration", "creativity", "decision making"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative","analytical thinker", "adaptable", "detail-oriented", "proactive"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
#     new_user = UserDB(full_name=user.get('full_name'), email=user['email'], 
#                       password=hash_password(user['password']), role=user.get('role', 'applicant'))
#     db.add(new_user); db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return {"id": user.id, "name": user.full_name, "role": user.role, "email": user.email}

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     new_status = status_update.get('status', 'Pending')
#     candidate.status = new_status
#     db.commit()

#     # TRIGGER NOTIFICATION FOR BOTH SHORTLISTED AND REJECTED
#     if new_status in ["Shortlisted", "Rejected"]:
#         trigger_email_notification(candidate.email, candidate.first_name, new_status)

#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found")
#     return {"first_name": application.first_name, "status": application.status, 
#             "date": application.upload_date, "fit_score": application.overall_fit,"role": application.role}

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": ", ".join(found_soft_skills),
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }

# # --- NEW ENDPOINTS: JOBS (Inserted here) ---
# @app.post("/jobs")
# def add_job(job_data: dict, db: Session = Depends(get_db)):
#     """Recruiter posts a new job listing to the database."""
#     if db.query(JobListingDB).filter(JobListingDB.role_name == job_data['role_name']).first():
#         raise HTTPException(status_code=400, detail="Role already exists")
    
#     new_job = JobListingDB(
#         role_name=job_data['role_name'],
#         jd_link=job_data['jd_link'],
#         keywords=job_data['keywords']
#     )
#     db.add(new_job)
#     db.commit()
#     return {"message": "Job posted successfully"}

# @app.get("/jobs")
# def get_jobs(db: Session = Depends(get_db)):
#     """Fetches all active job listings for the Applicant Portal."""
#     return db.query(JobListingDB).filter(JobListingDB.is_active == 1).all()





# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session
# import requests 
# import smtplib 
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Strict Sync with MySQL)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis Data (Keeping all verified columns)
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics (Matches your MySQL result grid)
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")

# # Create tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. HYBRID EMAIL/WEBHOOK CONFIGURATION
# # ==========================================
# USE_N8N_FOR_EMAIL = False  # Set to True for n8n Webhook, False for Direct Gmail

# N8N_URL = "http://localhost:5678/webhook-test/shortlist-email-trigger"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SENDER_EMAIL = "pandugadharmateja05@gmail.com"
# SENDER_PASSWORD = "jjir ayjt hycd eacj" 

# def trigger_email_notification(email, name, status):
#     """Triggers either the n8n webhook or Direct Gmail SMTP with status context."""
#     if USE_N8N_FOR_EMAIL:
#         try:
#             # Added "status" to payload so n8n Switch node can route
#             requests.post(N8N_URL, json={"email": email, "first_name": name, "status": status})
#             print(f"n8n Webhook triggered for {email} with status {status}")
#         except Exception as e:
#             print(f"n8n trigger failed: {e}")
#     else:
#         try:
#             msg = MIMEMultipart()
#             msg['From'] = SENDER_EMAIL
#             msg['To'] = email
            
#             # Logic branch for Subject and Body based on status
#             if status == "Shortlisted":
#                 msg['Subject'] = "Congratulations! You've been Shortlisted"
#                 body = f"Hi {name},\n\nCongratulations! We have reviewed your application for the Full Stack AI Developer role and would like to move forward with your candidacy.\n\nYou have been SHORTLISTED for the next round of interviews. Our team will contact you shortly with the next steps.\n\nBest regards,\nThe Recruitment Team"
#             else:
#                 msg['Subject'] = "Update regarding your application"
#                 body = f"Hi {name},\n\nThank you for your interest in the Full Stack AI Developer role. After careful consideration, we have decided not to move forward with your application at this time.\n\nWe appreciate the time you took to apply and wish you the best in your search.\n\nBest regards,\nThe Recruitment Team"

#             msg.attach(MIMEText(body, 'plain'))
#             server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#             server.quit()
#             print(f"Direct Gmail sent to {email} (Status: {status})")
#         except Exception as e:
#             print(f"SMTP failed: {e}")

# # ==========================================
# # 3. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving","adaptability", "time management", "critical thinking","collaboration", "creativity", "decision making"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative","analytical thinker", "adaptable", "detail-oriented", "proactive"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
#     new_user = UserDB(full_name=user.get('full_name'), email=user['email'], 
#                       password=hash_password(user['password']), role=user.get('role', 'applicant'))
#     db.add(new_user); db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return {"id": user.id, "name": user.full_name, "role": user.role, "email": user.email}

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     new_status = status_update.get('status', 'Pending')
#     candidate.status = new_status
#     db.commit()

#     # TRIGGER NOTIFICATION FOR BOTH SHORTLISTED AND REJECTED
#     if new_status in ["Shortlisted", "Rejected"]:
#         trigger_email_notification(candidate.email, candidate.first_name, new_status)

#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found")
#     return {"first_name": application.first_name, "status": application.status, 
#             "date": application.upload_date, "fit_score": application.overall_fit}

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": ", ".join(found_soft_skills),
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }





















# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session
# import requests 
# import smtplib 
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Strict Sync with MySQL)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis Data (Keeping all verified columns)
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics (Matches your MySQL result grid)
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")

# # Create tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. HYBRID EMAIL/WEBHOOK CONFIGURATION
# # ==========================================
# USE_N8N_FOR_EMAIL = False  # Set to True for n8n Webhook, False for Direct Gmail

# N8N_URL = "http://localhost:5678/webhook-test/shortlist-email-trigger"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SENDER_EMAIL = "pandugadharmateja05@gmail.com"
# SENDER_PASSWORD = "jjir ayjt hycd eacj" 

# def trigger_email_notification(email, name):
#     """Triggers either the n8n webhook or Direct Gmail SMTP."""
#     if USE_N8N_FOR_EMAIL:
#         try:
#             # THIS IS YOUR WEBHOOK LOGIC
#             requests.post(N8N_URL, json={"email": email, "first_name": name})
#             print(f"n8n Webhook triggered for {email}")
#         except Exception as e:
#             print(f"n8n trigger failed: {e}")
#     else:
#         try:
#             # THIS IS YOUR DIRECT GMAIL LOGIC
#             msg = MIMEMultipart()
#             msg['From'] = SENDER_EMAIL
#             msg['To'] = email
#             msg['Subject'] = "Congratulations! You've been Shortlisted"
#             body = f"Hi {name},\n\nCongratulations! We have reviewed your application for the Full Stack AI Developer role and would like to move forward with your candidacy.\n\nYou have been SHORTLISTED for the next round of interviews. Our team will contact you shortly with the next steps.\n\nBest regards,\nThe Recruitment Team"
#             msg.attach(MIMEText(body, 'plain'))
#             server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#             server.quit()
#             print(f"Direct Gmail sent to {email}")
#         except Exception as e:
#             print(f"SMTP failed: {e}")

# # ==========================================
# # 3. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving","adaptability", "time management", "critical thinking","collaboration", "creativity", "decision making"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative","analytical thinker", "adaptable", "detail-oriented", "proactive"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
#     new_user = UserDB(full_name=user.get('full_name'), email=user['email'], 
#                       password=hash_password(user['password']), role=user.get('role', 'applicant'))
#     db.add(new_user); db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return {"id": user.id, "name": user.full_name, "role": user.role, "email": user.email}

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     new_status = status_update.get('status', 'Pending')
#     candidate.status = new_status
#     db.commit()

#     # TRIGGER EMAIL/WEBHOOK IF SHORTLISTED
#     if new_status == "Shortlisted":
#         trigger_email_notification(candidate.email, candidate.first_name)

#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found")
#     return {"first_name": application.first_name, "status": application.status, 
#             "date": application.upload_date, "fit_score": application.overall_fit}

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": ", ".join(found_soft_skills),
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session
# import requests 
# import smtplib 
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Full Column Set)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis Data (Keeping everything except the non-existent 'skills')
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics (Matches your MySQL grid)
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")

# # Create tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. HYBRID EMAIL CONFIGURATION
# # ==========================================
# USE_N8N_FOR_EMAIL = True  # Toggle: True for n8n, False for Python SMTP  

# N8N_URL = "http://localhost:5678/webhook-test/shortlist-email-trigger"

# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SENDER_EMAIL = "pandugadharmateja05@gmail.com"
# SENDER_PASSWORD = "jjir ayjt hycd eacj" 

# def trigger_email_notification(email, name):
#     if USE_N8N_FOR_EMAIL:
#         try:
#             requests.post(N8N_URL, json={"email": email, "first_name": name})
#         except Exception as e:
#             print(f"n8n failed: {e}")
#     else:
#         try:
#             msg = MIMEMultipart()
#             msg['From'] = SENDER_EMAIL
#             msg['To'] = email
#             msg['Subject'] = "Congratulations! You've been Shortlisted"
#             body = f"Hi {name},\n\nCongratulations! We have reviewed your application for the Full Stack AI Developer role and would like to move forward with your candidacy.\n\nYou have been SHORTLISTED for the next round of interviews. Our team will contact you shortly with the next steps.\n\nBest regards,\nThe Recruitment Team"
#             msg.attach(MIMEText(body, 'plain'))
#             server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#             server.quit()
#         except Exception as e:
#             print(f"SMTP failed: {e}")

# # ==========================================
# # 3. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
#     new_user = UserDB(full_name=user.get('full_name'), email=user['email'], 
#                       password=hash_password(user['password']), role=user.get('role', 'applicant'))
#     db.add(new_user); db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return {"id": user.id, "name": user.full_name, "role": user.role, "email": user.email}

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     new_status = status_update.get('status', 'Pending')
#     candidate.status = new_status
#     db.commit()
#     if new_status == "Shortlisted":
#         trigger_email_notification(candidate.email, candidate.first_name)
#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found")
#     return {"first_name": application.first_name, "status": application.status, 
#             "date": application.upload_date, "fit_score": application.overall_fit}

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": ", ".join(found_soft_skills),
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session
# import requests # Necessary for n8n
# import smtplib # Necessary for direct email
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Strict Sync with Existing MySQL)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Existing Analysis Data ONLY
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     overall_fit = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")

# # Create ALL tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. HYBRID EMAIL CONFIGURATION
# # ==========================================
# USE_N8N_FOR_EMAIL = False  # Toggle: True for n8n, False for Python SMTP

# # n8n Webhook (Replace with your Production URL)
# N8N_URL = "http://localhost:5678/webhook-test/shortlist-email-trigger"

# # Direct SMTP Config (Replace with your details)
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# SENDER_EMAIL = "pandugadharmateja05@gmail.com"
# SENDER_PASSWORD = "jjir ayjt hycd eacj" 

# def trigger_email_notification(email, name):
#     """Handles the hybrid logic for sending emails."""
#     if USE_N8N_FOR_EMAIL:
#         try:
#             requests.post(N8N_URL, json={"email": email, "first_name": name})
#             print(f"n8n Webhook triggered for {email}")
#         except Exception as e:
#             print(f"n8n failed: {e}")
#     else:
#         try:
#             msg = MIMEMultipart()
#             msg['From'] = SENDER_EMAIL
#             msg['To'] = email
#             msg['Subject'] = "Congratulations! You've been Shortlisted"
#             body = f"""
#         Hi {name},

#             Congratulations! We have reviewed your application for the Full Stack AI Developer role 
#             and would like to move forward with your candidacy.

#             You have been SHORTLISTED for the next round of interviews. Our team will contact 
#              you shortly with the next steps.

#         Best regards,
#         The Recruitment Team
#         """
#             msg.attach(MIMEText(body, 'plain'))
            
#             server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#             server.quit()
#             print(f"Python SMTP email sent to {email}")
#         except Exception as e:
#             print(f"Python SMTP failed: {e}")

# # ==========================================
# # 3. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load AI Models
# print("Loading AI Models... Please wait.")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# # AI Logic Constants
# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. AUTHENTICATION ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     new_user = UserDB(
#         full_name=user.get('full_name'),
#         email=user['email'],
#         password=hash_password(user['password']),
#         role=user.get('role', 'applicant')
#     )
#     db.add(new_user)
#     db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     return {
#         "id": user.id,
#         "name": user.full_name,
#         "role": user.role,
#         "email": user.email
#     }

# # ==========================================
# # 5. RECRUITMENT & APPLICANT ENDPOINTS
# # ==========================================

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
    
#     new_status = status_update.get('status', 'Pending')
#     candidate.status = new_status
#     db.commit()

#     # TRIGGER NOTIFICATION IF SHORTLISTED
#     if new_status == "Shortlisted":
#         trigger_email_notification(candidate.email, candidate.first_name)

#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found for this email")
    
#     return {
#         "first_name": application.first_name,
#         "status": application.status,
#         "date": application.upload_date,
#         "fit_score": application.overall_fit
#     }

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session

# # --- AUTH & DB IMPORTS ---
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Strict Sync with Existing MySQL)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Existing Analysis Data ONLY
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     overall_fit = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending")

# # Create ALL tables
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load AI Models
# print("Loading AI Models... Please wait.")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# # AI Logic Constants
# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 3. AUTHENTICATION ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     new_user = UserDB(
#         full_name=user.get('full_name'),
#         email=user['email'],
#         password=hash_password(user['password']),
#         role=user.get('role', 'applicant')
#     )
#     db.add(new_user)
#     db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     return {
#         "id": user.id,
#         "name": user.full_name,
#         "role": user.role,
#         "email": user.email
#     }

# # ==========================================
# # 4. RECRUITMENT & APPLICANT ENDPOINTS
# # ==========================================

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     """Fetches candidates using only the columns that exist in DB."""
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     """Allows Recruiter to update status (Shortlisted/Rejected)."""
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     candidate.status = status_update.get('status', 'Pending')
#     db.commit()
#     return {"message": f"Updated to {candidate.status}"}

# @app.get("/my-application/{email}")
# def get_my_application_status(email: str, db: Session = Depends(get_db)):
#     """Allows an applicant to see their current progress safely."""
#     application = db.query(CandidateDB).filter(CandidateDB.email == email).order_by(CandidateDB.upload_date.desc()).first()
#     if not application:
#         raise HTTPException(status_code=404, detail="No application found for this email")
    
#     return {
#         "first_name": application.first_name,
#         "status": application.status,
#         "date": application.upload_date,
#         "fit_score": application.overall_fit
#     }

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     """Core AI processing logic."""
#     text = data.resume_text[:2000]
#     tone_result = tone_classifier(text, TONE_LABELS)
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }

# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline
# from sqlalchemy.orm import Session

# # --- AUTH & DB IMPORTS ---
# # This links your main app to the security logic in auth.py
# from auth import UserDB, hash_password, verify_password, Base, engine, get_db
# from sqlalchemy import Column, Integer, String, Float, Text, DateTime

# # ==========================================
# # 1. DATABASE MODELS (Exact Column Mapping)
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis Data
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Metadata & Tracking
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending") # Added for tracking in dashboards

# # Create ALL tables (Candidates and Users)
# Base.metadata.create_all(bind=engine)

# # ==========================================
# # 2. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load AI Models
# print("Loading AI Models... Please wait.")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# # AI Logic Constants
# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 3. AUTHENTICATION ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     if db.query(UserDB).filter(UserDB.email == user['email']).first():
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     new_user = UserDB(
#         full_name=user.get('full_name'),
#         email=user['email'],
#         password=hash_password(user['password']),
#         role=user.get('role', 'applicant')
#     )
#     db.add(new_user)
#     db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     return {
#         "id": user.id,
#         "name": user.full_name,
#         "role": user.role,
#         "email": user.email
#     }

# # ==========================================
# # 4. RECRUITMENT ENDPOINTS
# # ==========================================

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     """Fetches ALL columns defined in CandidateDB for the React table."""
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.patch("/candidates/{c_id}/status")
# def update_status(c_id: int, status_update: dict, db: Session = Depends(get_db)):
#     """Allows Recruiter to update status (Shortlisted/Rejected)."""
#     candidate = db.query(CandidateDB).filter(CandidateDB.id == c_id).first()
#     if not candidate:
#         raise HTTPException(status_code=404, detail="Candidate not found")
#     candidate.status = status_update.get('status', 'Pending')
#     db.commit()
#     return {"message": f"Updated to {candidate.status}"}

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     """Core AI processing logic used by n8n."""
#     text = data.resume_text[:2000]

#     # 1. Tone Analysis
#     tone_result = tone_classifier(text, TONE_LABELS)
    
#     # 2. Soft Skills
#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]

#     # 3. Job Stability (Based on company mentions)
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "Low" if len(company_mentions) >= 4 else "Moderate" if len(company_mentions) >= 2 else "High"

#     # 4. Grammar
#     grammar_errors = 0
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)

#     # 5. Entity Extraction (Personal Details)
#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)

#     # 6. Behavior Profiling
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
    
#     # 7. Sentiment
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_result["labels"][0], "confidence": round(float(tone_result["scores"][0]), 3)},
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
##################################################
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# # --- DATABASE IMPORTS ---
# from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session

# # --- AUTH IMPORTS ---
# # Ensure you have created the auth.py file in the same folder
# from auth import UserDB, hash_password, verify_password, Base as AuthBase

# # ==========================================
# # 1. DATABASE CONFIGURATION
# # ==========================================
# DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/recruitment_ai"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # --- Candidate Data Mapping ---
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)
#     vector_size = Column(Integer)
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)
#     status = Column(String(50), default="Pending") # Added for tracking

# # Create ALL tables (Candidates and Users)
# Base.metadata.create_all(bind=engine)
# AuthBase.metadata.create_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # ==========================================
# # 2. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Models
# print("Loading AI Models... Please wait.")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# # Constants
# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 3. AUTHENTICATION ENDPOINTS
# # ==========================================

# @app.post("/register")
# def register(user: dict, db: Session = Depends(get_db)):
#     existing_user = db.query(UserDB).filter(UserDB.email == user['email']).first()
#     if existing_user:
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     new_user = UserDB(
#         full_name=user.get('full_name'),
#         email=user['email'],
#         password=hash_password(user['password']),
#         role=user['role']
#     )
#     db.add(new_user)
#     db.commit()
#     return {"message": "User created successfully"}

# @app.post("/login")
# def login(data: dict, db: Session = Depends(get_db)):
#     user = db.query(UserDB).filter(UserDB.email == data['email']).first()
#     if not user or not verify_password(data['password'], user.password):
#         raise HTTPException(status_code=401, detail="Invalid credentials")
    
#     return {
#         "id": user.id,
#         "name": user.full_name,
#         "role": user.role,
#         "email": user.email
#     }

# # ==========================================
# # 4. RECRUITMENT ENDPOINTS
# # ==========================================

# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]

#     # AI Processing
#     tone_result = tone_classifier(text, TONE_LABELS)
#     tone_label = tone_result["labels"][0]
#     tone_score = tone_result["scores"][0]

#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]

#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     stability = "High" if len(company_mentions) < 2 else "Moderate" if len(company_mentions) < 4 else "Low"

#     grammar_errors = 0
#     if tool:
#         matches = tool.check(text)
#         grammar_errors = len(matches)

#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)

#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
    
#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_label, "confidence": round(float(tone_score), 3)},
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {"error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_result["labels"][0]},
#         "processed_at": datetime.utcnow().isoformat()
#     }
#########################################################
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# # --- DATABASE IMPORTS ---
# from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session

# # ==========================================
# # 1. DATABASE CONFIGURATION
# # ==========================================
# # ⚠️ MAKE SURE THE DB NAME IS CORRECT ('recruitment_ai' or 'recruitment_db')
# DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/recruitment_ai"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # ==========================================
# # 2. THE CRITICAL PART: EXACT COLUMN MAPPING
# # ==========================================
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     # I have mapped these EXACTLY to the list you gave me
#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
    
#     # Candidate Identity
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Deep Analysis
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Scores
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
    
#     # Advanced Metadata
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
#     entities = Column(Text)         # Added this
#     vector_size = Column(Integer)   # Added this
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)

# # Create tables if they don't exist
# Base.metadata.create_all(bind=engine)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # ==========================================
# # 3. APP SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Models
# print("Loading AI Models...")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except:
#     tool = None

# # Constants
# SOFT_SKILLS = ["leadership", "communication", "teamwork", "problem solving", "adaptability"]
# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]
# BEHAVIOR_LABELS = ["team player", "independent worker", "leader", "innovative"]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 4. ENDPOINTS
# # ==========================================

# # --- Endpoint for React Dashboard ---
# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     # This fetches ALL columns defined in CandidateDB above
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()


# # --- Endpoint for n8n ---
# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]

#     # AI Processing Logic
#     tone_result = tone_classifier(text, TONE_LABELS)
#     tone_label = tone_result["labels"][0]
#     tone_score = tone_result["scores"][0]

#     found_soft_skills = [skill for skill in SOFT_SKILLS if skill.lower() in text.lower()]

#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     if len(company_mentions) >= 4: stability = "Low Stability"
#     elif len(company_mentions) >= 2: stability = "Moderate Stability"
#     else: stability = "High Stability"

#     grammar_errors = 0
#     grammar_quality = "Unknown"
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#         if grammar_errors > 20: grammar_quality = "Poor"
#         elif grammar_errors > 10: grammar_quality = "Moderate"
#         else: grammar_quality = "Good"

#     doc = nlp(text)
#     personal_details = {"names": [], "organizations": [], "locations": []}
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)

#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     behavior_label = behavior_result["labels"][0]
#     behavior_score = behavior_result["scores"][0]
    
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {"label": tone_label, "confidence": round(float(tone_score), 3)},
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {"quality": grammar_quality, "error_count": grammar_errors},
#         "personal_details": personal_details,
#         "behavior_profile": {"type": behavior_label, "confidence": round(float(behavior_score), 3)},
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
#fine bou no auth
###################################################################################
# from fastapi import FastAPI, Depends, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Optional
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# # --- DATABASE IMPORTS ---
# from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session

# # ==========================================
# # 1. DATABASE CONFIGURATION
# # ==========================================
# # ⚠️ UPDATE 'root:password' with your actual MySQL username and password
# DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/recruitment_ai"

# engine = create_engine(DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Define the MySQL Table Structure for Python
# class CandidateDB(Base):
#     __tablename__ = "candidates"

#     id = Column(Integer, primary_key=True, index=True)
#     upload_date = Column(DateTime, default=datetime.utcnow)
#     resume_link = Column(Text)
#     first_name = Column(String(100))
#     last_name = Column(String(100))
#     email = Column(String(150))
    
#     # Analysis Data
#     strengths = Column(Text)
#     weaknesses = Column(Text)
#     risk_factor = Column(Text)
#     reward_factor = Column(Text)
#     overall_fit = Column(Integer)
#     justification = Column(Text)
    
#     # AI Metrics
#     tone_label = Column(String(50))
#     tone_score = Column(Float)
#     soft_skills = Column(Text)
#     job_stability_score = Column(Float)
#     grammar_mistakes = Column(Integer)
#     personal_details = Column(Text)
#     behavior_prediction = Column(String(100))
    
#     # Meta
#     nlp_timestamp = Column(DateTime, default=datetime.utcnow)

# # Create tables if they don't exist
# Base.metadata.create_all(bind=engine)

# # Dependency to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # ==========================================
# # 2. APP & AI SETUP
# # ==========================================
# app = FastAPI(title="AI Recruitment Insight Engine")

# # CORS: Allow your React Frontend (Vite uses port 5173)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], # Allow all origins for dev; restrict in prod
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load Models (Your Original Setup)
# print("Loading AI Models... this may take a moment.")
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# try:
#     tool = language_tool_python.LanguageTool('en-US')
# except Exception as e:
#     print(f"Warning: LanguageTool failed to load (Java missing?). Grammar check will be limited. Error: {e}")
#     tool = None

# # Constants
# SOFT_SKILLS = [
#     "leadership", "communication", "teamwork", "problem solving",
#     "adaptability", "time management", "critical thinking",
#     "collaboration", "creativity", "decision making"
# ]

# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]

# BEHAVIOR_LABELS = [
#     "team player", "independent worker", "leader",
#     "innovative", "analytical thinker", "detail oriented"
# ]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# # ==========================================
# # 3. ENDPOINTS
# # ==========================================

# # --- NEW: Get List of Candidates for React Dashboard ---
# @app.get("/candidates/")
# def get_candidates(db: Session = Depends(get_db)):
#     # This queries the MySQL database and returns the list as JSON
#     return db.query(CandidateDB).order_by(CandidateDB.upload_date.desc()).all()


# # --- ORIGINAL: Process Resume Text (Used by n8n) ---
# @app.post("/process")
# def process_resume(data: ResumePayload):
#     text = data.resume_text[:2000]  # prevent transformer overflow

#     # 1️⃣ Tone Detection
#     tone_result = tone_classifier(text, TONE_LABELS)
#     tone_label = tone_result["labels"][0]
#     tone_score = tone_result["scores"][0]

#     # 2️⃣ Soft Skills Detection
#     found_soft_skills = [
#         skill for skill in SOFT_SKILLS
#         if skill.lower() in text.lower()
#     ]

#     # 3️⃣ Job Stability Detection
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     if len(company_mentions) >= 4:
#         stability = "Low Stability (Frequent Switching)"
#     elif len(company_mentions) >= 2:
#         stability = "Moderate Stability"
#     else:
#         stability = "High Stability"

#     # 4️⃣ Grammar Check
#     grammar_errors = 0
#     grammar_quality = "Unknown"
    
#     if tool:
#         grammar_matches = tool.check(text)
#         grammar_errors = len(grammar_matches)
#         if grammar_errors > 20: grammar_quality = "Poor"
#         elif grammar_errors > 10: grammar_quality = "Moderate"
#         else: grammar_quality = "Good"

#     # 5️⃣ Personal Details Extraction
#     doc = nlp(text)
#     personal_details = {
#         "names": [], "organizations": [], "locations": [], "dates": []
#     }
#     for ent in doc.ents:
#         if ent.label_ == "PERSON": personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG": personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE": personal_details["locations"].append(ent.text)
#         elif ent.label_ == "DATE": personal_details["dates"].append(ent.text)

#     # 6️⃣ Behavior Profiling
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     behavior_label = behavior_result["labels"][0]
#     behavior_score = behavior_result["scores"][0]

#     # 7️⃣ Sentiment
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {
#             "label": tone_label,
#             "confidence": round(float(tone_score), 3)
#         },
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {
#             "quality": grammar_quality,
#             "error_count": grammar_errors
#         },
#         "personal_details": personal_details,
#         "behavior_profile": {
#             "type": behavior_label,
#             "confidence": round(float(behavior_score), 3)
#         },
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
#########################################################################################
# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime
# import spacy
# import re
# import language_tool_python
# from sentence_transformers import SentenceTransformer, util
# from transformers import pipeline

# app = FastAPI(title="AI Recruitment Insight Engine")

# # Load lightweight models
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# tool = language_tool_python.LanguageTool('en-US')

# # Soft skill database
# SOFT_SKILLS = [
#     "leadership", "communication", "teamwork", "problem solving",
#     "adaptability", "time management", "critical thinking",
#     "collaboration", "creativity", "decision making"
# ]

# TONE_LABELS = ["professional", "confident", "formal", "casual", "neutral"]

# BEHAVIOR_LABELS = [
#     "team player", "independent worker", "leader",
#     "innovative", "analytical thinker", "detail oriented"
# ]

# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str


# @app.post("/process")
# def process_resume(data: ResumePayload):

#     text = data.resume_text[:2000]  # prevent transformer overflow

#     # =========================
#     # 1️⃣ Tone Detection
#     # =========================
#     tone_result = tone_classifier(text, TONE_LABELS)
#     tone_label = tone_result["labels"][0]
#     tone_score = tone_result["scores"][0]

#     # =========================
#     # 2️⃣ Soft Skills Detection
#     # =========================
#     found_soft_skills = [
#         skill for skill in SOFT_SKILLS
#         if skill.lower() in text.lower()
#     ]

#     # =========================
#     # 3️⃣ Job Stability Detection
#     # =========================
#     company_mentions = re.findall(r"(Pvt|Ltd|Inc|LLC|Company)", text)
#     years = re.findall(r"(20\d{2})", text)

#     if len(company_mentions) >= 4:
#         stability = "Low Stability (Frequent Switching)"
#     elif len(company_mentions) == 2 or len(company_mentions) == 3:
#         stability = "Moderate Stability"
#     else:
#         stability = "High Stability"

#     # =========================
#     # 4️⃣ Grammar Check
#     # =========================
#     grammar_matches = tool.check(text)
#     grammar_errors = len(grammar_matches)

#     if grammar_errors > 20:
#         grammar_quality = "Poor"
#     elif grammar_errors > 10:
#         grammar_quality = "Moderate"
#     else:
#         grammar_quality = "Good"

#     # =========================
#     # 5️⃣ Personal Details Extraction
#     # =========================
#     doc = nlp(text)
#     personal_details = {
#         "names": [],
#         "organizations": [],
#         "locations": [],
#         "dates": []
#     }

#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             personal_details["names"].append(ent.text)
#         elif ent.label_ == "ORG":
#             personal_details["organizations"].append(ent.text)
#         elif ent.label_ == "GPE":
#             personal_details["locations"].append(ent.text)
#         elif ent.label_ == "DATE":
#             personal_details["dates"].append(ent.text)

#     # =========================
#     # 6️⃣ Behavior Profiling
#     # =========================
#     behavior_result = tone_classifier(text, BEHAVIOR_LABELS)
#     behavior_label = behavior_result["labels"][0]
#     behavior_score = behavior_result["scores"][0]

#     # =========================
#     # 7️⃣ Sentiment (Backup Tone)
#     # =========================
#     sentiment = sentiment_model(text, truncation=True, max_length=512)[0]

#     return {
#         "candidate_id": data.candidate_id,
#         "tone": {
#             "label": tone_label,
#             "confidence": round(float(tone_score), 3)
#         },
#         "soft_skills": found_soft_skills,
#         "job_stability": stability,
#         "grammar": {
#             "quality": grammar_quality,
#             "error_count": grammar_errors
#         },
#         "personal_details": personal_details,
#         "behavior_profile": {
#             "type": behavior_label,
#             "confidence": round(float(behavior_score), 3)
#         },
#         "sentiment": sentiment,
#         "processed_at": datetime.utcnow().isoformat()
#     }
#/////////////////////////////////////////////////////////////////////////////////////////
# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime
# import spacy
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline

# app = FastAPI()

# # Load models
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("all-MiniLM-L6-v2")
# tone_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# # Define request format
# class ResumePayload(BaseModel):
#     candidate_id: str
#     resume_text: str

# @app.post("/process")
# def process_resume(data: ResumePayload):
#     # Extract entities
#     doc = nlp(data.resume_text)
#     entities = [ent.text for ent in doc.ents]

#     # Analyze tone
#     tone = tone_model(data.resume_text)[0]

#     # Generate similarity self score (optional demonstration)
#     emb = sbert.encode(data.resume_text)

#     result = {
#         "candidate_id": data.candidate_id,
#         "entities": entities,
#         "tone": tone,
#         "vector_size": len(emb),
#         "timestamp": datetime.utcnow().isoformat()
#     }
#     return result

# from fastapi import FastAPI
# from pydantic import BaseModel
# from datetime import datetime
# import spacy
# from sentence_transformers import SentenceTransformer
# from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer

# app = FastAPI(title="Recruitment AI Insight Engine")

# # Load models
# nlp = spacy.load("en_core_web_sm")
# sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# tone_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# # Schema for analysis request
# class AIRequest(BaseModel):
#     candidate_id: str
#     resume_text: str
#     strengths: str
#     weaknesses: str
#     missing_skills: str
#     tone: str

# @app.post("/generate-insights")
# def generate_ai_insights(req: AIRequest):
#     # Generate interview questions using missing skills and strengths
#     q_input = f"Generate interview questions for a candidate with strengths: {req.strengths}, missing skills: {req.missing_skills}"
#     q_inputs = tokenizer(q_input, return_tensors="pt")
#     q_outputs = t5_model.generate(**q_inputs, max_length=120)
#     questions = tokenizer.decode(q_outputs[0])

#     # Generate professional resume rewrite suggestions
#     r_input = "Rewrite professionally: " + req.resume_text
#     r_inputs = tokenizer(r_input, return_tensors="pt")
#     r_outputs = t5_model.generate(**r_inputs, max_length=150)
#     rewritten = tokenizer.decode(r_outputs[0])

#     # Compute semantic similarity score for strengths vs missing skills
#     resume_emb = sbert.encode(req.strengths)
#     skill_emb = sbert.encode(req.missing_skills)
#     gap_similarity = float((resume_emb @ skill_emb) / ((resume_emb @ resume_emb)**0.5 * (skill_emb @ skill_emb)**0.5))

#     result = {
#         "candidate_id": req.candidate_id,
#         "interview_questions": questions,
#         "resume_rewrite_suggestion": rewritten,
#         "semantic_gap_score": gap_similarity,
#         "gap_similarity_score": gap_similarity,
#         "processed_at": datetime.utcnow().isoformat()
#     }
#     return result
