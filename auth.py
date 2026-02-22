import os
from dotenv import load_dotenv
from passlib.context import CryptContext
from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load environment variables
load_dotenv()

# 1. Database Configuration (Matches your recruitment_ai DB)
DATABASE_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:root@localhost:3306/recruitment_ai")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. The User Model (Matches your MySQL screenshot)
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100))
    email = Column(String(150), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False) # 'applicant' or 'recruiter'
    created_at = Column(DateTime, default=datetime.utcnow)

# 3. Security Logic
# Using Bcrypt for industry-standard password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str):
    """Encodes a plain text password into a secure hash."""
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    """Checks if the entered password matches the stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

# 4. Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# from passlib.context import CryptContext
# from sqlalchemy import Column, Integer, String, DateTime, create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime

# # 1. Database Configuration (Matches your recruitment_ai DB mysql://root:iUXDzzzdIDKFoQAOrUyVIOQQVBrPyOPu@nozomi.proxy.rlwy.net:44730/railway")
# DATABASE_URL = "mysql+pymysql://root:root@localhost:3306/recruitment_ai"
# engine = create_engine(DATABASE_URL, pool_pre_ping=True)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # 2. The User Model (Matches your MySQL screenshot)
# class UserDB(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     full_name = Column(String(100))
#     email = Column(String(150), unique=True, nullable=False)
#     password = Column(String(255), nullable=False)
#     role = Column(String(50), nullable=False) # 'applicant' or 'recruiter'
#     created_at = Column(DateTime, default=datetime.utcnow)

# # 3. Security Logic
# # Using Bcrypt for industry-standard password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# def hash_password(password: str):
#     """Encodes a plain text password into a secure hash."""
#     return pwd_context.hash(password)

# def verify_password(plain_password, hashed_password):
#     """Checks if the entered password matches the stored hash."""
#     return pwd_context.verify(plain_password, hashed_password)

# # 4. Dependency to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
# from passlib.context import CryptContext
# from sqlalchemy.orm import Session
# from sqlalchemy import Column, Integer, String, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from datetime import datetime

# # Setup for the User Table
# Base = declarative_base()
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# class UserDB(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     full_name = Column(String(100))
#     email = Column(String(150), unique=True, nullable=False)
#     password = Column(String(255), nullable=False)
#     role = Column(String(50), nullable=False) # 'applicant' or 'recruiter'
#     created_at = Column(DateTime, default=datetime.utcnow)

# class AuthHandler:
#     @staticmethod
#     def get_password_hash(password):
#         return pwd_context.hash(password)

#     @staticmethod
#     def verify_password(plain_password, hashed_password):
#         return pwd_context.verify(plain_password, hashed_password)

#     @staticmethod
#     def register_user(db: Session, user_data: dict):
#         hashed_pwd = AuthHandler.get_password_hash(user_data['password'])
#         new_user = UserDB(
#             full_name=user_data.get('full_name'),
#             email=user_data['email'],
#             password=hashed_pwd,
#             role=user_data['role']
#         )
#         db.add(new_user)
#         db.commit()
#         db.refresh(new_user)
#         return new_user

#     @staticmethod
#     def authenticate_user(db: Session, email, password):
#         user = db.query(UserDB).filter(UserDB.email == email).first()
#         if not user or not AuthHandler.verify_password(password, user.password):
#             return None
#         return user