import hashlib
import sqlite3
from scripts.db import register_user, login_user

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register(username, password, role="user"):
    # normalize username
    username = username.strip().lower()

    if not username or not password:
        raise ValueError("Username and password required")

    hashed = hash_password(password)

    try:
        register_user(username, hashed, role)

    except sqlite3.IntegrityError:
        # This ONLY happens when username already exists
        raise ValueError("USERNAME_EXISTS")


def login(username, password):
    username = username.strip().lower()
    hashed = hash_password(password)
    return login_user(username, hashed)
