import sqlite3
import bcrypt

DB_NAME = "finguard.db"

def init_db():
    """Creates the user database if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def hash_password(password):
    """Encrypts the password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed):
    """Verifies an encrypted password."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def add_user(email, password):
    """Adds a new user to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, password_hash) VALUES (?, ?)", 
                       (email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Email already exists
    finally:
        conn.close()

def verify_user(email):
    """Marks user as verified after OTP."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (email,))
    conn.commit()
    conn.close()
    
def get_user(email):
    """Retrieves a user's details."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    return user

# Initialize the database when this file is run
if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!")