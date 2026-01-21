import sqlite3

DB_NAME = "tickets.db"

# =====================================
# DATABASE CONNECTION
# =====================================
def get_connection():
    """
    Creates and returns a SQLite database connection.
    check_same_thread=False allows Streamlit to access DB safely.
    """
    return sqlite3.connect(DB_NAME, check_same_thread=False)


# =====================================
# CREATE TICKETS TABLE
# =====================================
def create_table():
    """
    Creates the tickets table if it does not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Open',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME
        )
    """)

    conn.commit()
    conn.close()


# =====================================
# CREATE USERS TABLE
# =====================================
def create_user_table():
    """
    Creates the users table for authentication.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)

    conn.commit()
    conn.close()


# =====================================
# INSERT NEW TICKET
# =====================================
def insert_ticket(title, description, category, priority):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO tickets (title, description, category, priority)
        VALUES (?, ?, ?, ?)
    """, (title, description, category, priority))

    conn.commit()
    conn.close()


# =====================================
# FETCH ACTIVE TICKETS
# =====================================
def fetch_active_tickets():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM tickets
        WHERE status != 'Closed'
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


# =====================================
# FETCH CLOSED TICKETS
# =====================================
def fetch_closed_tickets():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM tickets
        WHERE status = 'Closed'
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return rows


# =====================================
# UPDATE TICKET STATUS
# =====================================
def update_status(ticket_id, status):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE tickets
        SET status = ?, updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (status, ticket_id))

    conn.commit()
    conn.close()


# =====================================
# ANALYTICS COUNTS
# =====================================
def get_counts():
    conn = get_connection()
    cursor = conn.cursor()

    total = cursor.execute(
        "SELECT COUNT(*) FROM tickets"
    ).fetchone()[0]

    open_tickets = cursor.execute(
        "SELECT COUNT(*) FROM tickets WHERE status = 'Open'"
    ).fetchone()[0]

    high_priority = cursor.execute(
        "SELECT COUNT(*) FROM tickets WHERE priority = 'High'"
    ).fetchone()[0]

    closed_tickets = cursor.execute(
        "SELECT COUNT(*) FROM tickets WHERE status = 'Closed'"
    ).fetchone()[0]

    conn.close()

    return {
        "total": total,
        "open": open_tickets,
        "high": high_priority,
        "closed": closed_tickets
    }


# =====================================
# REGISTER USER
# =====================================
def register_user(username, hashed_password, role="user"):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO users (username, password, role)
        VALUES (?, ?, ?)
    """, (username, hashed_password, role))

    conn.commit()
    conn.close()


# =====================================
# LOGIN USER
# =====================================
def login_user(username, hashed_password):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, role
        FROM users
        WHERE username = ? AND password = ?
    """, (username, hashed_password))

    user = cursor.fetchone()
    conn.close()
    return user
