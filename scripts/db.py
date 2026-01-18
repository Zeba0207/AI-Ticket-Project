import sqlite3

DB_NAME = "tickets.db"

# =====================================
# DATABASE CONNECTION
# =====================================
def get_connection():
    return sqlite3.connect(DB_NAME, check_same_thread=False)

# =====================================
# CREATE TABLE
# =====================================
def create_table():
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
# INSERT TICKET
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
# UPDATE STATUS
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

    closed = cursor.execute(
        "SELECT COUNT(*) FROM tickets WHERE status = 'Closed'"
    ).fetchone()[0]

    conn.close()

    return {
        "total": total,
        "open": open_tickets,
        "high": high_priority,
        "closed": closed
    }
