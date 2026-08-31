import sqlite3

conn = sqlite3.connect("lost_found.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT,
    description TEXT,
    image_name TEXT,
    prediction TEXT,
    confidence REAL,
    lost_date TEXT,
    location TEXT,
    contact TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS found_items(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_name TEXT,
    description TEXT,
    found_date TEXT,
    location TEXT,
    contact TEXT,
    image_name TEXT,
    prediction TEXT,
    confidence REAL
)
""")

conn.commit()
conn.close()

print("Database Created Successfully!")