import sqlite3

conn = sqlite3.connect("lost_found.db")
cursor = conn.cursor()

cursor.execute("DELETE FROM items")
cursor.execute("DELETE FROM found_items")

conn.commit()
conn.close()

print("All uploaded items deleted successfully!")