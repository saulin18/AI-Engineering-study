import sqlite3
from pathlib import Path
import aiohttp


database_url = (
    "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
)

# We'll be using a local sqlite database but in production you would use a remote database
local_path = Path("Chinook.db")


async def get_or_create_db(local_path: Path):
    if not local_path.exists():
        # If you only want to have local database only create the file with touch
        # local_path.touch()
        async with aiohttp.ClientSession() as session:
            response = await session.get(database_url)
            if response.status == 200:
                local_path.write_bytes(response.content)
                print(f"File downloaded and saved as {local_path}")
            else:
                print(
                    f"Failed to download the file. Status code: {response.status}"
                )


def test_remote_db():
    """
    Test if remote db has the correct tables, print tables out to the console
    """
    conn = sqlite3.connect(local_path.absolute())
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
    print("Dialect: sqlite")
    print(f"Available tables: {tables}")
    cursor.execute("SELECT * FROM Artist LIMIT 5;")
    print(f"Sample output: {cursor.fetchall()}")
    conn.close()
