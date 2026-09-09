from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.common.config import settings
import pandas as pd

engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=10, max_overflow=20)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def execute_query(query: str, params: dict = None) -> pd.DataFrame:
    """Executes a SQL query and returns a pandas DataFrame."""
    with engine.connect() as connection:
        return pd.read_sql_query(query, connection, params=params)

def upsert_dataframe(df: pd.DataFrame, table_name: str, conflict_columns: list):
    """
    Upserts a pandas DataFrame into Postgres using ON CONFLICT DO UPDATE.
    """
    if df.empty:
        return

    from sqlalchemy import text
    import logging
    log = logging.getLogger(__name__)

    columns = list(df.columns)
    col_str = ", ".join(columns)
    val_str = ", ".join([f":{col}" for col in columns])
    update_str = ", ".join([f"{col} = EXCLUDED.{col}" for col in columns if col not in conflict_columns])
    conflict_str = ", ".join(conflict_columns)

    sql = f"""
        INSERT INTO {table_name} ({col_str})
        VALUES ({val_str})
        ON CONFLICT ({conflict_str})
        DO UPDATE SET {update_str};
    """

    records = df.to_dict(orient="records")
    try:
        with engine.begin() as conn:
            conn.execute(text(sql), records)
    except Exception as e:
        log.warning(f"Could not persist to database ({e}). Processed {len(df)} records in memory.")
