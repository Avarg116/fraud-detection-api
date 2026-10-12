"""Postgres connection pool and prediction inserts."""

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from fraud.api.schemas import Transaction

INSERT_PREDICTION = """
    INSERT INTO predictions (amount, features, fraud_score, decision, model_version, threshold)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id
"""


def open_pool(database_url: str) -> ConnectionPool:
    # Reusing a few open connections is faster than connecting on every request.
    pool = ConnectionPool(database_url, min_size=1, max_size=5, open=True)
    pool.wait(timeout=10)
    return pool


def save_prediction(
    pool: ConnectionPool,
    tx: Transaction,
    fraud_score: float,
    decision: str,
    model_version: str,
    threshold: float,
) -> int:
    features = tx.model_dump(exclude={"amount"})
    
    with pool.connection() as conn:
        row = conn.execute(
            INSERT_PREDICTION,
            (tx.amount, Jsonb(features), fraud_score, decision, model_version, threshold),
        ).fetchone()
    return row[0]
