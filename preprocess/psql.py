import os
import time
import psycopg2.pool

DEBUG = False
DATABASE_URL = ""

pool = psycopg2.pool.SimpleConnectionPool(dsn=DATABASE_URL, minconn=2, maxconn=16)
def connect_db(db_type=None):
    conn = None
    while not conn:
        try:
            conn = pool.getconn()
            print("success")
        except psycopg2.pool.PoolError:
            print("cannot get a connection ... retry")
            time.sleep(1)

    return conn

def close_db(db):
    pool.putconn(db)

def create_table(db, query):
    cursor = db.cursor()
    cursor.execute(query)
    db.commit()
    cursor.close()

def drop_table(db, query, verbose=True):
    cursor = db.cursor()
    try:
        cursor.execute(query)
        db.commit()
        print(f"{query} ... fin")
    finally:
        cursor.close()
        print("failed")

def insert_col(db, query):
    cursor = db.cursor()
    cursor.execute(query)
    db.commit()
    cursor.close()


def bool4sql(target):
    return "true" if target else "false"


def get_raw_output_paths(systems=None):
    _systems = systems or [
        "sat",
        "ort",
        "clipcap_mlp",
        "clipcap_trm",
        "trm3",
        "trm6",
        "trm12",
        "m2",
        "ersan",
        "dlct",
        "random"
    ]
    systems = _systems
    paths = {
        system: os.path.join(
            "system/",
            system +
            "_eval.json") for system in systems}
    return paths