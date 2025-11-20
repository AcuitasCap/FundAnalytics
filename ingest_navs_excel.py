import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus

from sqlalchemy import create_engine

# Fill these from the Shared Pooler section in Supabase
PG_HOST = "aws-1-ap-south-1.pooler.supabase.com"  # example; paste your exact host
PG_PORT = 5432                                   # pooler port from Supabase
PG_USER = "postgres.bmhlrjtkjevlpclfaqxl"                             # or your DB user
PG_PASSWORD = "Acuitas@777"               # no need to URL-encode here
PG_DBNAME = "postgres"

engine = create_engine(
    "postgresql+psycopg2://",
    connect_args={
        "host": PG_HOST,
        "port": PG_PORT,
        "user": PG_USER,
        "password": PG_PASSWORD,
        "dbname": PG_DBNAME,
        "sslmode": "require",
    },
    pool_pre_ping=True,
)

def _test_connection():
    with engine.begin() as conn:
        val = conn.execute(text("select 1")).scalar()
        print("DB test select 1 ->", val)

if __name__ == "__main__":
    _test_connection()
    # comment out the actual ingest first, then re-enable once test passes
    # ingest_funds_xlsx("Fund NAVs.xlsx")
    # ingest_bench_xlsx("BM NAVs.xlsx")

with engine.begin() as conn:
    print(conn.execute(text("select 1")).scalar())

# ---------- helpers ----------
def to_eom(s):
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.to_period("M").dt.to_timestamp("M").dt.date  # pure DATE

def coerce_num(s):
    return pd.to_numeric(s, errors="coerce")

def upsert_lookup(conn, table, key_col, values):
    """
    Insert any new lookup values (category/style), return {name -> id}.
    """
    vals = sorted({v for v in values if pd.notna(v) and str(v).strip() != ""})
    if not vals:
        return {}
    conn.execute(
        text(f"INSERT INTO fundlab.{table} ({key_col}) SELECT unnest(:vals) "
             f"ON CONFLICT ({key_col}) DO NOTHING"),
        {"vals": vals}
    )
    rows = conn.execute(text(
        f"SELECT * FROM fundlab.{table} WHERE {key_col} = ANY(:vals)"
    ), {"vals": vals}).mappings().all()
    id_col = [c for c in rows[0].keys() if c.endswith("_id")][0] if rows else None
    return {r[key_col]: r[id_col] for r in rows}

# ---------- funds ingest ----------
def ingest_funds_xlsx(path_funds_xlsx: str):
    df = pd.read_excel(path_funds_xlsx, engine="openpyxl")
    # Normalize column names to lower, and map your headings
    df.columns = [c.strip().lower() for c in df.columns]
    # Expecting: Fund, Date, NAV, Category, Style (case-insensitive)
    rename_map = {
        "fund": "fund_name",
        "date": "nav_date",
        "nav": "nav_value",
        "category": "fund_category",
        "style": "fund_style",
    }
    df = df.rename(columns=rename_map)

    # Basic checks
    needed = {"fund_name", "nav_date", "nav_value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in funds file: {missing}")

    df["nav_date"] = to_eom(df["nav_date"])
    df["nav_value"] = coerce_num(df["nav_value"])

    # Optional attributes
    cats = df["fund_category"].dropna().unique().tolist() if "fund_category" in df else []
    styles = df["fund_style"].dropna().unique().tolist() if "fund_style" in df else []

    with engine.begin() as conn:
        # upsert lookups
        cat_map = upsert_lookup(conn, "category", "category_name", cats)
        style_map = upsert_lookup(conn, "style", "style_name", styles)

        # ensure funds exist (IDs auto-assigned)
        for name in sorted({n for n in df["fund_name"] if pd.notna(n)}):
            conn.execute(text("""
                INSERT INTO fundlab.fund (fund_name) VALUES (:n)
                ON CONFLICT (fund_name) DO NOTHING
            """), {"n": name})

        # set category/style for each fund (first non-null seen)
        if "fund_category" in df or "fund_style" in df:
            for name, g in df.groupby("fund_name"):
                sets, params = [], {"name": name}
                if "fund_category" in g and g["fund_category"].notna().any():
                    cid = cat_map.get(g["fund_category"].dropna().iloc[0])
                    if cid is not None:
                        sets.append("category_id = :cid"); params["cid"] = cid
                if "fund_style" in g and g["fund_style"].notna().any():
                    sid = style_map.get(g["fund_style"].dropna().iloc[0])
                    if sid is not None:
                        sets.append("style_id = :sid"); params["sid"] = sid
                if sets:
                    conn.execute(text(f"""
                        UPDATE fundlab.fund SET {", ".join(sets)} WHERE fund_name = :name
                    """), params)

        # upsert NAV rows
        # Using simple per-row upsert for clarity; can switch to COPY temp table if you want speed
        ins = text("""
            INSERT INTO fundlab.fund_nav (fund_id, nav_date, nav_value)
            SELECT f.fund_id, :d, :v FROM fundlab.fund f WHERE f.fund_name=:n
            ON CONFLICT (fund_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            if pd.isna(r["fund_name"]) or pd.isna(r["nav_date"]) or pd.isna(r["nav_value"]):
                continue
            conn.execute(ins, {"n": r["fund_name"], "d": r["nav_date"], "v": r["nav_value"]})

    print("✅ Funds uploaded/updated.")

# ---------- benchmarks ingest ----------
def ingest_bench_xlsx(path_bench_xlsx: str):
    df = pd.read_excel(path_bench_xlsx, engine="openpyxl")
    df.columns = [c.strip().lower() for c in df.columns]
    # Expecting: BM name, Date, NAV, Category, Style
    rename_map = {
        "bm name": "benchmark_name",
        "date": "nav_date",
        "nav": "nav_value",
        "category": "fund_category",        # shares the same category lookup
        "style": "benchmark_style",         # shares the same style lookup
    }
    df = df.rename(columns=rename_map)

    needed = {"benchmark_name", "nav_date", "nav_value"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in benchmark file: {missing}")

    df["nav_date"] = to_eom(df["nav_date"])
    df["nav_value"] = coerce_num(df["nav_value"])

    cats = df["fund_category"].dropna().unique().tolist() if "fund_category" in df else []
    styles = df["benchmark_style"].dropna().unique().tolist() if "benchmark_style" in df else []

    with engine.begin() as conn:
        cat_map = upsert_lookup(conn, "category", "category_name", cats)
        style_map = upsert_lookup(conn, "style", "style_name", styles)

        # ensure benchmarks exist
        for name in sorted({n for n in df["benchmark_name"] if pd.notna(n)}):
            conn.execute(text("""
                INSERT INTO fundlab.benchmark (bench_name) VALUES (:n)
                ON CONFLICT (bench_name) DO NOTHING
            """), {"n": name})

        # set category/style (first non-null)
        if "fund_category" in df or "benchmark_style" in df:
            for name, g in df.groupby("benchmark_name"):
                sets, params = [], {"name": name}
                if "fund_category" in g and g["fund_category"].notna().any():
                    cid = cat_map.get(g["fund_category"].dropna().iloc[0])
                    if cid is not None:
                        sets.append("category_id = :cid"); params["cid"] = cid
                if "benchmark_style" in g and g["benchmark_style"].notna().any():
                    sid = style_map.get(g["benchmark_style"].dropna().iloc[0])
                    if sid is not None:
                        sets.append("style_id = :sid"); params["sid"] = sid
                if sets:
                    conn.execute(text(f"""
                        UPDATE fundlab.benchmark SET {", ".join(sets)} WHERE bench_name = :name
                    """), params)

        # upsert NAV rows
        ins = text("""
            INSERT INTO fundlab.bench_nav (bench_id, nav_date, nav_value)
            SELECT b.bench_id, :d, :v FROM fundlab.benchmark b WHERE b.bench_name=:n
            ON CONFLICT (bench_id, nav_date) DO UPDATE
            SET nav_value = EXCLUDED.nav_value
        """)
        for _, r in df.iterrows():
            if pd.isna(r["benchmark_name"]) or pd.isna(r["nav_date"]) or pd.isna(r["nav_value"]):
                continue
            conn.execute(ins, {"n": r["benchmark_name"], "d": r["nav_date"], "v": r["nav_value"]})

    print("✅ Benchmarks uploaded/updated.")

if __name__ == "__main__":
    # Change paths if needed
    ingest_funds_xlsx("Fund NAVs.xlsx")
    ingest_bench_xlsx("BM NAVs.xlsx")
