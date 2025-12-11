# -----------------storing llm output in llm_generated_info----------------
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
from groq import Groq
import warnings
import re
import time

# Suppress Pandas SQLAlchemy warning
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# --- Load environment variables ---
load_dotenv()

# --- Connect to PostgreSQL ---
try:
    conn = psycopg2.connect(
        host="localhost",
        dbname="exercises_db",
        user="postgres",
        password="1234",
        port=5432
    )
    print("✅ Connected to PostgreSQL Data Warehouse")
except Exception as e:
    print("❌ Failed to connect:", e)
    exit()

cur = conn.cursor()
cur.execute("SET search_path TO fitness_dw, public;")
conn.commit()

# --- Test DB connection ---
try:
    df_sample = pd.read_sql("SELECT * FROM combined_data LIMIT 5;", conn)
    print("\n📊 Sample data:\n", df_sample.head())
except Exception as e:
    print("⚠️ Could not load sample data:", e)

# --- Initialize LLM client ---
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(prompt, retries=2):
    """Send prompt to LLM with retry mechanism."""
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ LLM request failed (Attempt {attempt+1}/{retries}): {e}")
            time.sleep(2)
    return None


# --- Step 1: Take user query ---
user_query = input("\n💬 Enter your fitness or diet question: ").strip()
print(f"\n🔍 You asked: {user_query}")

# --- Extract number of days ---
match = re.search(r'(\d+)\s*-?\s*day', user_query.lower())
num_days = int(match.group(1)) if match else 3

# --- Detect limit ---
limit_match = re.search(r'show\s+(\d+)', user_query.lower())
result_limit = int(limit_match.group(1)) if limit_match else 5

# --- Fetch column names ---
cur.execute("""
    SELECT column_name FROM information_schema.columns 
    WHERE table_schema = 'fitness_dw' AND table_name = 'combined_data';
""")
columns = [r[0] for r in cur.fetchall()]
column_list = ", ".join(columns)

# --- Step 2: Generate SQL using LLM ---
sql_prompt = f"""
You are a PostgreSQL SQL expert.
Table: fitness_dw.combined_data
Columns: {column_list}

Convert the user question to SQL.
Rules:
1. Always SELECT specific columns from combined_data.
2. Use WHERE ... ILIKE '%keyword%' for text search.
3. Limit results to {result_limit}.
4. Output only SQL (no markdown or prefix text).
User question: "{user_query}"
"""

sql_query = (ask_llm(sql_prompt) or "").strip()
sql_query = sql_query.replace("```sql", "").replace("```", "")
sql_query = re.sub(r'^\s*sql\s+', '', sql_query, flags=re.IGNORECASE)
if not sql_query.endswith(";"):
    sql_query += ";"

print("\n🧠 LLM SQL Query:\n", sql_query)

# --- Step 3: Execute query or trigger schema expansion ---
try:
    df = pd.read_sql(sql_query, conn)

    if df.empty:
        print("\n⚠️ No DB match — generating new LLM-based info (schema expansion)...")

        fallback_prompt = f"""
You are a certified sports fitness and nutrition expert.
Database returned no results for: "{user_query}"

Generate a useful factual answer (short but informative).
"""
        llm_output = ask_llm(fallback_prompt)
        print(f"\n🤖 LLM-Generated Info:\n{llm_output}")

        # --- Check if schema expansion column exists ---
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='combined_data' AND column_name='llm_generated_info';
        """)
        if not cur.fetchone():
            print("🧩 Adding schema expansion column 'llm_generated_info'...")
            cur.execute("ALTER TABLE fitness_dw.combined_data ADD COLUMN llm_generated_info TEXT;")
            conn.commit()

        # --- Insert new row with LLM output ---
        cur.execute("""
            INSERT INTO fitness_dw.combined_data (llm_generated_info)
            VALUES (%s);
        """, (llm_output,))
        conn.commit()

        print("✅ Stored new LLM-generated info in 'llm_generated_info' column.")

    else:
        print(f"\n✅ Query Results Found ({len(df)} rows):")
        print(df.head(result_limit))

except Exception as e:
    print("\n❌ SQL execution failed:\n", e)



















# # ----------------- LLM fallback storage into llm_generated_responses -----------------

# import psycopg2
# import pandas as pd
# from dotenv import load_dotenv
# import os
# from groq import Groq
# import warnings
# import re
# import time

# warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# # ------------------- Load environment variables -------------------
# load_dotenv()

# # ------------------- Connect to PostgreSQL -------------------
# try:
#     conn = psycopg2.connect(
#         host="localhost",
#         dbname="exercises_db",
#         user="postgres",
#         password="1234",
#         port=5432
#     )
#     print("✅ Connected to PostgreSQL Data Warehouse")
# except Exception as e:
#     print("❌ Failed to connect:", e)
#     exit()

# cur = conn.cursor()
# cur.execute("SET search_path TO fitness_dw, public;")
# conn.commit()

# # ------------------- Ensure llm_generated_responses table exists -------------------
# cur.execute("""
# CREATE TABLE IF NOT EXISTS fitness_dw.llm_generated_responses (
#     id SERIAL PRIMARY KEY,
#     user_query TEXT NOT NULL,
#     llm_answer TEXT NOT NULL,
#     model_used TEXT DEFAULT 'llama-3.1-8b-instant',
#     created_at TIMESTAMP DEFAULT NOW()
# );
# """)
# conn.commit()


# # ------------------- Test DB Connection -------------------
# try:
#     df_sample = pd.read_sql("SELECT * FROM combined_data LIMIT 5;", conn)
#     print("\n📊 Sample data:\n", df_sample.head())
# except Exception as e:
#     print("⚠️ Could not load sample data:", e)


# # ------------------- LLM Client -------------------
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# def ask_llm(prompt, retries=2):
#     """Send prompt to LLM with retry mechanism."""
#     for attempt in range(retries):
#         try:
#             completion = client.chat.completions.create(
#                 model="llama-3.1-8b-instant",
#                 messages=[{"role": "user", "content": prompt}],
#                 temperature=0,
#                 timeout=60
#             )
#             return completion.choices[0].message.content.strip()
#         except Exception as e:
#             print(f"⚠️ LLM request failed (Attempt {attempt+1}/{retries}): {e}")
#             time.sleep(2)
#     return None


# # ------------------- Step 1: Get User Query -------------------
# user_query = input("\n💬 Enter your fitness or diet question: ").strip()
# print(f"\n🔍 You asked: {user_query}")

# # extract number of days
# match = re.search(r'(\d+)\s*-?\s*day', user_query.lower())
# num_days = int(match.group(1)) if match else 3

# # extract limit
# limit_match = re.search(r'show\s+(\d+)', user_query.lower())
# result_limit = int(limit_match.group(1)) if limit_match else 5


# # ------------------- Fetch Column Names -------------------
# cur.execute("""
#     SELECT column_name FROM information_schema.columns 
#     WHERE table_schema = 'fitness_dw' AND table_name = 'combined_data';
# """)
# columns = [r[0] for r in cur.fetchall()]
# column_list = ", ".join(columns)


# # ------------------- Step 2: Generate SQL using LLM -------------------
# sql_prompt = f"""
# You are a PostgreSQL expert.

# Convert the user question into a valid SQL query for:
#     fitness_dw.combined_data

# Rules:
# 1. Output ONLY a SQL SELECT statement.
# 2. Use WHERE ... ILIKE '%keyword%' when needed.
# 3. SQL MUST include:
#        SELECT <columns>
#        FROM fitness_dw.combined_data
# 4. LIMIT must be {result_limit}.
# 5. Never output explanation, only SQL.

# User query: "{user_query}"
# Columns: {column_list}
# """

# raw_sql = ask_llm(sql_prompt) or ""
# sql_query = raw_sql.replace("```sql", "").replace("```", "").strip()

# # clean incorrectly generated SQL
# sql_query = re.sub(r'--.*', '', sql_query)   # remove SQL comments
# sql_query = sql_query.split("\n")[0]         # first valid SQL line

# if not sql_query.lower().startswith("select"):
#     keyword = user_query.split()[-1]
#     sql_query = f"""
#         SELECT * FROM fitness_dw.combined_data
#         WHERE exercise_name ILIKE '%{keyword}%' 
#            OR food_name ILIKE '%{keyword}%'
#         LIMIT {result_limit};
#     """

# if not sql_query.endswith(";"):
#     sql_query += ";"

# print("\n🧠 LLM SQL Query:\n", sql_query)


# # ------------------- Step 3: Execute SQL -------------------
# try:
#     df = pd.read_sql(sql_query, conn)

#     if df.empty:
#         print("\n⚠️ No DB match — generating LLM fallback answer...")

#         fb_prompt = f"""
# You are a certified sports fitness and nutrition expert.
# Database returned no results for: "{user_query}"

# Provide a short, factual and useful answer.
# """
#         llm_output = ask_llm(fb_prompt)
#         print("\n🤖 LLM Answer:\n", llm_output)

#         # ------------------- Insert into llm_generated_responses -------------------
#         cur.execute("""
#             INSERT INTO fitness_dw.llm_generated_responses
#             (user_query, llm_answer, model_used)
#             VALUES (%s, %s, %s);
#         """, (
#             user_query,
#             llm_output,
#             "llama-3.1-8b-instant"
#         ))
#         conn.commit()

#         print("✅ Stored LLM output into fitness_dw.llm_generated_responses")

#     else:
#         print(f"\n✅ Query Results Found ({len(df)} rows):")
#         print(df.head(result_limit))

# except Exception as e:
#     print("\n❌ SQL execution failed:\n", e)


































# # final_updated_llm_sql_fallback.py
# # ----------------- FINAL UPDATED CODE (Entity detection + SQL auto-fix + no fuzzy matching) -----------------

# import os
# import re
# import time
# import warnings
# from dotenv import load_dotenv

# import pandas as pd
# import psycopg2
# from groq import Groq

# warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

# # ---------------------
# # Helper functions
# # ---------------------
# def extract_keyword_and_flags(user_query):
#     """Return (keyword, flags) where flags indicate intent."""
#     uq = user_query.lower()
#     is_unique = any(w in uq for w in ["unique", "distinct"])
#     is_exercise_query = any(w in uq for w in ["exercise", "workout", "training", "rep", "set"])
#     is_food_query = any(w in uq for w in ["food", "calorie", "fat", "protein", "carb", "carbohydrate", "nutrition", "meal"])
#     is_muscle_query = any(w in uq for w in ["muscle", "muscles", "pector", "pec", "biceps", "triceps", "chest", "legs", "back", "shoulder", "shoulders"])
#     # tokenize and remove some stopwords
#     tokens = re.findall(r"[a-zA-Z0-9]+", uq)
#     stopwords = {"give","me","for","with","value","how","much","in","show","list","name","names","the","a","an","of"}
#     filtered = [t for t in tokens if t not in stopwords]
#     keyword = filtered[-1] if filtered else uq
#     return keyword, is_unique, is_exercise_query, is_food_query, is_muscle_query

# def extract_sql_from_llm(text):
#     """
#     Extract the SQL portion if LLM returns SQL plus explanation.
#     We only keep the first contiguous block of lines that start with SQL keywords.
#     """
#     if not text:
#         return ""
#     t = text.replace("```sql", "").replace("```", "")
#     sql_lines = []
#     started = False
#     for line in t.splitlines():
#         s = line.strip()
#         low = s.lower()
#         if low.startswith(("select", "with", "insert", "update", "delete")):
#             started = True
#             sql_lines.append(s)
#         elif started:
#             # stop when encountering non-sql explanatory line
#             break
#     sql = "\n".join(sql_lines).strip()
#     if sql and not sql.endswith(";"):
#         sql += ";"
#     return sql

# def is_valid_sql(sql):
#     """Quick validator for SQL we will execute."""
#     if not sql:
#         return False
#     l = sql.lower()
#     return l.startswith("select") and "from" in l and ("combined_data" in l or "exercise" in l or "nutrition" in l)

# def build_safe_sql(keyword, table, domain_flag, distinct=False, limit=5):
#     """
#     Build a safe SQL query for a given domain:
#       - table: table name to query (exercise/nutrition/combined_data)
#       - domain_flag: 'exercise' or 'food' or 'muscle' used to choose columns
#     We always use ILIKE '%keyword%' searches and explicit selected columns.
#     """
#     sel = "*"
#     if distinct:
#         # choose appropriate distinct column
#         if domain_flag == "exercise":
#             sel = "DISTINCT exercise_name"
#         elif domain_flag == "food":
#             sel = "DISTINCT food_name"
#         else:
#             sel = "DISTINCT exercise_name"
#     else:
#         # choose columns to return (explicit)
#         if domain_flag == "exercise":
#             sel = "exercise_id, exercise_name, muscles, secondary_muscles, equipment"
#         elif domain_flag == "food":
#             sel = "food_name, energy_kcal, protein_g, carbohydrates_g, fat_g"
#         else:
#             # generic selection from combined_data
#             sel = "exercise_id, exercise_name, muscles, secondary_muscles, equipment, food_name, energy_kcal, protein_g, carbohydrates_g, fat_g"

#     # Build WHERE clause covering relevant text columns
#     where_clauses = []
#     if domain_flag == "exercise":
#         where_clauses = [
#             "exercise_name ILIKE %(kw)s",
#             "muscles ILIKE %(kw)s",
#             "secondary_muscles ILIKE %(kw)s"
#         ]
#     elif domain_flag == "food":
#         where_clauses = [
#             "food_name ILIKE %(kw)s"
#         ]
#     else:
#         where_clauses = [
#             "exercise_name ILIKE %(kw)s",
#             "muscles ILIKE %(kw)s",
#             "food_name ILIKE %(kw)s"
#         ]

#     where_sql = " OR ".join(where_clauses)
#     sql = f"SELECT {sel} FROM {table} WHERE {where_sql} LIMIT {limit};"
#     return sql

# def ensure_column_exists(conn, cur, table, column_name):
#     """Add llm_generated_info column if missing (for schema expansion)."""
#     cur.execute("""
#         SELECT column_name FROM information_schema.columns
#         WHERE table_schema = 'fitness_dw' AND table_name = %s AND column_name = %s;
#     """, (table, column_name))
#     if not cur.fetchone():
#         alter = f'ALTER TABLE fitness_dw.{table} ADD COLUMN {column_name} TEXT;'
#         cur.execute(alter)
#         conn.commit()
#         return True
#     return False

# def annotate_source(row):
#     """Return EXERCISE DATABASE or NUTRITION DATABASE or UNKNOWN based on row contents."""
#     # row may be a pandas Series
#     if "exercise_name" in row and pd.notna(row.get("exercise_name")) and str(row.get("exercise_name")).strip():
#         return "EXERCISE DATABASE"
#     if "food_name" in row and pd.notna(row.get("food_name")) and str(row.get("food_name")).strip():
#         return "NUTRITION DATABASE"
#     return "UNKNOWN SOURCE"

# # ---------------------
# # Setup: env, DB, LLM
# # ---------------------
# load_dotenv()
# GROQ_KEY = os.getenv("GROQ_API_KEY", "")
# client = Groq(api_key=GROQ_KEY)

# # Connect to the same PostgreSQL instance
# try:
#     conn = psycopg2.connect(
#         host="localhost",
#         dbname="exercises_db",
#         user="postgres",
#         password="1234",
#         port=5432
#     )
#     print("✅ Connected to PostgreSQL")
# except Exception as e:
#     print("❌ DB connection failed:", e)
#     raise SystemExit(1)

# cur = conn.cursor()
# cur.execute("SET search_path TO fitness_dw, public;")
# conn.commit()

# # Detect which logical tables are available (exercise / nutrition / combined_data)
# cur.execute("""
#     SELECT table_name FROM information_schema.tables
#     WHERE table_schema = 'fitness_dw';
# """)
# available = {r[0].lower() for r in cur.fetchall()}

# has_exercise_table = 'exercise' in available
# has_nutrition_table = 'nutrition' in available
# has_combined = 'combined_data' in available

# # If none of the above exist, we'll default to combined_data but warn.
# if not (has_exercise_table or has_nutrition_table or has_combined):
#     print("⚠️ Warning: No expected tables found in schema 'fitness_dw'. Expected 'exercise', 'nutrition' or 'combined_data'.")
#     # still proceed but queries will likely fail

# # ---------------------
# # Ask user
# # ---------------------
# user_query = input("\n💬 Enter your fitness or diet question: ").strip()
# print("\n🔍 You asked:", user_query)

# # extract keyword and flags
# keyword, is_unique, is_exercise_query, is_food_query, is_muscle_query = extract_keyword_and_flags(user_query)
# print(f"🔑 Extracted keyword: {keyword}  | flags -> unique:{is_unique} exercise:{is_exercise_query} food:{is_food_query} muscle:{is_muscle_query}")

# limit = 5

# # 1) Try: ask LLM to build SQL
# sql_prompt = f"""
# You are a PostgreSQL assistant. Produce ONLY one valid SQL SELECT query for PostgreSQL.
# User request: "{user_query}"
# Schema: fitness_dw
# Available columns (combined_data or exercise/nutrition): exercise_id, exercise_name, muscles, secondary_muscles, equipment, food_name, energy_kcal, protein_g, carbohydrates_g, fat_g

# Rules:
# - Output only a single SELECT query (no explanations).
# - Use ILIKE '%keyword%' for text matching.
# - Include LIMIT {limit}.
# - If user asked for uniqueness (words 'unique' or 'distinct'), include DISTINCT on the right column (exercise_name or food_name).
# - Use combined_data if you must; otherwise it's OK to reference exercise or nutrition tables if present.

# Keyword to search: {keyword}
# """
# # ask LLM (non-blocking safety: we will not run broken SQL)
# try:
#     llm_resp = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[{"role":"user", "content": sql_prompt}],
#         temperature=0,
#         timeout=40
#     )
#     llm_text = llm_resp.choices[0].message.content.strip()
# except Exception as e:
#     print("⚠️ LLM call failed:", e)
#     llm_text = ""

# # Extract SQL from LLM output (if any)
# llm_sql = extract_sql_from_llm(llm_text)
# if llm_sql:
#     print("\n🧠 LLM produced SQL (extracted):")
#     print(llm_sql)
# else:
#     print("\n🧠 LLM produced no usable SQL; will auto-build a safe SQL.")

# # Decide domain/table to query according to entity detection (preference)
# # Priority:
# #  - muscle -> exercise table (if exists) else combined_data
# #  - exercise query -> exercise table (if exists) else combined_data
# #  - food query -> nutrition table (if exists) else combined_data
# #  - fallback: combined_data if exists else exercise then nutrition
# domain = None
# table_to_query = None

# if is_muscle_query or is_exercise_query:
#     domain = "exercise"
#     if has_exercise_table:
#         table_to_query = "exercise"
#     elif has_combined:
#         table_to_query = "combined_data"
#     else:
#         table_to_query = "exercise" if has_exercise_table else ("nutrition" if has_nutrition_table else "combined_data")
# elif is_food_query:
#     domain = "food"
#     if has_nutrition_table:
#         table_to_query = "nutrition"
#     elif has_combined:
#         table_to_query = "combined_data"
#     else:
#         table_to_query = "nutrition" if has_nutrition_table else ("exercise" if has_exercise_table else "combined_data")
# else:
#     # unknown intent: prefer combined_data, else exercise table, else nutrition
#     domain = "mixed"
#     if has_combined:
#         table_to_query = "combined_data"
#     elif has_exercise_table:
#         table_to_query = "exercise"
#     elif has_nutrition_table:
#         table_to_query = "nutrition"
#     else:
#         table_to_query = "combined_data"

# # If LLM SQL is valid and references allowed table(s), use it; otherwise build safe SQL
# use_sql = None
# if llm_sql and is_valid_sql(llm_sql):
#     # ensure the LLM query references the table we are allowed to query (fitness_dw.*)
#     llm_l = llm_sql.lower()
#     allowed_tables = {table_to_query.lower()}
#     # allow combined_data too if present
#     if has_combined:
#         allowed_tables.add("combined_data")
#     # check if any allowed table name appears
#     if any(tbl in llm_l for tbl in allowed_tables):
#         use_sql = llm_sql
#     else:
#         # LLM SQL references a different table; ignore it and auto-build
#         use_sql = None

# if use_sql is None:
#     # Auto-build safe SQL per domain and flags
#     sql_domain = domain
#     if domain == "exercise" and not (has_exercise_table or has_combined):
#         # fallback to nutrition if nothing else
#         sql_domain = "food" if has_nutrition_table else "mixed"
#     if domain == "food" and not (has_nutrition_table or has_combined):
#         sql_domain = "exercise" if has_exercise_table else "mixed"

#     # pick table
#     tbl = None
#     if sql_domain == "exercise":
#         tbl = "exercise" if has_exercise_table else ("combined_data" if has_combined else "exercise")
#     elif sql_domain == "food":
#         tbl = "nutrition" if has_nutrition_table else ("combined_data" if has_combined else "nutrition")
#     else:
#         tbl = "combined_data" if has_combined else ( "exercise" if has_exercise_table else "nutrition" )

#     # Build SQL
#     use_sql = build_safe_sql(keyword, tbl, sql_domain, distinct=is_unique, limit=limit)
#     print("\n🛠️ Auto-built safe SQL:")
#     print(use_sql)
# else:
#     print("\n✅ Using LLM SQL as-is.")

# # Run the SQL safely and only once (no fuzzy matching)
# df = pd.DataFrame()
# params = {"kw": f"%{keyword}%"}
# try:
#     df = pd.read_sql(use_sql, conn, params=params)
# except Exception as e:
#     print("\n❌ SQL execution failed:", e)
#     # try a very safe fallback using combined_data if available
#     if has_combined:
#         fallback_sql = build_safe_sql(keyword, "combined_data", "mixed", distinct=is_unique, limit=limit)
#         print("\n🔁 Trying fallback combined_data SQL:")
#         print(fallback_sql)
#         try:
#             df = pd.read_sql(fallback_sql, conn, params=params)
#         except Exception as e2:
#             print("❌ Fallback SQL failed:", e2)
#             df = pd.DataFrame()

# # If we have results, annotate with source and print
# if not df.empty:
#     # Normalize columns to include exercise_name / food_name where possible
#     # Add SOURCE_DB column based on row content
#     out = df.head(limit).copy()
#     out["SOURCE_DB"] = out.apply(annotate_source, axis=1)
#     print("\n✅ RESULTS (from database):\n")
#     print(out.to_string(index=False))
# else:
#     # No results found in DB → call LLM fallback to answer directly and store
#     print("\n🚫 No rows returned by DB queries. Invoking LLM fallback to generate an expert answer...")

#     fb_prompt = f"Database has no structured info for: '{user_query}'. Provide a short factual expert answer (fitness + nutrition domain)."
#     try:
#         fb_resp = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role":"user","content": fb_prompt}],
#             temperature=0,
#             timeout=40
#         )
#         answer = fb_resp.choices[0].message.content.strip()
#     except Exception as e:
#         print("⚠️ LLM fallback failed:", e)
#         answer = "Sorry — I couldn't find matching data and the LLM fallback failed."

#     print("\n🤖 LLM Answer:\n")
#     print(answer)

#     # store into schema expansion column llm_generated_info on combined_data (or target table)
#     target_table = "combined_data" if has_combined else ( "exercise" if has_exercise_table else ("nutrition" if has_nutrition_table else "combined_data") )
#     try:
#         ensure_column_exists(conn, cur, target_table, "llm_generated_info")
#         cur.execute(f"INSERT INTO fitness_dw.{target_table} (llm_generated_info) VALUES (%s);", (answer,))
#         conn.commit()
#         print(f"\n✅ Stored LLM answer into {target_table}.llm_generated_info")
#     except Exception as e:
#         print("⚠️ Could not store LLM answer into DB:", e)

# # close DB cursor/conn gracefully
# try:
#     cur.close()
#     conn.close()
# except:
#     pass

# # End of script
