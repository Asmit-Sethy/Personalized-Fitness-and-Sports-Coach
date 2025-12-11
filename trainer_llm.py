# # =======================BEST======================
# # ----trainer_llm.py-----
# import psycopg2
# import pandas as pd
# from dotenv import load_dotenv
# import os, json, time, re
# from groq import Groq
# import warnings

# warnings.filterwarnings("ignore")

# # -------------------------------------------------------------
# # JSON SANITIZER & HELPERS
# # -------------------------------------------------------------
# def to_jsonable(obj):
#     if isinstance(obj, dict):
#         return {k: to_jsonable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [to_jsonable(x) for x in obj]
#     if isinstance(obj, (str, int, float, bool)) or obj is None:
#         return obj
#     return str(obj)


# def safe_json_parse(text: str, default: dict) -> dict:
#     """
#     Robust JSON parser for LLM output.

#     Strategy:
#       1. If empty -> default + raw_text=None
#       2. Strip  / json fences if present
#       3. Cut from first '{' to last '}' (ignore extra prose)
#       4. Try json.loads(cleaned)
#       5. If that fails, remove trailing commas before ']' and '}' and try again
#       6. If still fails, try json.loads(raw)
#       7. On failure -> default + raw_text=text
#     """
#     if not text:
#         out = dict(default)
#         out["raw_text"] = None
#         return out

#     raw = text
#     t = raw.strip()

#     # 1) strip leading  or json line
#     if t.startswith(""):
#         newline_idx = t.find("\n")
#         if newline_idx != -1:
#             t = t[newline_idx + 1 :].strip()

#     # 2) strip trailing 
#     if t.endswith("```"):
#         t = t[:-3].strip()

#     # 3) cut from first '{' to last '}'
#     start = t.find("{")
#     end = t.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         candidate = t[start : end + 1]
#     else:
#         candidate = t

#     # helper to attempt parse with optional comma massage
#     def try_parse(s: str):
#         try:
#             return json.loads(s)
#         except Exception:
#             # remove trailing commas before ] or }
#             s2 = re.sub(r",(\s*[\]}])", r"\1", s)
#             try:
#                 return json.loads(s2)
#             except Exception:
#                 return None

#     # 4) cleaned candidate
#     parsed = try_parse(candidate)
#     if parsed is not None:
#         return parsed

#     # 6) raw
#     try:
#         return json.loads(raw)
#     except Exception:
#         out = dict(default)
#         out["raw_text"] = raw
#         return out


# # -------------------------------------------------------------
# # CONFIG
# # -------------------------------------------------------------
# load_dotenv()

# DB_CONFIG = {
#     "host": "localhost",
#     "dbname": "trainer_dw",
#     "user": "postgres",
#     "password": "1234",
#     "port": 5432
# }

# GROQ_MODEL = "llama-3.1-8b-instant"
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

# # -------------------------------------------------------------
# # DB CONNECTION
# # -------------------------------------------------------------
# conn = psycopg2.connect(**DB_CONFIG)
# cur = conn.cursor()
# print("✅ Connected to trainer_dw")

# # -------------------------------------------------------------
# # LLM CALL
# # -------------------------------------------------------------
# def call_llm(prompt, temperature=0.0):
#     """Generic LLM call: returns (text, meta_dict)."""
#     try:
#         completion = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature
#         )
#         txt = completion.choices[0].message.content.strip()
#         usage = getattr(completion, "usage", None)
#         meta = {}
#         if usage:
#             meta = {
#                 "prompt_tokens": getattr(usage, "prompt_tokens", None),
#                 "completion_tokens": getattr(usage, "completion_tokens", None),
#                 "total_tokens": getattr(usage, "total_tokens", None),
#             }
#         return txt, meta
#     except Exception as e:
#         print("❌ LLM ERROR:", e)
#         return None, {"error": str(e)}

# # -------------------------------------------------------------
# # USER INPUT
# # -------------------------------------------------------------
# user_query = input("\n💬 Enter your question: ").strip()
# print("\n📌 Your Query:", user_query)

# # -------------------------------------------------------------
# # STEP 1 — DECOMPOSITION
# # -------------------------------------------------------------
# def decompose_query(q):
#     prompt = f"""
# You MUST return ONLY JSON. No commentary, no markdown.

# Extract structured details from this fitness question:

# {q}

# Return EXACTLY this JSON structure (fill in values, keep keys):

# {{
#   "focus_muscles": [],
#   "exercise_goal": "",
#   "nutrient_limits": {{
#     "protein_g_min": null,
#     "protein_g_max": null,
#     "fat_g_min": null,
#     "fat_g_max": null,
#     "energy_kcal_max": null
#   }},
#   "duration_days": null,
#   "intensity": "",
#   "dietary_restrictions": []
# }}
# """
#     txt, meta = call_llm(prompt, temperature=0.0)

#     default = {
#         "focus_muscles": [],
#         "exercise_goal": "",
#         "nutrient_limits": {
#             "protein_g_min": None,
#             "protein_g_max": None,
#             "fat_g_min": None,
#             "fat_g_max": None,
#             "energy_kcal_max": None
#         },
#         "duration_days": None,
#         "intensity": "",
#         "dietary_restrictions": []
#     }

#     parsed = safe_json_parse(txt, default)
#     # ensure all keys exist
#     for k, v in default.items():
#         if k not in parsed:
#             parsed[k] = v

#     return parsed, txt, meta

# decomp, raw_decomp, decomp_meta = decompose_query(user_query)
# decomp = to_jsonable(decomp)
# print("\n🧩 DECOMPOSED QUERY:\n", json.dumps(decomp, indent=2))

# # -------------------------------------------------------------
# # STEP 2 — RESOLVE QUERY (FDW -> dw.resolve_query)
# # -------------------------------------------------------------
# resolve_sql = """
# SELECT canonical_id, entity_type, exercise_data, nutrition_data
# FROM dw.resolve_query(%s);
# """

# try:
#     df = pd.read_sql(resolve_sql, conn, params=[user_query])
# except Exception as e:
#     print("⚠ Resolver failed:", e)
#     df = pd.DataFrame(columns=["canonical_id","entity_type","exercise_data","nutrition_data"])

# exercise_list = []
# nutrition_list = []

# for _, row in df.iterrows():
#     if row["exercise_data"] not in (None, {}, []):
#         exercise_list.append(to_jsonable(row["exercise_data"]))
#     if row["nutrition_data"] not in (None, {}, []):
#         nutrition_list.append(to_jsonable(row["nutrition_data"]))

# print("\n📦 Exercise matches (sample):", exercise_list[:2])
# print("🍽 Nutrition matches (sample):", nutrition_list[:2])

# # -------------------------------------------------------------
# # STEP 3 — GAP DETECTION
# # -------------------------------------------------------------
# nut_limits = decomp.get("nutrient_limits", {}) or {}

# gaps = {
#     "exercise_missing": len(exercise_list) == 0,
#     "nutrition_missing": len(nutrition_list) == 0,
#     "needs_nutrient_planning": any(
#         v is not None for v in [
#             nut_limits.get("protein_g_min"),
#             nut_limits.get("protein_g_max"),
#             nut_limits.get("fat_g_min"),
#             nut_limits.get("fat_g_max"),
#             nut_limits.get("energy_kcal_max"),
#         ]
#     ),
#     "nutrition_missing_macros_for": []
# }

# for food in nutrition_list:
#     if not isinstance(food, dict):
#         continue
#     if food.get("protein_g") is None or food.get("fat_g") is None or food.get("energy_kcal") is None:
#         gaps["nutrition_missing_macros_for"].append(food.get("name", "unknown"))

# print("\n🕳 GAP STATUS:\n", json.dumps(gaps, indent=2))

# # -------------------------------------------------------------
# # STEP 4 — GAP-FILL PROMPT
# # -------------------------------------------------------------
# DEFAULT_GAP_OUTPUT = {
#     "reason": "",
#     "exercise_plan": [],
#     "diet_plan": [],
#     "generated_attrs": {
#         "serving_for_target_protein": "",
#         "additional_notes": ""
#     },
#     "score": 0.8,
#     "confidence": 0.8,
#     "summary": ""
# }

# gap_prompt = f"""
# You are a fitness + nutrition expert. You help expand a VIRTUAL table
# exercise_food_recommendation that combines exercises and foods.

# USER QUERY:
# {user_query}

# DATABASE EXERCISES (JSON array of rows):
# {json.dumps(exercise_list, indent=2)}

# DATABASE FOODS (JSON array of rows):
# {json.dumps(nutrition_list, indent=2)}

# DECOMPOSED QUERY (parameters):
# {json.dumps(decomp, indent=2)}

# GAPS DETECTED:
# {json.dumps(gaps, indent=2)}

# TASK:
# 1. USE the given DB rows as ground truth where possible.
# 2. If macros or attributes are missing for any DB food, IMPUTE reasonable values.
# 3. If there are NO suitable foods in the DB, you MUST still propose NEW snacks (virtual rows).
# 4. Plan an exercise routine + diet/snack suggestions that satisfy the user's constraints
#    (e.g., protein >= given minimum, fat <= limit, gluten-free etc.).
# 5. Treat the answer as MATERIALIZING a new row in an "exercise_food_recommendation" table.

# IMPORTANT CONSTRAINTS:
# - The array "exercise_plan" MUST contain AT LEAST ONE item.
# - The array "diet_plan" MUST contain AT LEAST ONE item.
# - DO NOT return empty arrays for exercise_plan or diet_plan.

# RETURN STRICT JSON ONLY with exactly these keys (no extra keys, no text outside JSON):

# {{
#   "reason": "",
#   "exercise_plan": [
#     {{
#       "exercise_name": "",
#       "sets": 3,
#       "reps": 10,
#       "notes": ""
#     }}
#   ],
#   "diet_plan": [
#     {{
#       "name": "",
#       "description": "",
#       "macros": {{
#         "protein_g": 0.0,
#         "fat_g": 0.0,
#         "energy_kcal": 0.0
#       }},
#       "gluten_free": true,
#       "serving_size": ""
#     }}
#   ],
#   "generated_attrs": {{
#       "serving_for_target_protein": "",
#       "additional_notes": ""
#   }},
#   "score": 0.0,
#   "confidence": 0.0,
#   "summary": ""
# }}
# """

# # -------------------------------------------------------------
# # STEP 5 — LLM CALL FOR GAP FILLING
# # -------------------------------------------------------------
# llm_raw, llm_usage = call_llm(gap_prompt, temperature=0.2)
# print("\n🧠 RAW LLM GAP OUTPUT (first 400 chars):\n", (llm_raw or "")[:400])

# llm_json = safe_json_parse(llm_raw, DEFAULT_GAP_OUTPUT)
# for k, v in DEFAULT_GAP_OUTPUT.items():
#     if k not in llm_json:
#         llm_json[k] = v

# # -------------------------------------------------------------
# # STEP 6 — STRUCTURED LLM OUTPUT
# # -------------------------------------------------------------
# reason = llm_json.get("reason")
# exercise_plan = llm_json.get("exercise_plan", [])
# diet_plan = llm_json.get("diet_plan", [])
# generated_attrs = llm_json.get("generated_attrs", {})
# score = llm_json.get("score", DEFAULT_GAP_OUTPUT["score"])
# confidence = llm_json.get("confidence", DEFAULT_GAP_OUTPUT["confidence"])
# summary = llm_json.get("summary")

# print("\n🏋 EXERCISE PLAN FROM LLM:\n", json.dumps(exercise_plan, indent=2))
# print("\n🍎 DIET PLAN FROM LLM:\n", json.dumps(diet_plan, indent=2))

# llm_struct = {
#     "user_query": user_query,
#     "decomposition": decomp,
#     "gaps": gaps,
#     "db_exercise_sample": exercise_list[:3],
#     "db_nutrition_sample": nutrition_list[:3],
#     "reason": reason,
#     "exercise_plan": exercise_plan,
#     "diet_plan": diet_plan,
#     "generated_attrs": generated_attrs,
#     "score": score,
#     "confidence": confidence,
#     "summary": summary,
#     "prompt": gap_prompt,
#     "model": GROQ_MODEL,
#     "usage": llm_usage
# }
# llm_struct = to_jsonable(llm_struct)

# # -------------------------------------------------------------
# # STEP 7 — STORE IN MATERIALIZED TABLE
# # -------------------------------------------------------------
# insert_sql = f"""
# INSERT INTO {MATERIALIZED_TABLE}
# (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
# VALUES (%s, %s, %s, %s, %s, %s, %s)
# RETURNING fact_id;
# """

# exercise_json = exercise_list[0] if exercise_list else None
# nutrition_json = nutrition_list[0] if nutrition_list else None

# cur.execute(
#     insert_sql,
#     (
#         None,
#         "exercise_food_recommendation",
#         json.dumps(exercise_json) if exercise_json else None,
#         json.dumps(nutrition_json) if nutrition_json else None,
#         json.dumps(llm_struct),
#         float(score) if score is not None else 0.8,
#         float(confidence) if confidence is not None else 0.8
#     )
# )

# fid = cur.fetchone()[0]
# conn.commit()

# print("\n💾 Saved fact_id:", fid)

# # -------------------------------------------------------------
# # STEP 8 — PRINT FINAL OUTPUT
# # -------------------------------------------------------------
# final_out = {
#     "fact_id": fid,
#     "user_query": user_query,
#     "score": score,
#     "confidence": confidence,
#     "summary": summary,
#     "diet_plan_sample": diet_plan[:3] if isinstance(diet_plan, list) else []
# }

# print("\n🎉 FINAL OUTPUT (compact):")
# print(json.dumps(final_out, indent=2))

# # -------------------------------------------------------------
# # CLEANUP
# # -------------------------------------------------------------
# cur.close()
# conn.close()
# print("\n🔚 Connection closed.")











# ============================BEST===========================
# import psycopg2
# import pandas as pd
# from dotenv import load_dotenv
# import os, json, re
# from groq import Groq

# load_dotenv()

# DB_CONFIG = {
#     "host": "localhost",
#     "dbname": "trainer_dw",
#     "user": "postgres",
#     "password": "1234",
#     "port": 5432
# }

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# conn = psycopg2.connect(**DB_CONFIG)
# cur = conn.cursor()
# print("✅ Connected to trainer_dw")


# # ------------------------- SIMPLE PARSER -----------------------------
# def parse_query(q: str):
#     q = q.lower()
#     muscle = None
#     ask_protein = False
#     food_name = None
#     exercise_name = None
#     ask_equipment = False

#     muscles_list = ["chest", "back", "legs", "biceps", "triceps", "shoulders", "abs", "glutes"]
#     for m in muscles_list:
#         if m in q:
#             muscle = m
#             break

#     if "protein" in q: ask_protein = True
#     if "equipment" in q: ask_equipment = True

#     food_keywords = ["food", "eat", "diet"]
#     if any(f in q for f in food_keywords):
#         words = q.split()
#         food_name = words[-1]

#     if "exercise" in q:
#         words = q.split()
#         exercise_name = words[-1]
    
#     return {
#         "muscle": muscle,
#         "exercise_name": exercise_name,
#         "food_name": food_name,
#         "ask_protein": ask_protein,
#         "ask_exercise_equipment": ask_equipment
#     }


# # ------------------------- STRUCTURED DB SEARCH -----------------------------
# def search_exercises(parsed):
#     if not parsed["muscle"]:
#         return pd.DataFrame()

#     sql = """
#         SELECT id, name, muscles, muscles_secondary, equipment
#         FROM exercise_fdw.exercises
#         WHERE %s = ANY(SELECT LOWER(m) FROM unnest(muscles) m)
#         OR %s = ANY(SELECT LOWER(ms) FROM unnest(muscles_secondary) ms)
#         LIMIT 8;
#     """

#     muscle = parsed["muscle"].lower()
#     print("🟦 Running SQL for exercises with muscle =", muscle)

#     df = pd.read_sql(sql, conn, params=[muscle, muscle])
#     return df


# def search_nutrition(parsed):
#     if not parsed["food_name"]:
#         return pd.DataFrame()

#     sql = """
#         SELECT id, name, protein_g, energy_kcal, fat_g
#         FROM nutrition_fdw.foods
#         WHERE LOWER(name) LIKE '%%' || %s || '%%'
#         LIMIT 6;
#     """

#     df = pd.read_sql(sql, conn, params=[parsed["food_name"].lower()])
#     return df


# # ------------------------- LLM FALLBACK -----------------------------
# def call_llm(prompt):
#     try:
#         completion = client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return completion.choices[0].message.content.strip()
#     except Exception as e:
#         return f"LLM ERROR: {str(e)}"


# # ------------------------- MAIN -----------------------------
# user_query = input("\n💬 Enter your question: ").strip()
# print("\n📌 Your Query:", user_query)

# parsed = parse_query(user_query)
# print("\n🔍 Parsed Query:", parsed)

# exercise_results = search_exercises(parsed)
# nutrition_results = search_nutrition(parsed)

# print("\n🏋 Structured Exercise Results:")
# print(exercise_results if not exercise_results.empty else "❌ No direct exercise DB match")

# print("\n🍎 Structured Nutrition Results:")
# print(nutrition_results if not nutrition_results.empty else "❌ No direct nutrition DB match")

# need_llm = exercise_results.empty and nutrition_results.empty

# if need_llm:
#     print("\n🤖 No structured DB match — calling LLM...")
#     llm_answer = call_llm(
#         f"""
#         The user asked: {user_query}.
#         Provide real exercises and foods.
#         Format as plain bullet list, not JSON.
#         """
#     )
#     print("\n🧠 LLM RESPONSE:\n", llm_answer)
# else:
#     print("\n🎉 Answer directly from structured DB — No LLM needed!")














# =========================================BEST======================================
# # trainer_llm.py

import os
import json
import re
import warnings

import psycopg2
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# JSON HELPERS
# -------------------------------------------------------------
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def safe_json_parse(text: str, default: dict) -> dict:
    """Robust JSON parser for LLM output."""
    if not text:
        out = dict(default)
        out["raw_text"] = None
        return out

    raw = text
    t = raw.strip()

    # strip ``` fences if any
    if t.startswith("```"):
        newline_idx = t.find("\n")
        if newline_idx != -1:
            t = t[newline_idx + 1 :].strip()
    if t.endswith("```"):
        t = t[:-3].strip()

    # cut from first '{' to last '}'
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start : end + 1]
    else:
        candidate = t

    def try_parse(s: str):
        try:
            return json.loads(s)
        except Exception:
            s2 = re.sub(r",(\s*[\]}])", r"\1", s)
            try:
                return json.loads(s2)
            except Exception:
                return None

    parsed = try_parse(candidate)
    if parsed is not None:
        return parsed

    # fallback to raw
    try:
        return json.loads(raw)
    except Exception:
        out = dict(default)
        out["raw_text"] = raw
        return out


# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "dbname": os.getenv("DB_NAME", "trainer_dw"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS", "1234"),
    "port": int(os.getenv("DB_PORT", 5432)),
}

GROQ_MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

# FDW tables
EXERCISE_TABLE = os.getenv("EXERCISE_TABLE", "exercise_fdw.exercises")
NUTRITION_TABLE = os.getenv("NUTRITION_TABLE", "nutrition_fdw.foods")

# -------------------------------------------------------------
# DB CONNECTION
# -------------------------------------------------------------
try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ Connected to trainer_dw")
except Exception as e:
    print(f"❌ DB connection failed: {e}")
    raise


# -------------------------------------------------------------
# LLM CALL
# -------------------------------------------------------------
def call_llm(prompt: str, temperature: float = 0.0):
    """Generic LLM call, returns (text, usage_meta)."""
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        txt = completion.choices[0].message.content.strip()
        usage = getattr(completion, "usage", None)
        meta = {}
        if usage:
            meta = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        return txt, meta
    except Exception as e:
        print("❌ LLM ERROR:", e)
        return None, {"error": str(e)}


# -------------------------------------------------------------
# STEP 0 – READ USER QUERY
# -------------------------------------------------------------
user_query = input("\n💬 Enter your question: ").strip()
print("\n📌 Your Query:", user_query)


# -------------------------------------------------------------
# STEP 1 – PARSE / DECOMPOSE NATURAL LANGUAGE QUERY
# -------------------------------------------------------------
def parse_nl_query(q: str) -> dict:
    """
    Ask LLM to convert user query to a structured intent that maps to our DB.
    """
    prompt = f"""
You are a query parser for a fitness + nutrition system.

User query:
{q}

You MUST return ONLY JSON with EXACTLY these keys:

{{
  "muscle": null,                 // e.g. "chest", "legs", or null
  "ask_protein": false,           // true if user cares about protein grams
  "protein_filter": null,         // or {{ "op": ">", "value": 20.0 }} ; op in ["<", "<=", ">", ">=", "==", "="]
  "food_name": null,              // specific food name if mentioned
  "ask_unique_exercises": false,  // true if user wants unique exercise names
  "wants_plan": false,            // true if user wants multi-day workout or plan (not just raw rows)
  "gluten_free": false,           // true if they ask gluten-free
  "kcal_target": null             // numeric kcal target if mentioned (e.g. 200)
}}
"""
    default = {
        "muscle": None,
        "ask_protein": False,
        "protein_filter": None,
        "food_name": None,
        "ask_unique_exercises": False,
        "wants_plan": False,
        "gluten_free": False,
        "kcal_target": None,
    }

    txt, meta = call_llm(prompt, temperature=0.0)
    parsed = safe_json_parse(txt, default)

    # ensure all keys exist
    for k, v in default.items():
        if k not in parsed:
            parsed[k] = v

    return parsed


parsed = parse_nl_query(user_query)
print("🔍 Parsed Query:", parsed)


# -------------------------------------------------------------
# STEP 2 – STRUCTURED SEARCH: EXERCISES
# -------------------------------------------------------------
def search_exercises(parsed: dict):
    """
    Use structured exercise_fdw.exercises data.

    Primary search:
      - muscle word inside muscles[] or muscles_secondary[] (case-insensitive)

    Fallback search (if primary is empty):
      - category ILIKE '%muscle%'
      - OR name ILIKE '%muscle%'
    """
    muscle = parsed.get("muscle")
    if not muscle:
        return []

    muscle_l = str(muscle).lower()

    distinct_clause = "DISTINCT ON (LOWER(name))" if parsed.get("ask_unique_exercises") else ""

    # ---------- PRIMARY: muscles / muscles_secondary arrays ----------
    primary_sql = f"""
        SELECT {distinct_clause}
            id,
            name,
            muscles,
            muscles_secondary,
            equipment
        FROM {EXERCISE_TABLE}
        WHERE
            %s = ANY (SELECT LOWER(m) FROM unnest(muscles) AS m)
            OR %s = ANY (SELECT LOWER(ms) FROM unnest(muscles_secondary) AS ms)
        ORDER BY LOWER(name)
        LIMIT 20;
    """

    print("🟦 Running PRIMARY SQL for exercises with muscle =", muscle_l)
    try:
        df = pd.read_sql(primary_sql, conn, params=[muscle_l, muscle_l])
    except Exception as e:
        print("⚠ Exercise PRIMARY query failed:", e)
        df = pd.DataFrame()

    # ---------- FALLBACK: category/name text search ----------
    if df.empty:
        fallback_sql = f"""
            SELECT {distinct_clause}
                id,
                name,
                muscles,
                muscles_secondary,
                equipment
            FROM {EXERCISE_TABLE}
            WHERE
                LOWER(category) = %s
                OR LOWER(name) LIKE '%%' || %s || '%%'
            ORDER BY LOWER(name)
            LIMIT 20;
        """
        print("🟦 Primary returned 0 rows, running FALLBACK exercise SQL on category/name =", muscle_l)
        try:
            df = pd.read_sql(fallback_sql, conn, params=[muscle_l, muscle_l])
        except Exception as e:
            print("⚠ Exercise FALLBACK query failed:", e)
            df = pd.DataFrame()

    if df.empty:
        return []

    results = []
    for _, r in df.iterrows():
        results.append(
            {
                "id": int(r["id"]),
                "name": r["name"],
                "muscles": r["muscles"],
                "muscles_secondary": r["muscles_secondary"],
                "equipment": r["equipment"],
            }
        )
    return results


# -------------------------------------------------------------
# STEP 3 – STRUCTURED SEARCH: NUTRITION
# -------------------------------------------------------------
def search_nutrition(parsed: dict):
    ask_protein = bool(parsed.get("ask_protein"))
    protein_filter = parsed.get("protein_filter") or None
    food_name = parsed.get("food_name")
    fname_l = food_name.lower() if food_name else None

    base_select = f"""
        SELECT
            id,
            name,
            protein_g,
            energy_kcal,
            fat_g
        FROM {NUTRITION_TABLE}
    """

    # Case A: protein filter exists
    if ask_protein and protein_filter and isinstance(protein_filter, dict):
        op = protein_filter.get("op", ">=")
        val = protein_filter.get("value", None)

        if val is None:
            return []

        valid_ops = {"<", "<=", ">", ">=", "=", "=="}
        if op not in valid_ops:
            op = ">="
        if op == "==":
            op = "="

        where = "WHERE protein_g IS NOT NULL AND protein_g " + op + " %s"
        params = [float(val)]

        if fname_l:
            where += " AND LOWER(name) LIKE %s"
            params.append(f"%{fname_l}%")

        if op in ("<", "<="):
            order = "ORDER BY protein_g ASC NULLS LAST"
        else:
            order = "ORDER BY protein_g DESC NULLS LAST"

        sql = f"""
            {base_select}
            {where}
            {order}
            LIMIT 20;
        """
        print("🟩 Running SQL for nutrition with protein filter ", op, val)
        try:
            df = pd.read_sql(sql, conn, params=params)
        except Exception as e:
            print("⚠ Nutrition query failed:", e)
            return []

    # Case B: only food_name (no protein filter)
    elif fname_l:
        sql = f"""
            {base_select}
            WHERE LOWER(name) LIKE %s
            ORDER BY name
            LIMIT 20;
        """
        print("🟩 Running SQL for nutrition with name LIKE", fname_l)
        try:
            df = pd.read_sql(sql, conn, params=[f"%{fname_l}%"])
        except Exception as e:
            print("⚠ Nutrition query failed:", e)
            return []
    else:
        return []

    if df.empty:
        return []

    results = []
    for _, r in df.iterrows():
        results.append(
            {
                "id": int(r["id"]),
                "name": r["name"],
                "protein_g": float(r["protein_g"]) if r["protein_g"] is not None else None,
                "energy_kcal": float(r["energy_kcal"]) if r["energy_kcal"] is not None else None,
                "fat_g": float(r["fat_g"]) if r["fat_g"] is not None else None,
            }
        )
    return results


# -------------------------------------------------------------
# RUN STRUCTURED SEARCHES
# -------------------------------------------------------------
exercise_results = search_exercises(parsed)
nutrition_results = search_nutrition(parsed)

print("\n🏋 Structured Exercise Results:")
if exercise_results:
    print(pd.DataFrame(exercise_results)[["id", "name", "muscles", "muscles_secondary", "equipment"]])
else:
    print("❌ None")

print("\n🍎 Structured Nutrition Results:")
if nutrition_results:
    print(pd.DataFrame(nutrition_results)[["id", "name", "protein_g", "energy_kcal", "fat_g"]])
else:
    print("❌ None")

has_ex_struct = len(exercise_results) > 0
has_nut_struct = len(nutrition_results) > 0

# -------------------------------------------------------------
# DECIDE WHETHER WE NEED LLM
# -------------------------------------------------------------
wants_plan = bool(parsed.get("wants_plan"))

llm_needed = wants_plan or (not has_ex_struct and not has_nut_struct)

llm_data = None
fact_id = None
cur = conn.cursor()

if llm_needed:
    DEFAULT_LLM_OUTPUT = {
        "reason": "",
        "exercise_plan": [],
        "diet_plan": [],
        "summary": "",
        "score": 0.8,
        "confidence": 0.8,
    }

    gap_prompt = f"""
You are a fitness + nutrition expert.

USER QUERY:
{user_query}

PARSED INTENT (JSON):
{json.dumps(parsed, indent=2)}

STRUCTURED EXERCISE ROWS FROM DB (use them, do NOT invent new DB facts):
{json.dumps(exercise_results, indent=2)}

STRUCTURED NUTRITION ROWS FROM DB (use them, do NOT invent new DB facts):
{json.dumps(nutrition_results, indent=2)}

If there is no structured data, you may synthesize a reasonable plan.

Return STRICT JSON only:

{{
  "reason": "",
  "exercise_plan": [
    {{
      "day": 1,
      "exercise_name": "",
      "sets": 3,
      "reps": 10,
      "notes": ""
    }}
  ],
  "diet_plan": [
    {{
      "name": "",
      "description": "",
      "macros": {{
        "protein_g": 0.0,
        "fat_g": 0.0,
        "energy_kcal": 0.0
      }},
      "gluten_free": true,
      "serving_size": ""
    }}
  ],
  "summary": "",
  "score": 0.8,
  "confidence": 0.8
}}
"""
    print("\n🤖 Calling LLM to synthesize plan...")
    llm_raw, llm_usage = call_llm(gap_prompt, temperature=0.2)
    print("\n🧠 LLM RESPONSE (first 400 chars):\n", (llm_raw or "")[:400])

    llm_json = safe_json_parse(llm_raw, DEFAULT_LLM_OUTPUT)
    for k, v in DEFAULT_LLM_OUTPUT.items():
        if k not in llm_json:
            llm_json[k] = v

    reason = llm_json.get("reason", "")
    exercise_plan = llm_json.get("exercise_plan", []) or []
    diet_plan = llm_json.get("diet_plan", []) or []
    summary = llm_json.get("summary", "")
    score = float(llm_json.get("score", 0.8) or 0.8)
    confidence = float(llm_json.get("confidence", 0.8) or 0.8)

    llm_data = {
        "reason": reason,
        "exercise_plan": exercise_plan,
        "diet_plan": diet_plan,
        "summary": summary,
        "score": score,
        "confidence": confidence,
    }

    # store in materialized table
    insert_sql = f"""
        INSERT INTO {MATERIALIZED_TABLE}
            (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
        RETURNING fact_id;
    """

    exercise_json = exercise_results[0] if exercise_results else None
    nutrition_json = nutrition_results[0] if nutrition_results else None

    try:
        cur.execute(
            insert_sql,
            (
                None,
                "exercise_food_recommendation",
                json.dumps(exercise_json) if exercise_json else None,
                json.dumps(nutrition_json) if nutrition_json else None,
                json.dumps(to_jsonable(llm_data)),
                score,
                confidence,
            ),
        )
        fact_id = cur.fetchone()[0]
        conn.commit()
        print(f"\n💾 LLM inserted fact_id: {fact_id}")
    except Exception as e:
        conn.rollback()
        print("❌ Failed to insert into materialized_recommendations:", e)

# -------------------------------------------------------------
# FINAL JSON OUTPUT
# -------------------------------------------------------------
result_payload = {
    "user_query": user_query,
    "parsed": parsed,
    "structured": {
        "exercises": exercise_results,
        "nutrition": nutrition_results,
    },
    "llm": llm_data,
    "materialized_fact_id": fact_id,
}

print("\n======= FINAL JSON RESULT =======")
print(json.dumps(to_jsonable(result_payload), indent=2))
print("[RESULT]", json.dumps(to_jsonable(result_payload)))

cur.close()
conn.close()
print("\n🔚 Connection closed.")




# trainer_llm.py

# import os
# import json
# import re
# import warnings

# import psycopg2
# import pandas as pd
# from dotenv import load_dotenv
# from groq import Groq

# warnings.filterwarnings("ignore")

# # -------------------------------------------------------------
# # JSON HELPERS
# # -------------------------------------------------------------
# def to_jsonable(obj):
#     if isinstance(obj, dict):
#         return {k: to_jsonable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [to_jsonable(x) for x in obj]
#     if isinstance(obj, (str, int, float, bool)) or obj is None:
#         return obj
#     return str(obj)


# def safe_json_parse(text: str, default: dict) -> dict:
#     """Robust JSON parser for LLM output."""
#     if not text:
#         out = dict(default)
#         out["raw_text"] = None
#         return out

#     raw = text
#     t = raw.strip()

#     # strip ``` fences if any
#     if t.startswith("```"):
#         newline_idx = t.find("\n")
#         if newline_idx != -1:
#             t = t[newline_idx + 1 :].strip()
#     if t.endswith("```"):
#         t = t[:-3].strip()

#     # cut from first '{' to last '}'
#     start = t.find("{")
#     end = t.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         candidate = t[start : end + 1]
#     else:
#         candidate = t

#     def try_parse(s: str):
#         try:
#             return json.loads(s)
#         except Exception:
#             s2 = re.sub(r",(\s*[\]}])", r"\1", s)
#             try:
#                 return json.loads(s2)
#             except Exception:
#                 return None

#     parsed = try_parse(candidate)
#     if parsed is not None:
#         return parsed

#     # fallback to raw
#     try:
#         return json.loads(raw)
#     except Exception:
#         out = dict(default)
#         out["raw_text"] = raw
#         return out


# # -------------------------------------------------------------
# # CONFIG
# # -------------------------------------------------------------
# load_dotenv()

# DB_CONFIG = {
#     "host": os.getenv("DB_HOST", "localhost"),
#     "dbname": os.getenv("DB_NAME", "trainer_dw"),
#     "user": os.getenv("DB_USER", "postgres"),
#     "password": os.getenv("DB_PASS", "1234"),
#     "port": int(os.getenv("DB_PORT", 5432)),
# }

# GROQ_MODEL = "llama-3.1-8b-instant"
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

# # physical tables (can be overridden via env)
# EXERCISE_TABLE = os.getenv("EXERCISE_TABLE", "exercise_fdw.exercises")
# NUTRITION_TABLE = os.getenv("NUTRITION_TABLE", "nutrition_fdw.nutrition")

# # -------------------------------------------------------------
# # SIMPLE GLOBAL → LOCAL SCHEMA MAPPING
# # (lightweight schema matching layer)
# # -------------------------------------------------------------
# SCHEMA_MAP = {
#     "exercise": {
#         "table": EXERCISE_TABLE,
#         "text_attrs": {
#             "name": "name",
#             "category": "category",
#             "muscle": "muscles",            # conceptual "muscle" → muscles[]
#             "muscle_secondary": "muscles_secondary",
#             "equipment": "equipment",
#         },
#     },
#     "nutrition": {
#         "table": NUTRITION_TABLE,
#         "numeric_attrs": {
#             "protein": "protein_g",
#             "calories": "energy_kcal",
#             "kcal": "energy_kcal",
#             "energy": "energy_kcal",
#             "fat": "fat_g",
#             "carb": "carbohydrates_g",
#             "carbs": "carbohydrates_g",
#         },
#         "text_attrs": {
#             "name": "name",
#         },
#     },
# }

# # -------------------------------------------------------------
# # DB CONNECTION
# # -------------------------------------------------------------
# try:
#     conn = psycopg2.connect(**DB_CONFIG)
#     print("✅ Connected to trainer_dw")
# except Exception as e:
#     print(f"❌ DB connection failed: {e}")
#     raise


# # -------------------------------------------------------------
# # LLM CALL
# # -------------------------------------------------------------
# def call_llm(prompt: str, temperature: float = 0.0):
#     """Generic LLM call, returns (text, usage_meta)."""
#     try:
#         completion = client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=temperature,
#         )
#         txt = completion.choices[0].message.content.strip()
#         usage = getattr(completion, "usage", None)
#         meta = {}
#         if usage:
#             meta = {
#                 "prompt_tokens": getattr(usage, "prompt_tokens", None),
#                 "completion_tokens": getattr(usage, "completion_tokens", None),
#                 "total_tokens": getattr(usage, "total_tokens", None),
#             }
#         return txt, meta
#     except Exception as e:
#         print("❌ LLM ERROR:", e)
#         return None, {"error": str(e)}


# # -------------------------------------------------------------
# # STEP 0 – READ USER QUERY
# # -------------------------------------------------------------
# user_query = input("\n💬 Enter your question: ").strip()
# print("\n📌 Your Query:", user_query)


# # -------------------------------------------------------------
# # STEP 1 – PARSE / DECOMPOSE NATURAL LANGUAGE QUERY
# # -------------------------------------------------------------
# def parse_nl_query(q: str) -> dict:
#     """
#     Ask LLM to convert user query to a structured intent that maps to our DB.
#     This is the "global conceptual schema" level. We then map to columns.
#     """
#     prompt = f"""
# You are a query parser for a fitness + nutrition system.

# User query:
# {q}

# You MUST return ONLY JSON with EXACTLY these keys:

# {{
#   "muscle": null,                 // e.g. "chest", "legs", "glutes", or null
#   "ask_protein": false,           // true if user cares about protein grams
#   "protein_filter": null,         // or {{ "op": "<", "value": 20.0 }} ; op in ["<", "<=", ">", ">=", "=", "=="]
#   "food_name": null,              // specific food name if mentioned (e.g. "beef madras")
#   "ask_unique_exercises": false,  // true if user wants unique exercise names (no duplicates)
#   "wants_plan": false,            // true if user wants a multi-day workout/meal plan (not just raw DB rows)
#   "gluten_free": false,           // true if they explicitly ask for gluten-free
#   "kcal_target": null             // numeric kcal limit if mentioned (e.g. "under 200 kcal" => 200)
# }}
# """
#     default = {
#         "muscle": None,
#         "ask_protein": False,
#         "protein_filter": None,
#         "food_name": None,
#         "ask_unique_exercises": False,
#         "wants_plan": False,
#         "gluten_free": False,
#         "kcal_target": None,
#     }

#     txt, meta = call_llm(prompt, temperature=0.0)
#     parsed = safe_json_parse(txt, default)

#     # ensure all keys exist
#     for k, v in default.items():
#         if k not in parsed:
#             parsed[k] = v

#     # normalize some types
#     if parsed.get("protein_filter") is not None and not isinstance(parsed["protein_filter"], dict):
#         parsed["protein_filter"] = None

#     # kcal_target should be a number; if they give an object, try to extract "value"
#     kcal = parsed.get("kcal_target")
#     if isinstance(kcal, dict):
#         val = kcal.get("value")
#         parsed["kcal_target"] = float(val) if val is not None else None
#     elif isinstance(kcal, (int, float)):
#         parsed["kcal_target"] = float(kcal)
#     else:
#         parsed["kcal_target"] = None

#     return parsed


# parsed = parse_nl_query(user_query)
# print("🔍 Parsed Query:", parsed)


# # -------------------------------------------------------------
# # STEP 2 – STRUCTURED SEARCH: EXERCISES
# # -------------------------------------------------------------
# def search_exercises(parsed: dict):
#     """
#     Use the schema mapping and the parsed intent to build SQL for exercises.
#     Currently we focus on "muscle" because that's the main structured dimension.
#     We ALSO search category/name/array-text so "back" doesn't fail if it's only
#     in category or substring.
#     """
#     muscle = parsed.get("muscle")
#     if not muscle:
#         return []

#     muscle_l = str(muscle).lower()

#     # DISTINCT ON if they asked for unique exercises
#     distinct_clause = "DISTINCT ON (LOWER(name))" if parsed.get("ask_unique_exercises") else ""

#     sql = f"""
#         SELECT {distinct_clause}
#             id,
#             name,
#             category,
#             muscles,
#             muscles_secondary,
#             equipment
#         FROM {SCHEMA_MAP["exercise"]["table"]}
#         WHERE
#             LOWER(category) LIKE %s
#             OR LOWER(name) LIKE %s
#             OR LOWER(muscles::text) LIKE %s
#             OR LOWER(muscles_secondary::text) LIKE %s
#         ORDER BY LOWER(name)
#         LIMIT 50;
#     """

#     like = f"%{muscle_l}%"
#     params = [like, like, like, like]

#     print("🟦 Running PRIMARY SQL for exercises with muscle =", muscle)
#     try:
#         df = pd.read_sql(sql, conn, params=params)
#     except Exception as e:
#         print("⚠ Exercise query failed:", e)
#         return []

#     if df.empty:
#         return []

#     results = []
#     for _, r in df.iterrows():
#         results.append(
#             {
#                 "id": int(r["id"]),
#                 "name": r["name"],
#                 "category": r.get("category"),
#                 "muscles": r.get("muscles"),
#                 "muscles_secondary": r.get("muscles_secondary"),
#                 "equipment": r.get("equipment"),
#             }
#         )
#     return results


# # -------------------------------------------------------------
# # STEP 3 – STRUCTURED SEARCH: NUTRITION
# # -------------------------------------------------------------
# def build_numeric_filter_sql(column: str, op: str, param_name: str):
#     """
#     Build a safe SQL snippet for a numeric filter.
#     op is already whitelisted.
#     """
#     if op == "==":
#         op = "="
#     return f" {column} IS NOT NULL AND {column} {op} %({param_name})s "


# def search_nutrition(parsed: dict):
#     """
#     Use schema mapping + parsed intent to query nutrition.
#     Supports:
#       - protein_filter (op, value) => protein_g
#       - kcal_target (numeric limit) => energy_kcal < kcal_target
#       - optional name filter if food_name present
#     """
#     ask_protein = bool(parsed.get("ask_protein"))
#     protein_filter = parsed.get("protein_filter") or None
#     food_name = parsed.get("food_name")
#     kcal_target = parsed.get("kcal_target")

#     numeric_map = SCHEMA_MAP["nutrition"]["numeric_attrs"]
#     protein_col = numeric_map["protein"]
#     kcal_col = numeric_map["kcal"]
#     name_col = SCHEMA_MAP["nutrition"]["text_attrs"]["name"]

#     where_clauses = []
#     params = {}

#     # protein filter
#     if ask_protein and protein_filter and isinstance(protein_filter, dict):
#         op = protein_filter.get("op", ">=")
#         val = protein_filter.get("value", None)

#         if val is not None:
#             valid_ops = {"<", "<=", ">", ">=", "=", "=="}
#             if op not in valid_ops:
#                 op = ">="
#             where_clauses.append(build_numeric_filter_sql(protein_col, op, "pval"))
#             params["pval"] = float(val)

#     # kcal filter
#     if isinstance(kcal_target, (int, float)):
#         # by default interpret "under 150 kcal" as energy_kcal < 150
#         op_k = "<"
#         where_clauses.append(build_numeric_filter_sql(kcal_col, op_k, "kval"))
#         params["kval"] = float(kcal_target)

#     # optional name filter
#     if food_name:
#         where_clauses.append(f" LOWER({name_col}) LIKE %(fname)s ")
#         params["fname"] = f"%{food_name.lower()}%"

#     if not where_clauses:
#         # nothing to constrain on => return empty (no generic scan)
#         return []

#     base_select = f"""
#         SELECT
#             id,
#             {name_col} AS name,
#             {protein_col} AS protein_g,
#             {kcal_col} AS energy_kcal,
#             {numeric_map.get("fat", "fat_g")} AS fat_g
#         FROM {SCHEMA_MAP["nutrition"]["table"]}
#     """

#     where_sql = " AND ".join(where_clauses)
#     sql = f"""
#         {base_select}
#         WHERE {where_sql}
#         ORDER BY {kcal_col} ASC NULLS LAST, {protein_col} ASC NULLS LAST
#         LIMIT 50;
#     """

#     print("🟩 Running SQL for nutrition with parsed filters")
#     try:
#         df = pd.read_sql(sql, conn, params=params)
#     except Exception as e:
#         print("⚠ Nutrition query failed:", e)
#         return []

#     if df.empty:
#         return []

#     results = []
#     for _, r in df.iterrows():
#         results.append(
#             {
#                 "id": int(r["id"]),
#                 "name": r["name"],
#                 "protein_g": float(r["protein_g"]) if r["protein_g"] is not None else None,
#                 "energy_kcal": float(r["energy_kcal"]) if r["energy_kcal"] is not None else None,
#                 "fat_g": float(r["fat_g"]) if r["fat_g"] is not None else None,
#             }
#         )
#     return results


# # -------------------------------------------------------------
# # RUN STRUCTURED SEARCHES
# # -------------------------------------------------------------
# exercise_results = search_exercises(parsed)
# nutrition_results = search_nutrition(parsed)

# print("\n🏋 Structured Exercise Results:")
# if exercise_results:
#     print(
#         pd.DataFrame(exercise_results)[
#             ["id", "name", "category", "muscles", "muscles_secondary", "equipment"]
#         ]
#     )
# else:
#     print("❌ None")

# print("\n🍎 Structured Nutrition Results:")
# if nutrition_results:
#     print(
#         pd.DataFrame(nutrition_results)[
#             ["id", "name", "protein_g", "energy_kcal", "fat_g"]
#         ]
#     )
# else:
#     print("❌ None")

# has_ex_struct = len(exercise_results) > 0
# has_nut_struct = len(nutrition_results) > 0

# # -------------------------------------------------------------
# # DECIDE WHETHER WE NEED LLM
# # -------------------------------------------------------------
# wants_plan = bool(parsed.get("wants_plan"))

# # policy:
# #   - If user wants a "plan" → use LLM with structured rows as context
# #   - If they just want "names/rows" → DB-only if possible
# llm_needed = wants_plan or (not has_ex_struct and not has_nut_struct)

# llm_data = None
# fact_id = None

# cur = conn.cursor()

# if llm_needed:
#     DEFAULT_LLM_OUTPUT = {
#         "reason": "",
#         "exercise_plan": [],
#         "diet_plan": [],
#         "summary": "",
#         "score": 0.8,
#         "confidence": 0.8,
#     }

#     gap_prompt = f"""
# You are a fitness + nutrition expert.

# USER QUERY:
# {user_query}

# PARSED INTENT (JSON):
# {json.dumps(parsed, indent=2)}

# STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth; do NOT invent new DB facts):
# {json.dumps(exercise_results, indent=2)}

# STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth; do NOT invent new DB facts):
# {json.dumps(nutrition_results, indent=2)}

# If there is no structured data for some part, you may synthesize a reasonable plan.

# Return STRICT JSON only:

# {{
#   "reason": "",
#   "exercise_plan": [
#     {{
#       "day": 1,
#       "exercise_name": "",
#       "sets": 3,
#       "reps": 10,
#       "notes": ""
#     }}
#   ],
#   "diet_plan": [
#     {{
#       "name": "",
#       "description": "",
#       "macros": {{
#         "protein_g": 0.0,
#         "fat_g": 0.0,
#         "energy_kcal": 0.0
#       }},
#       "gluten_free": true,
#       "serving_size": ""
#     }}
#   ],
#   "summary": "",
#   "score": 0.8,
#   "confidence": 0.8
# }}
# """
#     print("\n🤖 Calling LLM to synthesize plan...")
#     llm_raw, llm_usage = call_llm(gap_prompt, temperature=0.2)
#     print("\n🧠 LLM RESPONSE (first 400 chars):\n", (llm_raw or "")[:400])

#     llm_json = safe_json_parse(llm_raw, DEFAULT_LLM_OUTPUT)
#     for k, v in DEFAULT_LLM_OUTPUT.items():
#         if k not in llm_json:
#             llm_json[k] = v

#     reason = llm_json.get("reason", "")
#     exercise_plan = llm_json.get("exercise_plan", []) or []
#     diet_plan = llm_json.get("diet_plan", []) or []
#     summary = llm_json.get("summary", "")
#     score = float(llm_json.get("score", 0.8) or 0.8)
#     confidence = float(llm_json.get("confidence", 0.8) or 0.8)

#     llm_data = {
#         "reason": reason,
#         "exercise_plan": exercise_plan,
#         "diet_plan": diet_plan,
#         "summary": summary,
#         "score": score,
#         "confidence": confidence,
#     }
#     llm_data = to_jsonable(llm_data)

#     # Store in materialized DW
#     insert_sql = f"""
#         INSERT INTO {MATERIALIZED_TABLE}
#             (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
#         VALUES
#             (%s, %s, %s, %s, %s, %s, %s)
#         RETURNING fact_id;
#     """

#     exercise_json = exercise_results[0] if exercise_results else None
#     nutrition_json = nutrition_results[0] if nutrition_results else None

#     try:
#         cur.execute(
#             insert_sql,
#             (
#                 None,
#                 "exercise_food_recommendation",
#                 json.dumps(exercise_json) if exercise_json else None,
#                 json.dumps(nutrition_json) if nutrition_json else None,
#                 json.dumps(
#                     {
#                         "user_query": user_query,
#                         "parsed": parsed,
#                         "reason": reason,
#                         "exercise_plan": exercise_plan,
#                         "diet_plan": diet_plan,
#                         "summary": summary,
#                         "score": score,
#                         "confidence": confidence,
#                         "structured_exercises_used": exercise_results[:5],
#                         "structured_nutrition_used": nutrition_results[:5],
#                         "model": GROQ_MODEL,
#                         "prompt": gap_prompt,
#                         "usage": llm_usage,
#                     }
#                 ),
#                 score,
#                 confidence,
#             ),
#         )
#         fact_id = cur.fetchone()[0]
#         conn.commit()
#         print(f"\n💾 LLM inserted fact_id: {fact_id}")
#     except Exception as e:
#         conn.rollback()
#         print("❌ Failed to insert into materialized_recommendations:", e)

# # -------------------------------------------------------------
# # FINAL JSON OUTPUT
# # -------------------------------------------------------------
# result_payload = {
#     "user_query": user_query,
#     "parsed": parsed,
#     "structured": {
#         "exercises": exercise_results,
#         "nutrition": nutrition_results,
#     },
#     "llm": llm_data,
#     "materialized_fact_id": fact_id,
# }

# print("\n======= FINAL JSON RESULT =======")
# print(json.dumps(to_jsonable(result_payload), indent=2))

# print("[RESULT]", json.dumps(to_jsonable(result_payload)))

# cur.close()
# conn.close()
# print("\n🔚 Connection closed.")
