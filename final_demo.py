
# import os
# import re
# import json
# import psycopg2
# import pandas as pd
# from dataclasses import dataclass, asdict
# from typing import List, Optional, Dict, Any
# from dotenv import load_dotenv
# from groq import Groq
# import difflib

# # ============================================================
# # ENV + GLOBALS
# # ============================================================

# load_dotenv(override=True)

# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = int(os.getenv("DB_PORT", "5432"))
# DB_NAME = os.getenv("DB_NAME", "trainer_dw")
# DB_USER = os.getenv("DB_USER", "postgres")
# DB_PASS = os.getenv("DB_PASS", "1234")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = "llama-3.1-8b-instant"  # valid Groq model

# MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

# groq_client: Optional[Groq] = None
# if GROQ_API_KEY:
#     groq_client = Groq(api_key=GROQ_API_KEY)
# else:
#     print("[WARN] GROQ_API_KEY not set; LLM calls will be skipped.")

# # ============================================================
# # JSON HELPERS
# # ============================================================

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
#     If parsing fails, returns a copy of `default` and attaches raw_text.
#     """
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

# # ============================================================
# # PARSED QUERY DATACLASS
# # ============================================================

# @dataclass
# class ParsedQuery:
#     raw_text: str

#     muscle: Optional[str] = None
#     ask_protein: bool = False
#     protein_filter: Optional[Dict[str, Any]] = None
#     food_name: Optional[str] = None

#     ask_unique_exercises: bool = False
#     wants_plan: bool = False
#     gluten_free: bool = False
#     kcal_filter: Optional[Dict[str, Any]] = None
#     kcal_target: Optional[float] = None
#     fat_filter: Optional[Dict[str, Any]] = None

#     n_exercises: Optional[int] = None
#     n_days: Optional[int] = None

#     attr_filters: List[Dict[str, Any]] = None
#     needs_exercise: bool = False
#     needs_nutrition: bool = False

#     unsatisfied_concepts: List[str] = None

#     def to_public_dict(self) -> Dict[str, Any]:
#         d = asdict(self)
#         d["unmatched"] = d.pop("unsatisfied_concepts")
#         return d

# # ============================================================
# # SIMPLE UTILITIES
# # ============================================================

# STOPWORDS = {
#     "the", "a", "an", "for", "and", "with", "from", "your", "my", "me", "each",
#     "that", "has", "at", "least", "any", "give", "get", "please", "of", "in",
#     "on", "to", "is", "are", "be", "it", "this", "those", "these", "some"
# }

# MUSCLE_KEYWORDS = {
#     "arms": "Arms",
#     "arm": "Arms",
#     "back": "Back",
#     "chest": "Chest",
#     "legs": "Legs",
#     "leg": "Legs",
#     "shoulder": "Shoulders",
#     "shoulders": "Shoulders",
#     "abs": "Abs",
#     "core": "Abs",
#     "full-body": "Full Body",
#     "fullbody": "Full Body",
#     "full": "Full Body",
#     "glutes": "Legs",  # glutes exercises live mostly in legs category in your DB
# }

# def tokenize(text: str) -> List[str]:
#     return re.findall(r"[a-zA-Z0-9%]+", text.lower())

# def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
#     print(title)
#     print("-" * len(title))
#     if df is None or df.empty:
#         print("❌ None\n")
#         return
#     display_df = df.head(max_rows)
#     print(display_df.to_string(index=False))
#     print()

# def normalize_text(s: str) -> str:
#     return re.sub(r"\s+", " ", s.strip().lower())

# # ============================================================
# # QUERY PARSING
# # ============================================================

# def parse_user_query(q: str) -> ParsedQuery:
#     text = q.strip()
#     tokens = tokenize(text)

#     parsed = ParsedQuery(
#         raw_text=text,
#         attr_filters=[],
#         unsatisfied_concepts=[]
#     )

#     exercise_markers = {"exercise", "exercises", "workout", "routine", "plan"}
#     nutrition_markers = {
#         "food", "foods", "snack", "snacks", "diet", "protein",
#         "calories", "kcal", "carb", "carbs", "fat", "fats", "meal", "meals"
#     }

#     if any(t in exercise_markers or t in MUSCLE_KEYWORDS for t in tokens):
#         parsed.needs_exercise = True
#     if any(t in nutrition_markers for t in tokens):
#         parsed.needs_nutrition = True

#     if not parsed.needs_exercise and not parsed.needs_nutrition:
#         if "diet" in tokens or "meal" in tokens or "plan" in tokens:
#             parsed.needs_nutrition = True

#     m_days = re.search(r"(\d+)\s*day", text.lower())
#     if m_days:
#         parsed.n_days = int(m_days.group(1))
#         parsed.wants_plan = True

#     if "different" in tokens or "unique" in tokens or "variety" in tokens:
#         parsed.ask_unique_exercises = True

#     muscle_found = None
#     for t in tokens:
#         if t in MUSCLE_KEYWORDS:
#             muscle_found = MUSCLE_KEYWORDS[t]
#     parsed.muscle = muscle_found

#     protein_pattern = re.search(
#         r"(?:more than|over|greater than|at least|>=|>)\s*(\d+)\s*g\s*protein",
#         text.lower()
#     )
#     if protein_pattern:
#         val = float(protein_pattern.group(1))
#         parsed.ask_protein = True
#         parsed.protein_filter = {"op": ">", "value": val}
#         parsed.attr_filters.append({
#             "db": "nutrition",
#             "table": "nutrition_fdw.foods",
#             "column": "protein_g",
#             "op": ">",
#             "value": val,
#             "unit": "g",
#             "source_phrase": "protein"
#         })

#     if "low" in tokens and ("carb" in tokens or "carbs" in tokens):
#         parsed.needs_nutrition = True
#         parsed.unsatisfied_concepts.append("low-carb snack (no direct carb column)")

#     if "protein" in tokens and not parsed.protein_filter:
#         parsed.ask_protein = True

#     recognized_tokens = set()
#     for t in tokens:
#         if t.isdigit():
#             recognized_tokens.add(t)
#     recognized_tokens |= exercise_markers
#     recognized_tokens |= nutrition_markers
#     recognized_tokens |= set(MUSCLE_KEYWORDS.keys())
#     recognized_tokens |= {
#         "day", "days", "beginner", "advanced", "intermediate",
#         "full", "body", "routine", "using", "use",
#         "after", "before", "match", "matches", "matchup", "targets"
#     }

#     for t in tokens:
#         if t in STOPWORDS:
#             continue
#         if t in recognized_tokens:
#             continue
#         parsed.unsatisfied_concepts.append(t)

#     return parsed

# # ============================================================
# # DB CONNECTION + SQL HELPERS
# # ============================================================

# def get_connection():
#     conn = psycopg2.connect(
#         host=DB_HOST,
#         port=DB_PORT,
#         dbname=DB_NAME,
#         user=DB_USER,
#         password=DB_PASS,
#     )
#     return conn

# def run_exercise_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
#     base = """
#         SELECT
#             id,
#             name,
#             category,
#             muscles,
#             muscles_secondary,
#             equipment
#         FROM exercise_fdw.exercises
#     """

#     params: Dict[str, Any] = {}
#     where_clauses = []

#     if parsed.muscle:
#         where_clauses.append(
#             "(LOWER(category) = %(muscle_category)s OR %(muscle_like)s = ANY(muscles))"
#         )
#         params["muscle_category"] = parsed.muscle.lower()
#         params["muscle_like"] = parsed.muscle.lower()

#     sql = base
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " LIMIT 50"

#     print("→ Running exercise SQL")
#     return pd.read_sql(sql, conn, params=params)

# def run_nutrition_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
#     base = """
#         SELECT
#             id,
#             name,
#             protein_g,
#             energy_kcal,
#             fat_g
#         FROM nutrition_fdw.foods
#     """

#     params: Dict[str, Any] = {}
#     where_clauses = []

#     if parsed.protein_filter:
#         op = parsed.protein_filter["op"]
#         val = parsed.protein_filter["value"]
#         if op == ">":
#             where_clauses.append("protein_g > %(p_val)s")
#         elif op == ">=":
#             where_clauses.append("protein_g >= %(p_val)s")
#         elif op == "=":
#             where_clauses.append("protein_g = %(p_val)s")
#         params["p_val"] = val

#     sql = base
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " LIMIT 50"

#     print("→ Running nutrition SQL")
#     return pd.read_sql(sql, conn, params=params)

# # ============================================================
# # GAP DETECTION
# # ============================================================

# def detect_gap_tokens(parsed: ParsedQuery) -> List[str]:
#     return parsed.unsatisfied_concepts or []

# # ============================================================
# # LLM CALL
# # ============================================================

# def build_llm_prompt(user_query: str,
#                      parsed: ParsedQuery,
#                      df_ex: pd.DataFrame,
#                      df_nu: pd.DataFrame,
#                      gap_tokens: List[str]) -> str:

#     parsed_json = json.dumps(parsed.to_public_dict(), indent=2)
#     ex_json = df_ex.head(20).to_dict(orient="records") if df_ex is not None else []
#     nu_json = df_nu.head(20).to_dict(orient="records") if df_nu is not None else []

#     prompt = f"""
# You are a fitness + nutrition expert that extends a virtual table combining exercises and foods.

# USER QUERY:
# {user_query}

# PARSED INTENT (JSON):
# {parsed_json}

# STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(ex_json, indent=2)}

# STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(nu_json, indent=2)}

# UNMATCHED / GAP TOKENS (concepts not directly mapped to DB columns):
# {gap_tokens}

# TASK:
# 1. Interpret the user's goal.
# 2. Use ONLY the structured rows above as factual database items.
# 3. Use your general knowledge to fill the GAP concepts and build a helpful plan.

# Return STRICT JSON with this shape:

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
#       "gluten_free": null,
#       "serving_size": ""
#     }}
#   ],
#   "summary": "",
#   "score": 0.8,
#   "confidence": 0.8
# }}

# If the query is completely outside the scope of the databases (e.g. smartphones),
# you may ignore the DB rows and just answer the question with the same JSON shape,
# using your general knowledge.
# """
#     return prompt

# def call_llm_with_groq(user_query: str,
#                        parsed: ParsedQuery,
#                        df_ex: pd.DataFrame,
#                        df_nu: pd.DataFrame,
#                        gap_tokens: List[str]) -> Optional[Dict[str, Any]]:
#     if groq_client is None:
#         print("[WARN] GROQ client not initialised. Skipping LLM call.")
#         return None

#     prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

#     try:
#         completion = groq_client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are a precise, honest fitness and nutrition assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=1200,
#         )
#         content = completion.choices[0].message.content.strip()
#         print("\n🧠 Raw LLM response (first 400 chars):")
#         print(content[:400])

#         default_llm = {
#             "reason": "",
#             "exercise_plan": [],
#             "diet_plan": [],
#             "summary": "",
#             "score": 0.8,
#             "confidence": 0.8,
#         }
#         llm_json = safe_json_parse(content, default_llm)
#         for k, v in default_llm.items():
#             if k not in llm_json:
#                 llm_json[k] = v
#         return llm_json

#     except Exception as e:
#         print(f"[ERROR] LLM call failed: {e}")
#         return None

# # ============================================================
# # MATERIALIZATION (STORE IN DW)
# # ============================================================

# def materialize_llm_recommendation(conn,
#                                    user_query: str,
#                                    parsed: ParsedQuery,
#                                    df_ex: pd.DataFrame,
#                                    df_nu: pd.DataFrame,
#                                    llm_json: Dict[str, Any],
#                                    entity_type: str = "exercise_recommendation") -> Optional[int]:
#     """
#     Insert LLM result + structured rows into trainer_dw.materialized_recommendations.
#     """
#     if llm_json is None:
#         return None

#     score = 0.8
#     confidence = 0.8
#     if isinstance(llm_json, dict):
#         score = float(llm_json.get("score", 0.8) or 0.8)
#         confidence = float(llm_json.get("confidence", 0.8) or 0.8)

#     # Build payload to store in llm_data
#     llm_payload = {
#         "user_query": user_query,
#         "parsed": parsed.to_public_dict(),
#         "structured": {
#             "exercises": df_ex.to_dict(orient="records") if df_ex is not None and not df_ex.empty else [],
#             "nutrition": df_nu.to_dict(orient="records") if df_nu is not None and not df_nu.empty else [],
#         },
#         "llm_output": llm_json,
#     }

#     exercise_data = llm_payload["structured"]["exercises"]
#     nutrition_data = llm_payload["structured"]["nutrition"]

#     insert_sql = f"""
#         INSERT INTO {MATERIALIZED_TABLE}
#             (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
#         VALUES
#             (%s, %s, %s, %s, %s, %s, %s)
#         RETURNING fact_id;
#     """

#     cur = conn.cursor()
#     try:
#         cur.execute(
#             insert_sql,
#             (
#                 None,
#                 entity_type,
#                 json.dumps(to_jsonable(exercise_data)) if exercise_data else None,
#                 json.dumps(to_jsonable(nutrition_data)) if nutrition_data else None,
#                 json.dumps(to_jsonable(llm_payload)),
#                 score,
#                 confidence,
#             ),
#         )
#         fact_id = cur.fetchone()[0]
#         conn.commit()
#         print(f"💾 Stored LLM recommendation in materialized table with fact_id={fact_id}")
#         return fact_id
#     except Exception as e:
#         conn.rollback()
#         print("❌ Failed to insert into materialized_recommendations:", e)
#         return None
#     finally:
#         cur.close()

# # ============================================================
# # CACHE LOOKUP WITH FUZZY MATCH
# # ============================================================

# def fetch_cached_recommendation(conn,
#                                 user_query: str,
#                                 threshold: float = 0.85) -> Optional[Dict[str, Any]]:
#     """
#     Look in materialized_recommendations for a previous query whose
#     llm_data->>'user_query' is ~similar to current user_query.
#     If similarity >= threshold, return that record.
#     """
#     norm_q = normalize_text(user_query)

#     sql = f"""
#         SELECT fact_id, exercise_data, nutrition_data, llm_data
#         FROM {MATERIALIZED_TABLE}
#         WHERE llm_data IS NOT NULL
#         ORDER BY generation_ts DESC
#         LIMIT 200;
#     """

#     cur = conn.cursor()
#     try:
#         cur.execute(sql)
#         rows = cur.fetchall()
#     except Exception as e:
#         print("⚠ Failed to read from materialized_recommendations:", e)
#         cur.close()
#         return None

#     best = None
#     best_score = 0.0

#     for fact_id, ex_data, nu_data, llm_data in rows:
#         if isinstance(llm_data, str):
#             try:
#                 llm_obj = json.loads(llm_data)
#             except Exception:
#                 llm_obj = {}
#         else:
#             llm_obj = llm_data or {}

#         prev_q = llm_obj.get("user_query")
#         if not prev_q:
#             continue

#         sim = difflib.SequenceMatcher(None, norm_q, normalize_text(prev_q)).ratio()
#         if sim > best_score:
#             best_score = sim
#             best = (fact_id, ex_data, nu_data, llm_obj)

#     cur.close()

#     if best and best_score >= threshold:
#         fact_id, ex_data, nu_data, llm_obj = best
#         print(f"📦 Cache hit in materialized_recommendations (fact_id={fact_id}, similarity={best_score:.2f}).")
#         return {
#             "fact_id": fact_id,
#             "similarity": best_score,
#             "exercise_data": ex_data,
#             "nutrition_data": nu_data,
#             "llm_data": llm_obj,
#         }

#     return None

# # ============================================================
# # MAIN MEDIATOR LOOP
# # ============================================================

# def main():
#     print("✅ Connected to trainer_dw")
#     conn = get_connection()

#     try:
#         while True:
#             try:
#                 user_query = input("Enter your question: ").strip()
#             except EOFError:
#                 break

#             if not user_query:
#                 continue
#             if user_query.lower() in {"quit", "exit"}:
#                 break

#             print(f"\n📌 Your Query: {user_query}")

#             # 1) Parse query
#             parsed = parse_user_query(user_query)
#             print("🔍 Parsed Query:", json.dumps(parsed.to_public_dict(), indent=2))

#             # 2) Structured DB lookup
#             df_ex = pd.DataFrame()
#             df_nu = pd.DataFrame()

#             if parsed.needs_exercise:
#                 df_ex = run_exercise_sql(conn, parsed)
#             if parsed.needs_nutrition:
#                 df_nu = run_nutrition_sql(conn, parsed)

#             pretty_print_df("🏋 Structured Exercise Results:", df_ex)
#             pretty_print_df("🍎 Structured Nutrition Results:", df_nu)

#             # 3) Decide if we need LLM (for gaps or out-of-domain)
#             gap_tokens = detect_gap_tokens(parsed)
#             llm_result = None
#             source = "db_only"
#             materialized_fact_id = None
#             cache_similarity = None

#             need_llm = bool(gap_tokens) or (not parsed.needs_exercise and not parsed.needs_nutrition)

#             if need_llm:
#                 print(f"⚠ Gap detected, calling LLM for: {gap_tokens}" if gap_tokens else
#                       "⚠ Out-of-domain query, delegating to LLM.")

#                 # 3a) Try cache first (fuzzy match)
#                 cached = fetch_cached_recommendation(conn, user_query, threshold=0.85)
#                 if cached is not None:
#                     llm_result = cached["llm_data"]
#                     source = "cached"
#                     cache_similarity = cached["similarity"]
#                     materialized_fact_id = cached["fact_id"]
#                 else:
#                     # 3b) Actually call LLM
#                     llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
#                     if llm_result is not None:
#                         source = "hybrid" if (parsed.needs_exercise or parsed.needs_nutrition) else "llm_only"
#                         materialized_fact_id = materialize_llm_recommendation(
#                             conn, user_query, parsed, df_ex, df_nu, llm_result, entity_type="exercise_recommendation"
#                         )
#             else:
#                 source = "db_only"

#             # 4) Final JSON result
#             result = {
#                 "user_query": user_query,
#                 "parsed": parsed.to_public_dict(),
#                 "source": source,
#                 "structured": {
#                     "exercises": df_ex.to_dict(orient="records") if df_ex is not None else [],
#                     "nutrition": df_nu.to_dict(orient="records") if df_nu is not None else [],
#                 },
#                 "llm": llm_result,
#                 "materialized_fact_id": materialized_fact_id,
#                 "cache_similarity": cache_similarity,
#             }

#             print("\n======= FINAL JSON RESULT =======")
#             print(json.dumps(result, indent=4, ensure_ascii=False))
#             print("\n")

#     finally:
#         conn.close()
#         print("🔚 Connection closed.")


# if __name__ == "__main__":
#     main()



















import os
import re
import json
import psycopg2
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from groq import Groq
import difflib

# ============================================================
# ENV + GLOBALS
# ============================================================

load_dotenv(override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "trainer_dw")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "1234")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # valid Groq model

MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("[WARN] GROQ_API_KEY not set; LLM calls will be skipped.")

# ============================================================
# JSON HELPERS
# ============================================================

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def safe_json_parse(text: str, default: dict) -> dict:
    """
    Robust JSON parser for LLM output.
    If parsing fails, returns a copy of `default` and attaches raw_text.
    """
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

# ============================================================
# PARSED QUERY DATACLASS
# ============================================================

@dataclass
class ParsedQuery:
    raw_text: str

    muscle: Optional[str] = None
    ask_protein: bool = False
    protein_filter: Optional[Dict[str, Any]] = None
    food_name: Optional[str] = None

    ask_unique_exercises: bool = False
    wants_plan: bool = False
    gluten_free: bool = False
    kcal_filter: Optional[Dict[str, Any]] = None
    kcal_target: Optional[float] = None
    fat_filter: Optional[Dict[str, Any]] = None

    n_exercises: Optional[int] = None
    n_days: Optional[int] = None

    attr_filters: List[Dict[str, Any]] = None
    needs_exercise: bool = False
    needs_nutrition: bool = False

    unsatisfied_concepts: List[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["unmatched"] = d.pop("unsatisfied_concepts")
        return d

# ============================================================
# SIMPLE UTILITIES
# ============================================================

STOPWORDS = {
    "the", "a", "an", "for", "and", "with", "from", "your", "my", "me", "each",
    "that", "has", "at", "least", "any", "give", "get", "please", "of", "in",
    "on", "to", "is", "are", "be", "it", "this", "those", "these", "some"
}

MUSCLE_KEYWORDS = {
    "arms": "Arms",
    "arm": "Arms",
    "back": "Back",
    "chest": "Chest",
    "legs": "Legs",
    "leg": "Legs",
    "shoulder": "Shoulders",
    "shoulders": "Shoulders",
    "abs": "Abs",
    "core": "Abs",
    "full-body": "Full Body",
    "fullbody": "Full Body",
    "full": "Full Body",
    "glutes": "Legs",  # glutes exercises live mostly in legs category in your DB
}

def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9%]+", text.lower())

def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(title)
    print("-" * len(title))
    if df is None or df.empty:
        print("❌ None\n")
        return
    display_df = df.head(max_rows)
    print(display_df.to_string(index=False))
    print()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# ============================================================
# QUERY PARSING
# ============================================================

def parse_user_query(q: str) -> ParsedQuery:
    text = q.strip()
    tokens = tokenize(text)

    parsed = ParsedQuery(
        raw_text=text,
        attr_filters=[],
        unsatisfied_concepts=[]
    )

    exercise_markers = {"exercise", "exercises", "workout", "routine", "plan"}
    nutrition_markers = {
        "food", "foods", "snack", "snacks", "diet", "protein",
        "calories", "kcal", "carb", "carbs", "fat", "fats", "meal", "meals"
    }

    if any(t in exercise_markers or t in MUSCLE_KEYWORDS for t in tokens):
        parsed.needs_exercise = True
    if any(t in nutrition_markers for t in tokens):
        parsed.needs_nutrition = True

    if not parsed.needs_exercise and not parsed.needs_nutrition:
        if "diet" in tokens or "meal" in tokens or "plan" in tokens:
            parsed.needs_nutrition = True

    m_days = re.search(r"(\d+)\s*day", text.lower())
    if m_days:
        parsed.n_days = int(m_days.group(1))
        parsed.wants_plan = True

    if "different" in tokens or "unique" in tokens or "variety" in tokens:
        parsed.ask_unique_exercises = True

    muscle_found = None
    for t in tokens:
        if t in MUSCLE_KEYWORDS:
            muscle_found = MUSCLE_KEYWORDS[t]
    parsed.muscle = muscle_found

    protein_pattern = re.search(
        r"(?:more than|over|greater than|at least|>=|>)\s*(\d+)\s*g\s*protein",
        text.lower()
    )
    if protein_pattern:
        val = float(protein_pattern.group(1))
        parsed.ask_protein = True
        parsed.protein_filter = {"op": ">", "value": val}
        parsed.attr_filters.append({
            "db": "nutrition",
            "table": "nutrition_fdw.foods",
            "column": "protein_g",
            "op": ">",
            "value": val,
            "unit": "g",
            "source_phrase": "protein"
        })

    if "low" in tokens and ("carb" in tokens or "carbs" in tokens):
        parsed.needs_nutrition = True
        parsed.unsatisfied_concepts.append("low-carb snack (no direct carb column)")

    if "protein" in tokens and not parsed.protein_filter:
        parsed.ask_protein = True

    recognized_tokens = set()
    for t in tokens:
        if t.isdigit():
            recognized_tokens.add(t)
    recognized_tokens |= exercise_markers
    recognized_tokens |= nutrition_markers
    recognized_tokens |= set(MUSCLE_KEYWORDS.keys())
    recognized_tokens |= {
        "day", "days", "beginner", "advanced", "intermediate",
        "full", "body", "routine", "using", "use",
        "after", "before", "match", "matches", "matchup", "targets"
    }

    for t in tokens:
        if t in STOPWORDS:
            continue
        if t in recognized_tokens:
            continue
        parsed.unsatisfied_concepts.append(t)

    return parsed

# ============================================================
# DB CONNECTION + SQL HELPERS
# ============================================================

def get_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    return conn

def run_exercise_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
    base = """
        SELECT
            id,
            name,
            category,
            muscles,
            muscles_secondary,
            equipment
        FROM exercise_fdw.exercises
    """

    params: Dict[str, Any] = {}
    where_clauses = []

    if parsed.muscle:
        where_clauses.append(
            "(LOWER(category) = %(muscle_category)s OR %(muscle_like)s = ANY(muscles))"
        )
        params["muscle_category"] = parsed.muscle.lower()
        params["muscle_like"] = parsed.muscle.lower()

    sql = base
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " LIMIT 50"

    print("→ Running exercise SQL")
    return pd.read_sql(sql, conn, params=params)

def run_nutrition_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
    base = """
        SELECT
            id,
            name,
            protein_g,
            energy_kcal,
            fat_g
        FROM nutrition_fdw.foods
    """

    params: Dict[str, Any] = {}
    where_clauses = []

    if parsed.protein_filter:
        op = parsed.protein_filter["op"]
        val = parsed.protein_filter["value"]
        if op == ">":
            where_clauses.append("protein_g > %(p_val)s")
        elif op == ">=":
            where_clauses.append("protein_g >= %(p_val)s")
        elif op == "=":
            where_clauses.append("protein_g = %(p_val)s")
        params["p_val"] = val

    sql = base
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " LIMIT 50"

    print("→ Running nutrition SQL")
    return pd.read_sql(sql, conn, params=params)

# ============================================================
# GAP DETECTION
# ============================================================

def detect_gap_tokens(parsed: ParsedQuery) -> List[str]:
    return parsed.unsatisfied_concepts or []

# ============================================================
# LLM CALL – PROMPT BUILDERS
# ============================================================

def build_llm_prompt(user_query: str,
                     parsed: ParsedQuery,
                     df_ex: pd.DataFrame,
                     df_nu: pd.DataFrame,
                     gap_tokens: List[str]) -> str:
    """
    Hybrid / in-domain prompt: fitness + nutrition, uses DB rows.
    """
    parsed_json = json.dumps(parsed.to_public_dict(), indent=2)
    ex_json = df_ex.head(20).to_dict(orient="records") if df_ex is not None else []
    nu_json = df_nu.head(20).to_dict(orient="records") if df_nu is not None else []

    prompt = f"""
You are a fitness + nutrition expert that extends a virtual table combining exercises and foods.

USER QUERY:
{user_query}

PARSED INTENT (JSON):
{parsed_json}

STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
{json.dumps(ex_json, indent=2)}

STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
{json.dumps(nu_json, indent=2)}

UNMATCHED / GAP TOKENS (concepts not directly mapped to DB columns):
{gap_tokens}

TASK:
1. Interpret the user's goal.
2. Use ONLY the structured rows above as factual database items.
3. Use your general knowledge to fill the GAP concepts and build a helpful plan.

Return STRICT JSON with this shape:

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
      "gluten_free": null,
      "serving_size": ""
    }}
  ],
  "summary": "",
  "score": 0.8,
  "confidence": 0.8
}}

If the query is completely outside the scope of the databases (e.g. smartphones),
you may ignore the DB rows and just answer the question with the same JSON shape,
using your general knowledge.
"""
    return prompt


def build_out_of_domain_prompt(user_query: str) -> str:
    """
    Pure out-of-domain prompt: general knowledge answer,
    no attempt to map to fitness/nutrition or DB.
    """
    prompt = f"""
You are a knowledgeable, precise general AI assistant.

The following question is completely outside our structured fitness/nutrition databases,
so you must answer it purely from your general world knowledge (as if you searched the web),
without trying to turn it into a workout or diet question.

USER QUERY:
{user_query}

REQUIREMENTS:
- Answer the question directly and stay on-topic.
- If the user asks for N items (like "5 bikes", "5 shoes under 10000 rupees"),
  give a short list of reasonable items.
- Do NOT talk about workouts, exercises, training, or nutrition unless the user explicitly asks for that.
- Return STRICT JSON ONLY in this shape:

{{
  "reason": "",
  "summary": "",
  "items": [
    {{
      "name": "",
      "description": "",
      "price": null,
      "extra": ""
    }}
  ],
  "exercise_plan": [],
  "diet_plan": [],
  "score": 0.8,
  "confidence": 0.8
}}

Notes:
- Put the main factual answer inside the 'items' list (e.g., the 5 bike names).
- 'price' can be a number or a short string like "₹85,000" / "$1,500", or null if not applicable.
- 'extra' can store any additional info (e.g., brand, category, region, etc.).
- 'exercise_plan' and 'diet_plan' should normally be empty lists here.
"""
    return prompt

# def call_llm_with_groq(user_query: str,
#                        parsed: ParsedQuery,
#                        df_ex: pd.DataFrame,
#                        df_nu: pd.DataFrame,
#                        gap_tokens: List[str]) -> Optional[Dict[str, Any]]:
#     if groq_client is None:
#         print("[WARN] GROQ client not initialised. Skipping LLM call.")
#         return None

#     # Decide if this is an out-of-domain query:
#     # no exercise intent, no nutrition intent
#     is_out_of_domain = (not parsed.needs_exercise and not parsed.needs_nutrition)

#     if is_out_of_domain:
#         prompt = build_out_of_domain_prompt(user_query)
#     else:
#         prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

#     try:
#         completion = groq_client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a precise, honest assistant. "
#                         "Follow the user's instructions and the JSON schema exactly."
#                     ),
#                 },
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=1200,
#         )
#         content = completion.choices[0].message.content.strip()
#         print("\n🧠 Full Raw LLM response:")
#         print(content)
#         print("\n----- END OF RAW RESPONSE -----\n")


#         default_llm = {
#             "reason": "",
#             "exercise_plan": [],
#             "diet_plan": [],
#             "summary": "",
#             "score": 0.8,
#             "confidence": 0.8,
#         }
#         llm_json = safe_json_parse(content, default_llm)
#         for k, v in default_llm.items():
#             if k not in llm_json:
#                 llm_json[k] = v
#         return llm_json

#     except Exception as e:
#         print(f"[ERROR] LLM call failed: {e}")
#         return None




# ---------------------------- LLM CALL ----------------------------

def call_llm_with_groq(user_query: str,
                       parsed: ParsedQuery,
                       df_ex: pd.DataFrame,
                       df_nu: pd.DataFrame,
                       gap_tokens: List[str]) -> Optional[Dict[str, Any]]:
    if groq_client is None:
        print("[WARN] GROQ client not initialised. Skipping LLM call.")
        return None

    # Determine if out-of-domain (pure LLM)
    is_out_of_domain = (not parsed.needs_exercise and not parsed.needs_nutrition)

    if is_out_of_domain:
        prompt = build_out_of_domain_prompt(user_query)
    else:
        prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise, honest assistant. "
                        "Follow the user's instructions and the JSON schema exactly."
                    ),
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2200,
        )

        content = completion.choices[0].message.content.strip()
        
        # ---------------- FULL PRINT WITHOUT TRUNCATION ----------------
        print("\n🧠 FULL RAW LLM RESPONSE:")
        print(content)
        print("\n----- END OF RAW RESPONSE -----\n")
        # ----------------------------------------------------------------

        default_llm = {
            "reason": "",
            "exercise_plan": [],
            "diet_plan": [],
            "summary": "",
            "score": 0.8,
            "confidence": 0.8,
        }
        llm_json = safe_json_parse(content, default_llm)
        for k, v in default_llm.items():
            if k not in llm_json:
                llm_json[k] = v
        return llm_json

    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
        return None


# ============================================================
# MATERIALIZATION (STORE IN DW)
# ============================================================

def materialize_llm_recommendation(conn,
                                   user_query: str,
                                   parsed: ParsedQuery,
                                   df_ex: pd.DataFrame,
                                   df_nu: pd.DataFrame,
                                   llm_json: Dict[str, Any],
                                   entity_type: str = "exercise_recommendation") -> Optional[int]:
    """
    Insert LLM result + structured rows into trainer_dw.materialized_recommendations.
    """
    if llm_json is None:
        return None

    score = 0.8
    confidence = 0.8
    if isinstance(llm_json, dict):
        score = float(llm_json.get("score", 0.8) or 0.8)
        confidence = float(llm_json.get("confidence", 0.8) or 0.8)

    # Build payload to store in llm_data
    llm_payload = {
        "user_query": user_query,
        "parsed": parsed.to_public_dict(),
        "structured": {
            "exercises": df_ex.to_dict(orient="records") if df_ex is not None and not df_ex.empty else [],
            "nutrition": df_nu.to_dict(orient="records") if df_nu is not None and not df_nu.empty else [],
        },
        "llm_output": llm_json,
    }

    exercise_data = llm_payload["structured"]["exercises"]
    nutrition_data = llm_payload["structured"]["nutrition"]

    insert_sql = f"""
        INSERT INTO {MATERIALIZED_TABLE}
            (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s)
        RETURNING fact_id;
    """

    cur = conn.cursor()
    try:
        cur.execute(
            insert_sql,
            (
                None,
                entity_type,
                json.dumps(to_jsonable(exercise_data)) if exercise_data else None,
                json.dumps(to_jsonable(nutrition_data)) if nutrition_data else None,
                json.dumps(to_jsonable(llm_payload)),
                score,
                confidence,
            ),
        )
        fact_id = cur.fetchone()[0]
        conn.commit()
        print(f"💾 Stored LLM recommendation in materialized table with fact_id={fact_id}")
        return fact_id
    except Exception as e:
        conn.rollback()
        print("❌ Failed to insert into materialized_recommendations:", e)
        return None
    finally:
        cur.close()

# ============================================================
# CACHE LOOKUP WITH FUZZY MATCH
# ============================================================

def fetch_cached_recommendation(conn,
                                user_query: str,
                                threshold: float = 0.85) -> Optional[Dict[str, Any]]:
    """
    Look in materialized_recommendations for a previous query whose
    llm_data->>'user_query' is ~similar to current user_query.
    If similarity >= threshold, return that record.
    """
    norm_q = normalize_text(user_query)

    sql = f"""
        SELECT fact_id, exercise_data, nutrition_data, llm_data
        FROM {MATERIALIZED_TABLE}
        WHERE llm_data IS NOT NULL
        ORDER BY generation_ts DESC
        LIMIT 200;
    """

    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
    except Exception as e:
        print("⚠ Failed to read from materialized_recommendations:", e)
        cur.close()
        return None

    best = None
    best_score = 0.0

    for fact_id, ex_data, nu_data, llm_data in rows:
        if isinstance(llm_data, str):
            try:
                llm_obj = json.loads(llm_data)
            except Exception:
                llm_obj = {}
        else:
            llm_obj = llm_data or {}

        prev_q = llm_obj.get("user_query")
        if not prev_q:
            continue

        sim = difflib.SequenceMatcher(None, norm_q, normalize_text(prev_q)).ratio()
        if sim > best_score:
            best_score = sim
            best = (fact_id, ex_data, nu_data, llm_obj)

    cur.close()

    if best and best_score >= threshold:
        fact_id, ex_data, nu_data, llm_obj = best
        print(f"📦 Cache hit in materialized_recommendations (fact_id={fact_id}, similarity={best_score:.2f}).")
        return {
            "fact_id": fact_id,
            "similarity": best_score,
            "exercise_data": ex_data,
            "nutrition_data": nu_data,
            "llm_data": llm_obj,
        }

    return None

# ============================================================
# HUMAN-READABLE LLM OUTPUT
# ============================================================

def print_trainer_recommendation(llm_result: Optional[Dict[str, Any]]) -> None:
    """
    Print a readable paragraph-style summary of the LLM output to the terminal.
    Works for both hybrid (fitness) and out-of-domain queries.
    """
    print("\n📝 Trainer Recommendation (Readable Format)")
    print("-------------------------------------------")

    if not isinstance(llm_result, dict):
        print("No LLM recommendation available.")
        return

    reason = llm_result.get("reason")
    summary = llm_result.get("summary")
    exercise_plan = llm_result.get("exercise_plan") or []
    diet_plan = llm_result.get("diet_plan") or []
    items = llm_result.get("items") or []

    if reason:
        print(f"💡 Reasoning: {reason}\n")

    if summary:
        print(f"📌 Summary: {summary}\n")

    # If out-of-domain items are present, show them
    if items:
        print("📋 Results:")
        for idx, item in enumerate(items, 1):
            name = item.get("name", "Item")
            desc = item.get("description", "")
            price = item.get("price")
            extra = item.get("extra", "")
            line = f" • {idx}. {name}"
            if price not in (None, ""):
                line += f" – Price: {price}"
            print(line)
            if desc:
                print(f"    {desc}")
            if extra:
                print(f"    {extra}")
        print()

    # Fitness-style exercise plan (for hybrid queries)
    if exercise_plan:
        print("🏋 Recommended Exercises:")
        for ex in exercise_plan:
            day = ex.get("day")
            name = ex.get("exercise_name", "Exercise")
            sets = ex.get("sets")
            reps = ex.get("reps")
            notes = ex.get("notes")
            day_str = f"Day {day} – " if day is not None else ""
            sets_str = f"{sets}" if sets is not None else "N/A"
            reps_str = f"{reps}" if reps is not None else "N/A"
            print(f" • {day_str}{name} ({sets_str} sets × {reps_str} reps)", end="")
            if notes:
                print(f" – {notes}")
            else:
                print()
        print()

    # Nutrition recommendations (for hybrid queries)
    if diet_plan:
        print("🥗 Recommended Nutrition:")
        for food in diet_plan:
            fname = food.get("name", "Food item")
            desc = food.get("description", "")
            macros = food.get("macros") or {}
            protein = macros.get("protein_g")
            fat = macros.get("fat_g")
            kcal = macros.get("energy_kcal")
            serving = food.get("serving_size", "")
            print(f" • {fname}:", end="")
            if desc:
                print(f" {desc}", end="")
            print()
            if any(v is not None for v in [protein, fat, kcal]):
                print(f"    | Protein {protein}g, Fat {fat}g, {kcal} kcal", end="")
                if serving:
                    print(f" | Serving: {serving}")
                else:
                    print()
            elif serving:
                print(f"    | Serving: {serving}")
        print()

# ============================================================
# MAIN MEDIATOR LOOP
# ============================================================

def main():
    print("✅ Connected to trainer_dw")
    conn = get_connection()

    try:
        while True:
            try:
                user_query = input("Enter your question: ").strip()
            except EOFError:
                break

            if not user_query:
                continue
            if user_query.lower() in {"quit", "exit"}:
                break

            print(f"\n📌 Your Query: {user_query}")

            # 1) Parse query
            parsed = parse_user_query(user_query)
            print("🔍 Parsed Query:", json.dumps(parsed.to_public_dict(), indent=2))

            # 2) Structured DB lookup
            df_ex = pd.DataFrame()
            df_nu = pd.DataFrame()

            if parsed.needs_exercise:
                df_ex = run_exercise_sql(conn, parsed)
            if parsed.needs_nutrition:
                df_nu = run_nutrition_sql(conn, parsed)

            pretty_print_df("🏋 Structured Exercise Results:", df_ex)
            pretty_print_df("🍎 Structured Nutrition Results:", df_nu)

            # 3) Decide if we need LLM (for gaps or out-of-domain)
            gap_tokens = detect_gap_tokens(parsed)
            llm_result = None
            source = "db_only"
            materialized_fact_id = None
            cache_similarity = None

            need_llm = bool(gap_tokens) or (not parsed.needs_exercise and not parsed.needs_nutrition)

            if need_llm:
                print(
                    f"⚠ Gap detected, calling LLM for: {gap_tokens}"
                    if gap_tokens else
                    "⚠ Out-of-domain query, delegating to LLM."
                )

                # 3a) Try cache first (fuzzy match)
                cached = fetch_cached_recommendation(conn, user_query, threshold=0.85)
                if cached is not None:
                    llm_result = cached["llm_data"]
                    source = "cached"
                    cache_similarity = cached["similarity"]
                    materialized_fact_id = cached["fact_id"]
                else:
                    # 3b) Actually call LLM
                    llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
                    if llm_result is not None:
                        source = "hybrid" if (parsed.needs_exercise or parsed.needs_nutrition) else "llm_only"
                        materialized_fact_id = materialize_llm_recommendation(
                            conn, user_query, parsed, df_ex, df_nu, llm_result,
                            entity_type="exercise_recommendation"
                        )
            else:
                source = "db_only"

            # 4) Final JSON result
            result = {
                "user_query": user_query,
                "parsed": parsed.to_public_dict(),
                "source": source,
                "structured": {
                    "exercises": df_ex.to_dict(orient="records") if df_ex is not None else [],
                    "nutrition": df_nu.to_dict(orient="records") if df_nu is not None else [],
                },
                "llm": llm_result,
                "materialized_fact_id": materialized_fact_id,
                "cache_similarity": cache_similarity,
            }

            print("\n======= FINAL JSON RESULT =======")
            print(json.dumps(result, indent=4, ensure_ascii=False))

            # 5) Human-readable explanation
            if llm_result is not None:
                print_trainer_recommendation(llm_result)

            print("\n")

    finally:
        conn.close()
        print("🔚 Connection closed.")


if __name__ == "__main__":
    main()
