# import os
# import re
# import json
# import psycopg2
# import pandas as pd
# from dataclasses import dataclass, asdict
# from typing import List, Optional, Dict, Any

# from dotenv import load_dotenv
# from groq import Groq

# # ------------------------------------------------------------------------------------
# # ENV + GLOBALS
# # ------------------------------------------------------------------------------------

# # Load .env BEFORE reading any env vars
# load_dotenv(override=True)

# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = int(os.getenv("DB_PORT", "5432"))
# DB_NAME = os.getenv("DB_NAME", "trainer_dw")
# DB_USER = os.getenv("DB_USER", "postgres")
# DB_PASS = os.getenv("DB_PASS", "1234")

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# # Materialized table for storing LLM results
# MATERIALIZED_TABLE = "trainer_dw.materialized_recommendations"

# # Groq client (None if key missing)
# groq_client: Optional[Groq] = None
# if GROQ_API_KEY:
#     groq_client = Groq(api_key=GROQ_API_KEY)
# else:
#     print("[WARN] GROQ_API_KEY not set in environment. LLM gap-filling will be skipped.")

# # ------------------------------------------------------------------------------------
# # JSON HELPERS
# # ------------------------------------------------------------------------------------

# def to_jsonable(obj):
#     """
#     Make sure objects are JSON-serializable (used for llm_data before inserting into DB).
#     """
#     if isinstance(obj, dict):
#         return {k: to_jsonable(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [to_jsonable(x) for x in obj]
#     if isinstance(obj, (str, int, float, bool)) or obj is None:
#         return obj
#     return str(obj)

# # ------------------------------------------------------------------------------------
# # PARSED QUERY DATACLASS
# # ------------------------------------------------------------------------------------

# @dataclass
# class ParsedQuery:
#     raw_text: str

#     muscle: Optional[str] = None  # primary muscle/category
#     ask_protein: bool = False
#     protein_filter: Optional[Dict[str, Any]] = None  # {op, value}
#     food_name: Optional[str] = None

#     ask_unique_exercises: bool = False
#     wants_plan: bool = False
#     gluten_free: bool = False  # kept for compatibility, but NOT applied as SQL
#     kcal_filter: Optional[Dict[str, Any]] = None
#     kcal_target: Optional[float] = None
#     fat_filter: Optional[Dict[str, Any]] = None

#     n_exercises: Optional[int] = None
#     n_days: Optional[int] = None

#     # attr-level filters for SQL building
#     attr_filters: List[Dict[str, Any]] = None

#     # flags telling which DBs are needed
#     needs_exercise: bool = False
#     needs_nutrition: bool = False

#     # tokens that didn't map to any structured concept
#     unsatisfied_concepts: List[str] = None

#     def to_public_dict(self) -> Dict[str, Any]:
#         d = asdict(self)
#         # rename to match earlier logs
#         d["unmatched"] = d.pop("unsatisfied_concepts")
#         return d


# # ------------------------------------------------------------------------------------
# # SIMPLE UTILITIES
# # ------------------------------------------------------------------------------------

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
# }

# def tokenize(text: str) -> List[str]:
#     # simple word tokenizer
#     return re.findall(r"[a-zA-Z0-9%]+", text.lower())


# def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
#     print(title)
#     print("-" * len(title))
#     if df is None or df.empty:
#         print("❌ None\n")
#         return
#     if len(df) > max_rows:
#         display_df = df.head(max_rows)
#     else:
#         display_df = df
#     print(display_df.to_string(index=False))
#     print()


# # ------------------------------------------------------------------------------------
# # QUERY PARSING
# # ------------------------------------------------------------------------------------

# def parse_user_query(q: str) -> ParsedQuery:
#     text = q.strip()
#     tokens = tokenize(text)

#     parsed = ParsedQuery(
#         raw_text=text,
#         attr_filters=[],
#         unsatisfied_concepts=[]
#     )

#     # --- Decide if user needs exercise / nutrition ---
#     exercise_markers = {"exercise", "exercises", "workout", "routine", "plan"}
#     nutrition_markers = {
#         "food", "foods", "snack", "snacks", "diet", "protein",
#         "calories", "kcal", "carb", "carbs", "fat", "fats", "meal", "meals"
#     }

#     if any(t in exercise_markers or t in MUSCLE_KEYWORDS for t in tokens):
#         parsed.needs_exercise = True
#     if any(t in nutrition_markers for t in tokens):
#         parsed.needs_nutrition = True

#     # If neither detected, but user mentions 'diet' or 'meal' or 'plan', assume nutrition
#     if not parsed.needs_exercise and not parsed.needs_nutrition:
#         if "diet" in tokens or "meal" in tokens or "plan" in tokens:
#             parsed.needs_nutrition = True

#     # --- days / number of days ---
#     m_days = re.search(r"(\d+)\s*day", text.lower())
#     if m_days:
#         parsed.n_days = int(m_days.group(1))
#         parsed.wants_plan = True

#     # --- unique exercises requested? ---
#     if "different" in tokens or "unique" in tokens or "variety" in tokens:
#         parsed.ask_unique_exercises = True

#     # --- muscle / body-part detection (single main muscle) ---
#     muscle_found = None
#     for t in tokens:
#         if t in MUSCLE_KEYWORDS:
#             muscle_found = MUSCLE_KEYWORDS[t]
#     parsed.muscle = muscle_found

#     # --- Protein filter: e.g. "more than 15g protein", "at least 20g protein" ---
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

#     # --- low-carb detection (conceptual only) ---
#     if "low" in tokens and ("carb" in tokens or "carbs" in tokens):
#         parsed.needs_nutrition = True
#         parsed.unsatisfied_concepts.append("low-carb snack (no direct carb column)")

#     # --- Ask protein explicitly without numeric filter ---
#     if "protein" in tokens and not parsed.protein_filter:
#         parsed.ask_protein = True

#     # --- Compute unmatched concept tokens for GAP ---
#     recognized_tokens = set()

#     # numbers, days
#     for t in tokens:
#         if t.isdigit():
#             recognized_tokens.add(t)

#     # recognized markers
#     recognized_tokens |= exercise_markers
#     recognized_tokens |= nutrition_markers
#     recognized_tokens |= set(MUSCLE_KEYWORDS.keys())
#     recognized_tokens |= {
#         "day", "days", "beginner", "advanced", "intermediate",
#         "full", "body", "routine", "using", "use",
#         "after", "before", "match", "matches", "matchup"
#     }

#     for t in tokens:
#         if t in STOPWORDS:
#             continue
#         if t in recognized_tokens:
#             continue
#         # Everything else is a candidate unmatched concept
#         parsed.unsatisfied_concepts.append(t)

#     return parsed


# # ------------------------------------------------------------------------------------
# # DB CONNECTION + SQL HELPERS
# # ------------------------------------------------------------------------------------

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
#     """
#     Very simple exercise query:
#     - If muscle detected: filter by category or muscles containing that word
#     - Else: just return some generic exercises
#     """
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
#             "(LOWER(category) = %(muscle_category)s OR %(muscle_like)s = ANY(ARRAY[LOWER(category)]) OR %(muscle_like)s = ANY(muscles))"
#         )
#         params["muscle_category"] = parsed.muscle.lower()
#         params["muscle_like"] = parsed.muscle.lower()

#     sql = base
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " LIMIT 50"

#     print("→ Running exercise SQL")
#     # pandas warning about psycopg2 is fine
#     return pd.read_sql(sql, conn, params=params)


# def run_nutrition_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
#     """
#     Simple nutrition query:
#     - If protein filter exists: apply it
#     - Otherwise, just return some rows
#     """
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


# # ------------------------------------------------------------------------------------
# # GAP DETECTION + LLM CALL
# # ------------------------------------------------------------------------------------

# def detect_gap_tokens(parsed: ParsedQuery) -> List[str]:
#     # unsatisfied_concepts already built during parsing
#     # Plus conceptual flags like "low-carb snack (no direct carb column)"
#     return parsed.unsatisfied_concepts or []


# def build_llm_prompt(
#     user_query: str,
#     parsed: ParsedQuery,
#     df_ex: pd.DataFrame,
#     df_nu: pd.DataFrame,
#     gap_tokens: List[str]
# ) -> str:
#     """
#     Prompt that gives the LLM:
#     - original user query
#     - parsed intent (as JSON)
#     - structured DB rows
#     - unmatched / gap tokens
#     and asks it to synthesize a final fitness + diet plan.
#     """
#     parsed_json = json.dumps(parsed.to_public_dict(), indent=2)
#     ex_json = df_ex.head(20).to_dict(orient="records") if df_ex is not None else []
#     nu_json = df_nu.head(20).to_dict(orient="records") if df_nu is not None else []

#     prompt = f"""
# You are a fitness + nutrition expert that extends a virtual table combining exercise and foods.

# USER QUERY:
# {user_query}

# PARSED INTENT (JSON):
# {parsed_json}

# STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(ex_json, indent=2)}

# STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(nu_json, indent=2)}

# UNMATCHED / GAP TOKENS FROM THE QUERY (concepts not directly mapped to DB columns):
# {gap_tokens}

# TASK:
# 1. Interpret the user's goal.
# 2. Use ONLY the structured rows above as factual database items.
# 3. For parts that are not covered by the DB (the gap tokens), you may use your general knowledge to:
#    - build a beginner-friendly or goal-specific workout structure,
#    - choose which of the given DB rows are best for the request,
#    - decide ordering, sets, reps, per-day structure, etc.
# 4. Return a rich but concise combined plan in **JSON** with this shape:

# {{
#   "reason": "<short reasoning>",
#   "exercise_plan": [
#     {{
#       "day": 1,
#       "exercise_name": "<name from exercises DB>",
#       "sets": 3,
#       "reps": 10,
#       "notes": "<how to perform, level info, etc>"
#     }}
#   ],
#   "diet_plan": [
#     {{
#       "name": "<name from nutrition DB>",
#       "description": "<why this snack or food fits>",
#       "macros": {{
#         "protein_g": <float>,
#         "fat_g": <float>,
#         "energy_kcal": <float>
#       }},
#       "gluten_free": null,
#       "serving_size": ""
#     }}
#   ],
#   "summary": "<plain-language summary for the user>",
#   "score": 0.8,
#   "confidence": 0.8
# }}

# If the query is completely outside the scope of both databases (no useful DB rows), you MUST still:
# - explain that no DB rows matched,
# - but still give a reasonable plan using your general knowledge,
# - and clearly state that this part is purely LLM-generated, not from the DB.

# Important:
# - Use only exercises and foods that actually appear in the structured DB rows above.
# - It's OK if some fields like gluten_free or serving_size remain empty/approximate.
# """
#     return prompt


# def build_llm_prompt_out_of_scope(user_query: str) -> str:
#     """
#     Simpler prompt when the query is unrelated to exercise/nutrition.
#     Here we just behave like a normal assistant and answer the user's request.
#     """
#     prompt = f"""
# The following user question is NOT related to our exercise/nutrition databases.
# Act as a helpful general AI assistant and answer the user's question directly.

# USER QUERY:
# {user_query}
# """
#     return prompt


# def call_llm_with_groq(
#     user_query: str,
#     parsed: ParsedQuery,
#     df_ex: pd.DataFrame,
#     df_nu: pd.DataFrame,
#     gap_tokens: List[str]
# ) -> Optional[Dict[str, Any]]:
#     if groq_client is None:
#         print("[WARN] GROQ client not initialised (no API key). Skipping LLM call.")
#         return None

#     # Out-of-domain: no exercise/nutrition requested at all
#     out_of_scope = not parsed.needs_exercise and not parsed.needs_nutrition

#     if out_of_scope:
#         prompt = build_llm_prompt_out_of_scope(user_query)
#         system_msg = "You are a helpful general AI assistant."
#     else:
#         prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)
#         system_msg = "You are a precise, honest fitness and nutrition assistant."

#     try:
#         completion = groq_client.chat.completions.create(
#             model="llama-3.1-8b-instant",
#             messages=[
#                 {"role": "system", "content": system_msg},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=900
#         )
#         content = completion.choices[0].message.content.strip()
#         print("\n🧠 Raw LLM response (first 400 chars):")
#         print(content[:400])

#         # For out-of-domain, we just return the raw answer
#         if out_of_scope:
#             return {"raw_response": content}

#         # For in-domain, try to locate JSON in the response
#         json_text = None
#         if content.startswith("{"):
#             json_text = content
#         else:
#             m = re.search(r"\{.*\}", content, flags=re.S)
#             if m:
#                 json_text = m.group(0)
#         if not json_text:
#             print("[WARN] LLM response did not contain JSON. Returning raw text.")
#             return {"raw_response": content}

#         try:
#             llm_json = json.loads(json_text)
#         except Exception as e:
#             print(f"[WARN] Failed to parse LLM JSON: {e}. Returning raw response instead.")
#             return {"raw_response": content}

#         return llm_json

#     except Exception as e:
#         print(f"[ERROR] LLM call failed: {e}")
#         return None


# # ------------------------------------------------------------------------------------
# # MAIN MEDIATOR LOOP
# # ------------------------------------------------------------------------------------

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

#             # 1) Parse
#             parsed = parse_user_query(user_query)
#             print("🔍 Parsed Query:", json.dumps(parsed.to_public_dict(), indent=2))

#             df_ex = pd.DataFrame()
#             df_nu = pd.DataFrame()

#             # 2) Run SQL for exercises
#             if parsed.needs_exercise:
#                 df_ex = run_exercise_sql(conn, parsed)

#             # 3) Run SQL for nutrition
#             if parsed.needs_nutrition:
#                 df_nu = run_nutrition_sql(conn, parsed)

#             # 4) Pretty-print structured results
#             pretty_print_df("🏋 Structured Exercise Results:", df_ex)
#             pretty_print_df("🍎 Structured Nutrition Results:", df_nu)

#             # 5) GAP detection
#             gap_tokens = detect_gap_tokens(parsed)
#             llm_result = None
#             source = "db_only"

#             # Decide whether / how to call LLM
#             if gap_tokens:
#                 print(f"⚠ Gap detected, calling LLM for: {gap_tokens}")
#                 llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
#                 if llm_result is not None:
#                     # hybrid if we actually used DB
#                     if (parsed.needs_exercise and not df_ex.empty) or (parsed.needs_nutrition and not df_nu.empty):
#                         source = "hybrid"
#                     else:
#                         source = "llm_only"
#             else:
#                 # No gaps in mapping, but we might still want pure-LLM for out-of-domain
#                 if not parsed.needs_exercise and not parsed.needs_nutrition:
#                     print("⚠ Query seems out-of-scope for both databases; calling general LLM.")
#                     llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, [])
#                     if llm_result is not None:
#                         source = "llm_only"

#             # 6) MATERIALIZE LLM RESULT (ADD-ON)
#             materialized_fact_id = None
#             if llm_result is not None:
#                 try:
#                     # Take first exercise/nutrition row if available
#                     exercise_json = None
#                     nutrition_json = None
#                     if df_ex is not None and not df_ex.empty:
#                         exercise_json = df_ex.head(1).to_dict(orient="records")[0]
#                     if df_nu is not None and not df_nu.empty:
#                         nutrition_json = df_nu.head(1).to_dict(orient="records")[0]

#                     # Infer entity type
#                     if parsed.needs_exercise and parsed.needs_nutrition:
#                         entity_type = "exercise_food_recommendation"
#                     elif parsed.needs_exercise:
#                         entity_type = "exercise_recommendation"
#                     elif parsed.needs_nutrition:
#                         entity_type = "food_recommendation"
#                     else:
#                         entity_type = "out_of_domain_query"

#                     # Try to read score/confidence from LLM JSON if present
#                     score = 0.8
#                     confidence = 0.8
#                     if isinstance(llm_result, dict):
#                         if "score" in llm_result:
#                             try:
#                                 score = float(llm_result["score"])
#                             except Exception:
#                                 pass
#                         if "confidence" in llm_result:
#                             try:
#                                 confidence = float(llm_result["confidence"])
#                             except Exception:
#                                 pass
#                         # If only one present, mirror into the other
#                         if "score" in llm_result and "confidence" not in llm_result:
#                             confidence = score
#                         if "confidence" in llm_result and "score" not in llm_result:
#                             score = confidence

#                     insert_sql = f"""
#                         INSERT INTO {MATERIALIZED_TABLE}
#                             (canonical_id, entity_type, exercise_data, nutrition_data, llm_data, score, confidence)
#                         VALUES
#                             (%s, %s, %s, %s, %s, %s, %s)
#                         RETURNING fact_id;
#                     """

#                     with conn.cursor() as cur:
#                         cur.execute(
#                             insert_sql,
#                             (
#                                 None,
#                                 entity_type,
#                                 json.dumps(exercise_json) if exercise_json is not None else None,
#                                 json.dumps(nutrition_json) if nutrition_json is not None else None,
#                                 json.dumps(to_jsonable(llm_result)),
#                                 score,
#                                 confidence,
#                             ),
#                         )
#                         materialized_fact_id = cur.fetchone()[0]
#                         conn.commit()
#                         print(f"💾 Stored LLM recommendation in materialized table with fact_id={materialized_fact_id}")
#                 except Exception as e:
#                     conn.rollback()
#                     print(f"❌ Failed to insert into {MATERIALIZED_TABLE}: {e}")

#             # 7) Build final JSON result
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
#             }

#             print("\n======= FINAL JSON RESULT =======")
#             print(json.dumps(to_jsonable(result), indent=4, ensure_ascii=False))
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
# LLM CALL
# ============================================================

def build_llm_prompt(user_query: str,
                     parsed: ParsedQuery,
                     df_ex: pd.DataFrame,
                     df_nu: pd.DataFrame,
                     gap_tokens: List[str]) -> str:

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

def call_llm_with_groq(user_query: str,
                       parsed: ParsedQuery,
                       df_ex: pd.DataFrame,
                       df_nu: pd.DataFrame,
                       gap_tokens: List[str]) -> Optional[Dict[str, Any]]:
    if groq_client is None:
        print("[WARN] GROQ client not initialised. Skipping LLM call.")
        return None

    prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise, honest fitness and nutrition assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        content = completion.choices[0].message.content.strip()
        print("\n🧠 Raw LLM response (first 400 chars):")
        print(content[:400])

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
# MAIN MEDIATOR LOOP
# ============================================================





def format_llm_as_paragraph(llm_json: Dict[str,Any]) -> str:
    """Convert LLM JSON into a readable human-style paragraph."""
    if not llm_json:
        return "No additional insight from LLM."

    reason = llm_json.get("reason","")
    summary = llm_json.get("summary","")

    para = ""
    if reason:
        para += f"\n💡 Reasoning: {reason}\n"
    if summary:
        para += f"\n📌 Summary: {summary}\n"

    exercises = llm_json.get("exercise_plan",[])
    if exercises:
        para += "\n🏋 Recommended Exercises:\n"
        for ex in exercises:
            para += f" • Day {ex.get('day','')} – {ex.get('exercise_name','')} ({ex.get('sets','?')} sets × {ex.get('reps','?')} reps)"
            note = ex.get("notes")
            if note:
                para += f" – {note}"
            para += "\n"

    diet = llm_json.get("diet_plan",[])
    if diet:
        para += "\n🥗 Recommended Nutrition:\n"
        for food in diet:
            para += f" • {food.get('name','')}: {food.get('description','')}"
            macros = food.get("macros",{})
            para += f" | Protein {macros.get('protein_g',0)}g, Fat {macros.get('fat_g',0)}g, {macros.get('energy_kcal',0)} kcal"
            if food.get("gluten_free"):
                para += " (Gluten-Free)"
            serving = food.get("serving_size")
            if serving:
                para += f" | Serving: {serving}"
            para += "\n"

    return para.strip()


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

            parsed = parse_user_query(user_query)
            print("🔍 Parsed Query:", json.dumps(parsed.to_public_dict(), indent=2))

            df_ex = pd.DataFrame()
            df_nu = pd.DataFrame()

            if parsed.needs_exercise:
                df_ex = run_exercise_sql(conn, parsed)
            if parsed.needs_nutrition:
                df_nu = run_nutrition_sql(conn, parsed)

            pretty_print_df("🏋 Structured Exercise Results:", df_ex)
            pretty_print_df("🍎 Structured Nutrition Results:", df_nu)

            gap_tokens = detect_gap_tokens(parsed)
            llm_result = None
            source = "db_only"
            materialized_fact_id = None
            cache_similarity = None

            need_llm = bool(gap_tokens) or (not parsed.needs_exercise and not parsed.needs_nutrition)

            if need_llm:
                print(f"⚠ Gap detected, calling LLM for: {gap_tokens}" if gap_tokens else
                      "⚠ Out-of-domain query, delegating to LLM.")

                cached = fetch_cached_recommendation(conn, user_query, threshold=0.85)
                if cached is not None:
                    llm_result = cached["llm_data"]
                    source = "cached"
                    cache_similarity = cached["similarity"]
                    materialized_fact_id = cached["fact_id"]
                else:
                    llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
                    if llm_result is not None:
                        source = "hybrid" if (parsed.needs_exercise or parsed.needs_nutrition) else "llm_only"
                        materialized_fact_id = materialize_llm_recommendation(
                            conn, user_query, parsed, df_ex, df_nu, llm_result
                        )

            else:
                source = "db_only"

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

            ### >>> ADDED FOR HUMAN READABLE OUTPUT <<<
            if llm_result:
                print("\n📝 Trainer Recommendation (Readable Format)")
                print("-------------------------------------------")
                print(format_llm_as_paragraph(llm_result))

            print("\n")

    finally:
        conn.close()
        print("🔚 Connection closed.")


if __name__ == "__main__":
    main()









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

