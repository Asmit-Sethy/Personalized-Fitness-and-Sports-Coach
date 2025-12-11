# # trainer_llm.py
# # ---------------------------------------------------------
# # SWAN-like trainer with:
# #  - LLM query decomposition
# #  - Schema matching via dictionary + inverted index + embeddings
# #  - Structured querying of exercise + nutrition DW
# #  - Hybrid gap-filling with LLM
# #  - Materialized view logging
# # ---------------------------------------------------------

# import os
# import json
# import re
# import warnings
# from collections import defaultdict
# from difflib import SequenceMatcher

# import psycopg2
# import pandas as pd
# import numpy as np
# from dotenv import load_dotenv
# from groq import Groq
# from sentence_transformers import SentenceTransformer

# warnings.filterwarnings("ignore")

# # =========================================================
# # JSON HELPERS
# # =========================================================

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


# # =========================================================
# # CONFIG
# # =========================================================

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

# # IMPORTANT: set your FDW / base tables here or via env
# EXERCISE_TABLE = os.getenv("EXERCISE_TABLE", "exercise_fdw.exercises")
# NUTRITION_TABLE = os.getenv("NUTRITION_TABLE", "nutrition_fdw.foods")

# # =========================================================
# # DB CONNECTION
# # =========================================================

# try:
#     conn = psycopg2.connect(**DB_CONFIG)
#     print("✅ Connected to trainer_dw")
# except Exception as e:
#     print(f"❌ DB connection failed: {e}")
#     raise

# cur = conn.cursor()

# # =========================================================
# # LLM CALL
# # =========================================================

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


# # =========================================================
# # SCHEMA DICTIONARY + MATCHER (with inverted index + embeddings)
# # =========================================================

# class SchemaMatcher:
#     """
#     Very small SWAN-style schema matcher.

#     - Dictionary of attributes (table, column, description, aliases)
#     - Inverted index: token -> candidate attributes
#     - SentenceTransformer embeddings for semantic similarity
#     - Lexical similarity via SequenceMatcher
#     """

#     def __init__(self, entries):
#         self.entries = entries[:]  # list of dicts
#         for i, e in enumerate(self.entries):
#             e["id"] = i
#             text_bits = [e.get("column", ""), e.get("description", "")]
#             text_bits.extend(e.get("aliases", []))
#             e["text"] = " ".join([str(t) for t in text_bits if t])

#         self.inverted = defaultdict(set)  # token -> set(entry_ids)
#         for e in self.entries:
#             tokens = re.findall(r"\w+", e["text"].lower())
#             for tok in tokens:
#                 self.inverted[tok].add(e["id"])

#         # Embedding backend: A) sentence-transformers
#         self.model = SentenceTransformer("all-MiniLM-L6-v2")
#         texts = [e["text"] for e in self.entries]
#         self.entry_embeddings = self.model.encode(texts, normalize_embeddings=True)

#     def _lexical_sim(self, a: str, b: str) -> float:
#         return SequenceMatcher(None, a.lower(), b.lower()).ratio()

#     def match(self, query_text: str, top_k: int = 3, table_hint: str = None):
#         """
#         Return top_k schema entries for this query_text.
#         If table_hint given, restrict to that table.
#         """
#         if not query_text:
#             return []

#         qt = query_text.lower()
#         q_tokens = re.findall(r"\w+", qt)
#         candidate_ids = set()
#         for tok in q_tokens:
#             candidate_ids.update(self.inverted.get(tok, set()))

#         # fallback: all entries
#         if not candidate_ids:
#             candidate_ids = set(range(len(self.entries)))

#         candidate_ids = list(candidate_ids)

#         # Embedding similarity
#         q_emb = self.model.encode([query_text], normalize_embeddings=True)[0]
#         cand_embs = self.entry_embeddings[candidate_ids]
#         cos_sims = np.dot(cand_embs, q_emb)

#         scored = []
#         for idx, entry_id in enumerate(candidate_ids):
#             e = self.entries[entry_id]
#             if table_hint and e["table"] != table_hint:
#                 continue
#             emb_sim = float(cos_sims[idx])
#             lex_sim = self._lexical_sim(query_text, e["text"])
#             score = 0.7 * emb_sim + 0.3 * lex_sim
#             scored.append((score, e))

#         scored.sort(key=lambda x: x[0], reverse=True)
#         return [e for (s, e) in scored[:top_k]]

#     def best_match(self, query_text: str, table_hint: str = None):
#         m = self.match(query_text, top_k=1, table_hint=table_hint)
#         return m[0] if m else None


# # =========================================================
# # BUILD SCHEMA DICTIONARY (EXERCISE + NUTRITION)
# # =========================================================

# schema_entries = [
#     # --- exercise_fdw.exercises ---
#     {
#         "table": EXERCISE_TABLE,
#         "column": "name",
#         "description": "exercise name",
#         "aliases": ["exercise", "movement name", "workout name"],
#     },
#     {
#         "table": EXERCISE_TABLE,
#         "column": "category",
#         "description": "body part category like chest, back, legs, shoulders, arms, abs",
#         "aliases": ["body part", "muscle group", "category"],
#     },
#     {
#         "table": EXERCISE_TABLE,
#         "column": "muscles",
#         "description": "primary muscles targeted by the exercise",
#         "aliases": ["primary muscles", "target muscles", "main muscle", "muscles worked"],
#     },
#     {
#         "table": EXERCISE_TABLE,
#         "column": "muscles_secondary",
#         "description": "secondary muscles involved",
#         "aliases": ["secondary muscles", "supporting muscles"],
#     },
#     {
#         "table": EXERCISE_TABLE,
#         "column": "equipment",
#         "description": "equipment required for the exercise",
#         "aliases": ["equipment", "machine", "dumbbell", "barbell", "bodyweight"],
#     },

#     # --- nutrition_fdw.foods / public.nutrition ---
#     {
#         "table": NUTRITION_TABLE,
#         "column": "name",
#         "description": "food or product name",
#         "aliases": ["food name", "snack name", "meal name"],
#     },
#     {
#         "table": NUTRITION_TABLE,
#         "column": "energy_kcal",
#         "description": "energy in kilocalories",
#         "aliases": ["kcal", "calories", "energy", "calorie"],
#     },
#     {
#         "table": NUTRITION_TABLE,
#         "column": "protein_g",
#         "description": "protein grams per 100g or per serving",
#         "aliases": ["protein", "grams of protein", "protein content"],
#     },
#     {
#         "table": NUTRITION_TABLE,
#         "column": "fat_g",
#         "description": "fat grams",
#         "aliases": ["fat", "grams of fat", "fat content"],
#     },
#     {
#         "table": NUTRITION_TABLE,
#         "column": "carbohydrates_g",
#         "description": "carbohydrates grams",
#         "aliases": ["carbs", "carbohydrates"],
#     },
# ]

# schema_matcher = SchemaMatcher(schema_entries)

# # =========================================================
# # STEP 0 – READ USER QUERY
# # =========================================================

# user_query = input("\n💬 Enter your question: ").strip()
# print("\n📌 Your Query:", user_query)

# # =========================================================
# # STEP 1 – PARSE / DECOMPOSE NATURAL LANGUAGE QUERY
# # =========================================================

# def parse_nl_query(q: str) -> dict:
#     """
#     Ask LLM to convert user query to a structured intent mapping to our DW.
#     Also capture counts to support 'give 3 exercises' type queries.
#     """
#     prompt = f"""
# You are a query parser for a fitness + nutrition system.

# User query:
# {q}

# You MUST return ONLY JSON with EXACTLY these keys and types:

# {{
#   "muscle": null,                 // e.g. "chest", "back", "legs", "glutes", or null
#   "ask_protein": false,           // true if user cares about protein grams
#   "protein_filter": null,         // or {{ "op": "<", "value": 20.0 }} ; op in ["<", "<=", ">", ">=", "=", "=="]
#   "food_name": null,              // specific food/snack name if mentioned ("banana", "snack", "rice"), else null
#   "ask_unique_exercises": false,  // true if user asks for unique exercise names
#   "wants_plan": false,            // true if user wants a workout/diet plan (multi-day, structured), not just raw rows
#   "gluten_free": false,           // true if they ask gluten-free
#   "kcal_filter": null,            // or {{ "op": "<", "value": 200.0 }} for calorie constraints on food
#   "kcal_target": null,            // numeric kcal target if mentioned (e.g. 200)
#   "n_exercises": null,            // integer number of exercises requested (e.g. 3) or null
#   "n_days": null                  // integer number of days in plan (e.g. 3) or null
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
#         "kcal_filter": None,
#         "kcal_target": None,
#         "n_exercises": None,
#         "n_days": None,
#     }

#     txt, meta = call_llm(prompt, temperature=0.0)
#     parsed = safe_json_parse(txt, default)

#     # ensure keys exist
#     for k, v in default.items():
#         if k not in parsed:
#             parsed[k] = v

#     # Normalize some types
#     if isinstance(parsed.get("kcal_filter"), dict):
#         try:
#             parsed["kcal_filter"]["value"] = float(parsed["kcal_filter"]["value"])
#         except Exception:
#             parsed["kcal_filter"] = None
#     if parsed.get("kcal_target") is not None:
#         try:
#             parsed["kcal_target"] = float(parsed["kcal_target"])
#         except Exception:
#             parsed["kcal_target"] = None
#     if parsed.get("n_exercises") is not None:
#         try:
#             parsed["n_exercises"] = int(parsed["n_exercises"])
#         except Exception:
#             parsed["n_exercises"] = None
#     if parsed.get("n_days") is not None:
#         try:
#             parsed["n_days"] = int(parsed["n_days"])
#         except Exception:
#             parsed["n_days"] = None

#     return parsed


# parsed = parse_nl_query(user_query)
# print("🔍 Parsed Query:", parsed)

# # =========================================================
# # STEP 2 – STRUCTURED SEARCH: EXERCISES
# # =========================================================

# def search_exercises(parsed: dict):
#     muscle = parsed.get("muscle")
#     if not muscle:
#         return []

#     muscle_l = str(muscle).lower()
#     ask_unique = bool(parsed.get("ask_unique_exercises"))
#     n_ex = parsed.get("n_exercises") or 20

#     # Schema match for relevant columns (muscles, muscles_secondary, category)
#     attr_muscles = schema_matcher.best_match("primary muscles", table_hint=EXERCISE_TABLE)
#     attr_muscles_sec = schema_matcher.best_match("secondary muscles", table_hint=EXERCISE_TABLE)
#     attr_category = schema_matcher.best_match("body part category", table_hint=EXERCISE_TABLE)

#     col_muscles = attr_muscles["column"] if attr_muscles else "muscles"
#     col_muscles_sec = attr_muscles_sec["column"] if attr_muscles_sec else "muscles_secondary"
#     col_category = attr_category["column"] if attr_category else "category"

#     distinct_clause = "DISTINCT ON (LOWER(name))" if ask_unique else ""

#     sql = f"""
#         SELECT {distinct_clause}
#             id,
#             name,
#             category,
#             {col_muscles} AS muscles,
#             {col_muscles_sec} AS muscles_secondary,
#             equipment
#         FROM {EXERCISE_TABLE}
#         WHERE
#             LOWER(%s) = ANY (SELECT LOWER(m) FROM unnest({col_muscles}) m)
#             OR LOWER(%s) = ANY (SELECT LOWER(ms) FROM unnest({col_muscles_sec}) ms)
#             OR LOWER({col_category}) = %s
#         ORDER BY LOWER(name)
#         LIMIT %s;
#     """

#     print("🟦 Running PRIMARY SQL for exercises with muscle = ", muscle)
#     try:
#         df = pd.read_sql(sql, conn, params=[muscle_l, muscle_l, muscle_l, n_ex])
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


# # =========================================================
# # STEP 3 – STRUCTURED SEARCH: NUTRITION
# # =========================================================

# def search_nutrition(parsed: dict):
#     """
#     Use schema matching to decide which columns to query:
#     - protein_g for protein_filter
#     - energy_kcal for kcal_filter/target
#     - name for textual food_name
#     """
#     ask_protein = bool(parsed.get("ask_protein"))
#     protein_filter = parsed.get("protein_filter") or None
#     food_name = parsed.get("food_name")
#     kcal_filter = parsed.get("kcal_filter") or None
#     kcal_target = parsed.get("kcal_target")

#     # Schema matches
#     attr_name = schema_matcher.best_match("food name", table_hint=NUTRITION_TABLE)
#     attr_protein = schema_matcher.best_match("protein grams", table_hint=NUTRITION_TABLE)
#     attr_kcal = schema_matcher.best_match("energy kcal", table_hint=NUTRITION_TABLE)

#     col_name = attr_name["column"] if attr_name else "name"
#     col_protein = attr_protein["column"] if attr_protein else "protein_g"
#     col_kcal = attr_kcal["column"] if attr_kcal else "energy_kcal"

#     base_select = f"""
#         SELECT
#             id,
#             {col_name} AS name,
#             {col_protein} AS protein_g,
#             {col_kcal} AS energy_kcal,
#             fat_g AS fat_g
#         FROM {NUTRITION_TABLE}
#     """

#     where_parts = []
#     params = []

#     # Protein filter
#     if ask_protein and isinstance(protein_filter, dict):
#         op = protein_filter.get("op", ">=")
#         val = protein_filter.get("value", None)
#         if val is not None:
#             valid_ops = {"<", "<=", ">", ">=", "=", "=="}
#             if op not in valid_ops:
#                 op = ">="
#             if op == "==":
#                 op = "="
#             where_parts.append(f"{col_protein} IS NOT NULL AND {col_protein} {op} %s")
#             params.append(float(val))

#     # Kcal filter
#     if isinstance(kcal_filter, dict) and kcal_filter.get("value") is not None:
#         op2 = kcal_filter.get("op", "<")
#         kval = kcal_filter.get("value")
#         valid_ops = {"<", "<=", ">", ">=", "=", "=="}
#         if op2 not in valid_ops:
#             op2 = "<"
#         if op2 == "==":
#             op2 = "="
#         where_parts.append(f"{col_kcal} IS NOT NULL AND {col_kcal} {op2} %s")
#         params.append(float(kval))
#     elif kcal_target is not None:
#         # if only numeric target is given, default to "<="
#         where_parts.append(f"{col_kcal} IS NOT NULL AND {col_kcal} <= %s")
#         params.append(float(kcal_target))

#     # Food name text filter (e.g. "snack", "banana")
#     if food_name:
#         where_parts.append(f"LOWER({col_name}) LIKE %s")
#         params.append(f"%{food_name.lower()}%")

#     if not where_parts:
#         # no structured filter requested for nutrition
#         return []

#     where_sql = "WHERE " + " AND ".join(where_parts)

#     # Sorting: if protein filter exists with op <, sort asc; else desc
#     order_clause = "ORDER BY "
#     if ask_protein and isinstance(protein_filter, dict) and protein_filter.get("value") is not None:
#         op = protein_filter.get("op", ">=")
#         if op in ("<", "<="):
#             order_clause += f"{col_protein} ASC NULLS LAST"
#         else:
#             order_clause += f"{col_protein} DESC NULLS LAST"
#     elif kcal_filter or kcal_target is not None:
#         # default: kcal ascending
#         order_clause += f"{col_kcal} ASC NULLS LAST"
#     else:
#         order_clause += f"{col_name} ASC"

#     sql = f"""
#         {base_select}
#         {where_sql}
#         {order_clause}
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


# # =========================================================
# # RUN STRUCTURED SEARCHES
# # =========================================================

# exercise_results = search_exercises(parsed)
# nutrition_results = search_nutrition(parsed)

# print("\n🏋 Structured Exercise Results:")
# if exercise_results:
#     df_ex = pd.DataFrame(exercise_results)
#     cols = [c for c in ["id", "name", "category", "muscles", "muscles_secondary", "equipment"] if c in df_ex.columns]
#     print(df_ex[cols])
# else:
#     print("❌ None")

# print("\n🍎 Structured Nutrition Results:")
# if nutrition_results:
#     df_n = pd.DataFrame(nutrition_results)
#     cols = [c for c in ["id", "name", "protein_g", "energy_kcal", "fat_g"] if c in df_n.columns]
#     print(df_n[cols])
# else:
#     print("❌ None")

# has_ex_struct = len(exercise_results) > 0
# has_nut_struct = len(nutrition_results) > 0

# # =========================================================
# # DECIDE WHETHER WE NEED LLM (PLANNING / GAP-FILL)
# # =========================================================

# # Heuristic: wants_plan OR query mentions "workout/plan/diet/snack"
# wants_plan_flag = bool(parsed.get("wants_plan"))
# q_low = user_query.lower()
# if any(w in q_low for w in ["plan", "workout", "routine", "schedule"]):
#     wants_plan_flag = True

# llm_needed = False
# if wants_plan_flag:
#     llm_needed = True
# elif not has_ex_struct and not has_nut_struct:
#     llm_needed = True

# llm_data = None
# materialized_fact_id = None

# # =========================================================
# # IF NEEDED: CALL LLM TO SYNTHESIZE PLAN (HYBRID / PURE)
# # =========================================================

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

# STRUCTURED EXERCISE ROWS FROM DB (use them, do NOT invent new DB facts):
# {json.dumps(exercise_results, indent=2)}

# STRUCTURED NUTRITION ROWS FROM DB (use them, do NOT invent new DB facts):
# {json.dumps(nutrition_results, indent=2)}

# If there is no structured data, you may synthesize a reasonable workout and diet/snack plan.

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

#     llm_data = {
#         "reason": llm_json.get("reason", ""),
#         "exercise_plan": llm_json.get("exercise_plan", []) or [],
#         "diet_plan": llm_json.get("diet_plan", []) or [],
#         "summary": llm_json.get("summary", ""),
#         "score": float(llm_json.get("score", 0.8) or 0.8),
#         "confidence": float(llm_json.get("confidence", 0.8) or 0.8),
#     }

#     # ---- store in materialized DW table ----
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
#                 json.dumps(to_jsonable({
#                     "user_query": user_query,
#                     "parsed": parsed,
#                     "structured_exercises_used": exercise_results[:10],
#                     "structured_nutrition_used": nutrition_results[:10],
#                     "llm_output": llm_data,
#                     "prompt": gap_prompt,
#                     "model": GROQ_MODEL,
#                     "usage": llm_usage,
#                 })),
#                 llm_data["score"],
#                 llm_data["confidence"],
#             ),
#         )
#         materialized_fact_id = cur.fetchone()[0]
#         conn.commit()
#         print(f"\n💾 LLM inserted fact_id: {materialized_fact_id}")
#     except Exception as e:
#         conn.rollback()
#         print("❌ Failed to insert into materialized_recommendations:", e)

# # =========================================================
# # FINAL JSON RESULT
# # =========================================================

# final_result = {
#     "user_query": user_query,
#     "parsed": parsed,
#     "structured": {
#         "exercises": exercise_results,
#         "nutrition": nutrition_results,
#     },
#     "llm": llm_data,
#     "materialized_fact_id": materialized_fact_id,
# }

# print("\n======= FINAL JSON RESULT =======")
# print(json.dumps(to_jsonable(final_result), indent=2))

# # For dashboard / streamlit consumption
# print("[RESULT]", json.dumps(to_jsonable(final_result)))

# # Optional: quick SQL to inspect materialized table (run in psql / DBeaver, not from Python):
# #   SELECT fact_id, entity_type, score, confidence
# #   FROM trainer_dw.materialized_recommendations
# #   ORDER BY fact_id DESC
# #   LIMIT 10;

# cur.close()
# conn.close()
# print("\n🔚 Connection closed.")
















# ============================================================😀😀😀😀======================================

# # final_llm.py
# # SWAN-like trainer with schema-matching + embedding + LLM + materialized table

# import os
# import json
# import re
# import warnings
# from typing import List, Dict, Any, Optional, Tuple

# import psycopg2
# import pandas as pd
# from dotenv import load_dotenv
# from groq import Groq

# from sentence_transformers import SentenceTransformer, util  # embedding backend B: paraphrase-MiniLM-L12-v2

# warnings.filterwarnings("ignore")

# # ============================================================
# # 1. JSON + SMALL HELPERS
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
#     - Strips ``` fences
#     - Extracts first {...}
#     - Removes trailing commas
#     - On failure, returns default + attaches raw_text for debugging
#     """
#     if not text:
#         out = dict(default)
#         out["raw_text"] = None
#         return out

#     raw = text
#     t = raw.strip()

#     # strip ``` fences if present
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
#             # remove trailing commas before ] or }
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
#         out["raw_text"] = raw[:2000]  # don't store insane length
#         return out

# # ============================================================
# # 2. CONFIG, DB, LLM, EMBEDDINGS
# # ============================================================

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

# # Adjust these if your schemas differ
# EXERCISE_TABLE = os.getenv("EXERCISE_TABLE", "exercise_fdw.exercises")
# NUTRITION_TABLE = os.getenv("NUTRITION_TABLE", "nutrition_fdw.foods")  # change to public.nutrition if needed

# # Embedding model (B) - local
# EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L12-v2"
# print("🔧 Loading embedding model:", EMBEDDING_MODEL_NAME)
# embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# # ============================================================
# # 3. SCHEMA & SCHEMA-MATCHING DICTIONARY
# # ============================================================

# # Approx schema based on your screenshots; add more as needed
# EXERCISE_SCHEMA = {
#     "db": "exercise",
#     "table": EXERCISE_TABLE,
#     "columns": {
#         "id": {"type": "int"},
#         "uuid": {"type": "text"},
#         "name": {"type": "text"},
#         "category": {"type": "text"},
#         "muscles": {"type": "text[]"},
#         "muscles_secondary": {"type": "text[]"},
#         "equipment": {"type": "text[]"},
#     },
# }

# NUTRITION_SCHEMA = {
#     "db": "nutrition",
#     "table": NUTRITION_TABLE,
#     "columns": {
#         "id": {"type": "int"},
#         "name": {"type": "text"},
#         "protein_g": {"type": "float"},
#         "energy_kcal": {"type": "float"},
#         "fat_g": {"type": "float"},
#         "carbohydrates_total_g": {"type": "float"},
#         "sugar_g": {"type": "float"},
#         "fiber_g": {"type": "float"},
#         "sodium_mg": {"type": "float"},
#         # add any extra columns here from your full schema
#     },
# }

# # Schema matching dictionary: each entry has a label text that we embed
# SCHEMA_MATCH_DICTIONARY = [
#     # --- Exercise domain ---
#     {"db": "exercise", "table": EXERCISE_TABLE, "column": "muscles", "label": "muscle group"},
#     {"db": "exercise", "table": EXERCISE_TABLE, "column": "muscles", "label": "target muscle"},
#     {"db": "exercise", "table": EXERCISE_TABLE, "column": "category", "label": "exercise type"},
#     {"db": "exercise", "table": EXERCISE_TABLE, "column": "name", "label": "exercise name"},

#     # --- Nutrition domain ---
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "name", "label": "food name"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "protein_g", "label": "protein grams"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "protein_g", "label": "protein"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "energy_kcal", "label": "calories"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "energy_kcal", "label": "kcal"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "energy_kcal", "label": "energy"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "fat_g", "label": "fat grams"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "fat_g", "label": "fat"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "carbohydrates_total_g", "label": "carbs"},
#     {"db": "nutrition", "table": NUTRITION_TABLE, "column": "sugar_g", "label": "sugar"},
# ]

# # Conceptual constraints that are NOT in schema (for LLM only)
# CONCEPTUAL_ONLY_FIELDS = {
#     "gluten_free": [
#         "gluten free",
#         "gluten-free",
#         "without gluten"
#     ],
# }


# def build_schema_index():
#     """Pre-compute embedding for each schema match label."""
#     labels = [e["label"] for e in SCHEMA_MATCH_DICTIONARY]
#     embeddings = embed_model.encode(labels, convert_to_tensor=True, show_progress_bar=False)
#     return embeddings


# SCHEMA_LABEL_EMBEDDINGS = build_schema_index()


# def match_schema_label(phrase: str, top_k: int = 3, threshold: float = 0.4) -> List[Dict[str, Any]]:
#     """
#     Given a phrase like "fat", "calories", etc., return top matching schema entries
#     with cosine similarity above threshold.
#     """
#     phrase_emb = embed_model.encode([phrase], convert_to_tensor=True, show_progress_bar=False)
#     cos_scores = util.cos_sim(phrase_emb, SCHEMA_LABEL_EMBEDDINGS)[0]
#     top_results = torch_topk(cos_scores, top_k)

#     matches = []
#     for score, idx in top_results:
#         if float(score) < threshold:
#             continue
#         entry = dict(SCHEMA_MATCH_DICTIONARY[int(idx)])
#         entry["score"] = float(score)
#         matches.append(entry)
#     return matches


# def torch_topk(scores, k):
#     """Small helper to return (score, index) pairs sorted desc."""
#     # scores is a 1D tensor
#     import torch
#     topk = torch.topk(scores, k)
#     vals = topk.values.cpu().tolist()
#     idxs = topk.indices.cpu().tolist()
#     return list(zip(vals, idxs))


# # ============================================================
# # 4. DB CONNECTION
# # ============================================================

# try:
#     conn = psycopg2.connect(**DB_CONFIG)
#     print("✅ Connected to trainer_dw")
# except Exception as e:
#     print(f"❌ DB connection failed: {e}")
#     raise

# # ============================================================
# # 5. LLM CALL
# # ============================================================

# def call_llm(prompt: str, temperature: float = 0.0) -> Tuple[Optional[str], Dict[str, Any]]:
#     """Groq LLM call, returns (text, usage_meta)."""
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

# # ============================================================
# # 6. RULE-BASED + LLM-ASSISTED PARSER
# # ============================================================

# MUSCLE_WORDS = [
#     "chest", "back", "biceps", "triceps", "shoulders", "legs", "quads",
#     "hamstrings", "glutes", "abs", "core", "calves", "arms",
# ]

# NUTRITION_TRIGGER_WORDS = [
#     "food", "foods", "snack", "snacks", "meal", "diet", "eat", "calorie", "calories", "kcal",
#     "protein", "fat", "carb", "carbs", "sugar", "nutrition",
# ]

# EXERCISE_TRIGGER_WORDS = [
#     "exercise", "exercises", "workout", "workouts", "train", "training", "routine", "plan", "sets", "reps",
# ]


# def rule_based_parse(q: str) -> Dict[str, Any]:
#     """
#     Deterministic parser: muscles, numeric filters (protein/fat/kcal),
#     whether they want a plan, n_exercises, n_days, gluten-free, etc.
#     """
#     ql = q.lower()

#     # muscle
#     muscle = None
#     for m in MUSCLE_WORDS:
#         if m in ql:
#             muscle = m
#             break

#     # plan / days
#     wants_plan = any(word in ql for word in ["plan", "program", "routine", "schedule"])
#     n_days = None
#     m_days = re.search(r"(\d+)\s*(day|days)", ql)
#     if m_days:
#         n_days = int(m_days.group(1))

#     # number of exercises
#     n_exercises = None
#     m_ex = re.search(r"(\d+)\s*(unique\s+)?(exercise|exercises)", ql)
#     if m_ex:
#         n_exercises = int(m_ex.group(1))

#     ask_unique_exercises = "unique" in ql

#     # gluten free concept
#     gluten_free = any(k in ql for k in ["gluten free", "gluten-free", "without gluten"])

#     # numeric filters (protein, fat, energy)
#     # pattern examples: "more than 20g protein", "less than 10 g fat", "under 150 kcal"
#     numeric_filters: List[Dict[str, Any]] = []

#     # canonical comparators for words
#     word_to_op = {
#         "more than": ">",
#         "greater than": ">",
#         "over": ">",
#         "at least": ">=",
#         "less than": "<",
#         "fewer than": "<",
#         "under": "<",
#         "at most": "<=",
#         "no more than": "<=",
#     }

#     # 1) word-based comparators
#     for phrase, op in word_to_op.items():
#         pattern = rf"{phrase}\s*(\d+(\.\d+)?)\s*(g|grams?|kcal)?\s*([a-zA-Z_ ]+)?"
#         for m in re.finditer(pattern, ql):
#             val = float(m.group(1))
#             unit = m.group(3) or ""
#             attr_text = (m.group(4) or "").strip()
#             if not attr_text:
#                 # if unit+attr appear before, fallback; but keep simple
#                 continue
#             numeric_filters.append({
#                 "phrase": attr_text,
#                 "op": op,
#                 "value": val,
#                 "unit": unit.lower(),
#             })

#     # 2) operator-based comparators: "protein > 20g", "fat < 10 g", "energy_kcal <= 150"
#     pattern2 = r"([a-zA-Z_ ]+?)\s*(>=|<=|>|<|=|==)\s*(\d+(\.\d+)?)\s*(g|grams?|kcal)?"
#     for m in re.finditer(pattern2, ql):
#         attr_text = m.group(1).strip()
#         op = m.group(2)
#         val = float(m.group(3))
#         unit = m.group(5) or ""
#         numeric_filters.append({
#             "phrase": attr_text,
#             "op": op,
#             "value": val,
#             "unit": unit.lower(),
#         })

#     # map numeric filters to DB columns via schema matching
#     attr_filters = []
#     for filt in numeric_filters:
#         phrase = filt["phrase"]
#         matches = match_schema_label(phrase, top_k=3, threshold=0.4)
#         if not matches:
#             continue
#         # choose best match
#         best = matches[0]
#         attr_filters.append({
#             "db": best["db"],
#             "table": best["table"],
#             "column": best["column"],
#             "op": filt["op"],
#             "value": filt["value"],
#             "unit": filt["unit"],
#             "source_phrase": phrase,
#         })

#     # high-level flags: which DBs are needed?
#     needs_exercise = any(w in ql for w in EXERCISE_TRIGGER_WORDS) or (muscle is not None)
#     needs_nutrition = any(w in ql for w in NUTRITION_TRIGGER_WORDS)

#     # ask_protein is a simpler boolean, though attr_filters is more detailed
#     ask_protein = any("protein" in (f["source_phrase"] or "") for f in attr_filters)

#     parsed = {
#         "muscle": muscle,
#         "ask_protein": ask_protein,
#         "protein_filter": None,   # filled below if present
#         "food_name": None,
#         "ask_unique_exercises": ask_unique_exercises,
#         "wants_plan": wants_plan,
#         "gluten_free": gluten_free,
#         "kcal_filter": None,
#         "kcal_target": None,
#         "fat_filter": None,
#         "n_exercises": n_exercises,
#         "n_days": n_days,
#         "attr_filters": attr_filters,
#         "needs_exercise": needs_exercise,
#         "needs_nutrition": needs_nutrition,
#     }

#     # Fill convenience fields protein_filter / fat_filter / kcal_filter from attr_filters
#     for f in attr_filters:
#         if f["db"] != "nutrition":
#             continue
#         col = f["column"]
#         pf = {"op": f["op"], "value": f["value"]}
#         if col == "protein_g":
#             parsed["protein_filter"] = pf
#             parsed["ask_protein"] = True
#         elif col == "fat_g":
#             parsed["fat_filter"] = pf
#         elif col == "energy_kcal":
#             parsed["kcal_filter"] = pf

#     if parsed["kcal_filter"] and parsed["kcal_filter"].get("op") in ["<", "<="]:
#         parsed["kcal_target"] = parsed["kcal_filter"]["value"]

#     return parsed


# def llm_assisted_parse(q: str) -> Dict[str, Any]:
#     """
#     LLM-assisted parser but with strong guardrails and fallback to rule-based.
#     We *only* let LLM fill semantic fields; numeric + schema mapping still come from rule-based part.
#     """
#     # first, rule-based parse
#     rb = rule_based_parse(q)

#     # If the query is simple (short, no "and", no obvious complexity), we might skip LLM
#     if len(q) < 40 and "and" not in q.lower():
#         return rb

#     prompt = f"""
# You are a strict JSON generator for a fitness + nutrition system.

# User query:
# {q}

# You MUST output ONLY a single JSON object, with EXACTLY these keys:

# {{
#   "muscle": null,                 // e.g. "chest", "back", "glutes", or null
#   "wants_plan": false,            // true if multi-day plan / program requested
#   "ask_unique_exercises": false,  // true if they explicitly ask for "unique exercises"
#   "needs_exercise": false,        // true if they want any workout / exercise
#   "needs_nutrition": false,       // true if they want any food / diet / snack
#   "gluten_free": false,           // true if they ask for gluten-free food
#   "n_exercises": null,            // number of exercises if specified (e.g. "10 unique exercises")
#   "n_days": null                  // number of days if specified (e.g. "3 day plan")
# }}

# Rules:
# - DO NOT generate SQL.
# - DO NOT add extra fields.
# - DO NOT add comments outside JSON.
# - DO NOT include code fences.
# """

#     default = {
#         "muscle": rb["muscle"],
#         "wants_plan": rb["wants_plan"],
#         "ask_unique_exercises": rb["ask_unique_exercises"],
#         "needs_exercise": rb["needs_exercise"],
#         "needs_nutrition": rb["needs_nutrition"],
#         "gluten_free": rb["gluten_free"],
#         "n_exercises": rb["n_exercises"],
#         "n_days": rb["n_days"],
#     }

#     txt, meta = call_llm(prompt, temperature=0.0)
#     if not txt:
#         return rb

#     # If LLM output is suspiciously long or contains "SELECT", treat as invalid
#     if len(txt) > 1000 or "select" in txt.lower():
#         return rb

#     parsed = safe_json_parse(txt, default)
#     # merge semantic fields back into rb, but keep numeric filters & attr_filters from rule-based
#     rb["muscle"] = parsed.get("muscle", rb["muscle"])
#     rb["wants_plan"] = bool(parsed.get("wants_plan", rb["wants_plan"]))
#     rb["ask_unique_exercises"] = bool(parsed.get("ask_unique_exercises", rb["ask_unique_exercises"]))
#     rb["needs_exercise"] = bool(parsed.get("needs_exercise", rb["needs_exercise"]))
#     rb["needs_nutrition"] = bool(parsed.get("needs_nutrition", rb["needs_nutrition"]))
#     rb["gluten_free"] = bool(parsed.get("gluten_free", rb["gluten_free"]))
#     rb["n_exercises"] = parsed.get("n_exercises", rb["n_exercises"])
#     rb["n_days"] = parsed.get("n_days", rb["n_days"])

#     # Keep rb["attr_filters"], protein_filter, fat_filter, kcal_filter from rule-based
#     rb["raw_text"] = txt[:1000]
#     return rb

# # ============================================================
# # 7. SCHEMA-AWARE DB SEARCH
# # ============================================================

# def search_exercises(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
#     needs_ex = parsed.get("needs_exercise", False)
#     muscle = parsed.get("muscle")
#     n_exercises = parsed.get("n_exercises") or 20

#     if not needs_ex and not muscle:
#         return []

#     muscle_l = muscle.lower() if muscle else None

#     # If they want unique names, use DISTINCT ON
#     distinct_clause = "DISTINCT ON (LOWER(name))" if parsed.get("ask_unique_exercises") else ""

#     where_clauses = []
#     params = []

#     if muscle_l:
#         where_clauses.append("""
#             (
#               %s = ANY (SELECT LOWER(m) FROM unnest(muscles) m)
#               OR %s = ANY (SELECT LOWER(ms) FROM unnest(muscles_secondary) ms)
#             )
#         """)
#         params.extend([muscle_l, muscle_l])

#     where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

#     sql = f"""
#         SELECT {distinct_clause}
#             id,
#             name,
#             category,
#             muscles,
#             muscles_secondary,
#             equipment
#         FROM {EXERCISE_TABLE}
#         {where_sql}
#         ORDER BY LOWER(name)
#         LIMIT %s;
#     """
#     params.append(n_exercises)

#     print("🟦 Running PRIMARY SQL for exercises with muscle = ", muscle)
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


# def search_nutrition(parsed: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
#     """
#     Returns (nutrition_rows, unsatisfied_constraints_for_nutrition)
#     unsatisfied_constraints tracks things like gluten_free where we have no column in schema.
#     """
#     needs_nut = parsed.get("needs_nutrition", False)
#     if not needs_nut:
#         return [], []

#     attr_filters = parsed.get("attr_filters", []) or []
#     constraints = []
#     params = []

#     # Track which conceptual constraints cannot be mapped to schema
#     unsatisfied_constraints = []

#     # 1) Handle numeric filters that correspond to known columns
#     for f in attr_filters:
#         if f["db"] != "nutrition":
#             continue
#         col = f["column"]
#         op = f["op"]
#         val = f["value"]
#         # sanitize op
#         if op == "==":
#             op = "="
#         if op not in ["<", "<=", ">", ">=", "="]:
#             continue
#         constraints.append(f"{col} IS NOT NULL AND {col} {op} %s")
#         params.append(float(val))

#     # 2) Conceptual constraints like gluten_free that don't map to any column
#     if parsed.get("gluten_free", False):
#         # check if schema has something like gluten_free
#         if "gluten_free" not in NUTRITION_SCHEMA["columns"]:
#             unsatisfied_constraints.append("gluten_free")
#         else:
#             constraints.append("gluten_free = TRUE")

#     where_sql = ""
#     if constraints:
#         where_sql = "WHERE " + " AND ".join(constraints)

#     base_select = f"""
#         SELECT
#             id,
#             name,
#             protein_g,
#             energy_kcal,
#             fat_g
#         FROM {NUTRITION_TABLE}
#     """

#     sql = f"""
#         {base_select}
#         {where_sql}
#         ORDER BY
#             COALESCE(energy_kcal, 1000000) ASC,
#             COALESCE(fat_g, 1000000) ASC,
#             COALESCE(protein_g, -1) DESC
#         LIMIT 50;
#     """
#     print("🟩 Running SQL for nutrition with parsed filters")
#     try:
#         df = pd.read_sql(sql, conn, params=params)
#     except Exception as e:
#         print("⚠ Nutrition query failed:", e)
#         return [], unsatisfied_constraints

#     if df.empty:
#         return [], unsatisfied_constraints

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
#     return results, unsatisfied_constraints

# # ============================================================
# # 8. MAIN FLOW
# # ============================================================

# user_query = input("\n💬 Enter your question: ").strip()
# print("\n📌 Your Query:", user_query)

# parsed = llm_assisted_parse(user_query)
# print("🔍 Parsed Query:", parsed)

# exercise_results = search_exercises(parsed)
# nutrition_results, unsatisfied_nutrition_constraints = search_nutrition(parsed)

# print("\n🏋 Structured Exercise Results:")
# if exercise_results:
#     print(pd.DataFrame(exercise_results)[["id", "name", "category", "muscles", "muscles_secondary", "equipment"]])
# else:
#     print("❌ None")

# print("\n🍎 Structured Nutrition Results:")
# if nutrition_results:
#     print(pd.DataFrame(nutrition_results)[["id", "name", "protein_g", "energy_kcal", "fat_g"]])
# else:
#     print("❌ None")

# has_ex_struct = len(exercise_results) > 0
# has_nut_struct = len(nutrition_results) > 0

# needs_ex = parsed.get("needs_exercise", False)
# needs_nut = parsed.get("needs_nutrition", False)
# wants_plan = parsed.get("wants_plan", False)

# # ============================================================
# # 9. DECIDE LLM NEED (SWAN-LIKE)
# # ============================================================

# # unsatisfied concepts for LLM (not only nutrition)
# unsatisfied_concepts = []
# if unsatisfied_nutrition_constraints:
#     unsatisfied_concepts.extend(unsatisfied_nutrition_constraints)

# # Example: if muscle is None but they clearly want workout, that is another unsatisfied concept;
# # for now we focus on nutrition conceptual constraints like gluten_free.

# llm_needed = (
#     wants_plan
#     or (needs_ex and not has_ex_struct)
#     or (needs_nut and not has_nut_struct)
#     or (len(unsatisfied_concepts) > 0)
# )

# source = "db_only"
# llm_data = None
# materialized_fact_id = None
# cur = conn.cursor()

# if llm_needed:
#     source = "llm_only" if (not has_ex_struct and not has_nut_struct) else "hybrid"

#     DEFAULT_LLM_OUTPUT = {
#         "reason": "",
#         "exercise_plan": [],
#         "diet_plan": [],
#         "summary": "",
#         "score": 0.8,
#         "confidence": 0.8,
#     }

#     gap_info = {
#         "needs_exercise": needs_ex,
#         "needs_nutrition": needs_nut,
#         "unsatisfied_concepts": unsatisfied_concepts,
#     }

#     gap_prompt = f"""
# You are a fitness + nutrition expert that extends a virtual table combining exercise and foods.

# USER QUERY:
# {user_query}

# PARSED INTENT (JSON):
# {json.dumps({k: v for k, v in parsed.items() if k != "attr_filters"}, indent=2)}

# ATTR-LEVEL FILTERS (mapped via schema matching):
# {json.dumps(parsed.get("attr_filters", []), indent=2)}

# STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(exercise_results, indent=2)}

# STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
# {json.dumps(nutrition_results, indent=2)}

# GAP INFO:
# {json.dumps(gap_info, indent=2)}

# If some concepts (like gluten-free) are not represented in the DB schema, you must handle them at the LLM layer:
# - For example, if "gluten_free" is in unsatisfied_concepts, choose foods that are typically gluten-free.

# TASK:
# 1. Use the given DB rows where possible (e.g., respect numeric filters like protein > 20, fat < 10, kcal < 150).
# 2. For unsatisfied_concepts, use your world knowledge (e.g., gluten-free).
# 3. Build an exercise plan and a diet/snack recommendation consistent with the user query.
# 4. Think of this as materializing a row into a table exercise_food_recommendation.

# Return STRICT JSON ONLY:

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
#         "user_query": user_query,
#         "parsed": parsed,
#         "reason": reason,
#         "exercise_plan": exercise_plan,
#         "diet_plan": diet_plan,
#         "summary": summary,
#         "score": score,
#         "confidence": confidence,
#         "structured_exercises_used": exercise_results[:10],
#         "structured_nutrition_used": nutrition_results[:10],
#         "unsatisfied_concepts": unsatisfied_concepts,
#         "prompt": gap_prompt,
#         "model": GROQ_MODEL,
#         "usage": llm_usage,
#     }
#     llm_data = to_jsonable(llm_data)

#     # store only if we actually used LLM
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
#                 json.dumps(llm_data),
#                 score,
#                 confidence,
#             ),
#         )
#         materialized_fact_id = cur.fetchone()[0]
#         conn.commit()
#         print(f"\n💾 LLM inserted fact_id: {materialized_fact_id}")
#     except Exception as e:
#         conn.rollback()
#         print("❌ Failed to insert into materialized_recommendations:", e)
# else:
#     print("\n🎯 Answer can be served from structured DB only — no LLM needed.")

# # ============================================================
# # 10. FINAL JSON RESULT
# # ============================================================

# result_payload = {
#     "user_query": user_query,
#     "parsed": {k: v for k, v in parsed.items() if k != "attr_filters"},
#     "source": source,
#     "structured": {
#         "exercises": exercise_results,
#         "nutrition": nutrition_results,
#     },
#     "llm": llm_data,
#     "materialized_fact_id": materialized_fact_id,
# }

# print("\n======= FINAL JSON RESULT =======")
# print(json.dumps(to_jsonable(result_payload), indent=2))
# print("[RESULT]", json.dumps(to_jsonable(result_payload)))

# cur.close()
# conn.close()
# print("\n🔚 Connection closed.")





















# ================================================BESTEST==========================================================
# import os
# import re
# import json
# import psycopg2
# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# from typing import List, Dict, Any, Optional, Tuple

# # ======================
# #  CONFIG
# # ======================

# DB_DSN = os.getenv(
#     "TRAINER_DSN",
#     "dbname=trainer_dw user=postgres password=postgres host=localhost port=5432",
# )

# # Use embedding backend B: paraphrase-MiniLM-L12-v2
# EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L12-v2"


# # ======================
# #  SCHEMA DESCRIPTORS
# # ======================

# # Logical schema description for schema matching
# ATTR_CATALOG = [
#     # ---- EXERCISE DB ----
#     {
#         "key": "exercise.muscles",
#         "db": "exercise",
#         "table": "exercise_fdw.exercises",
#         "column": "muscles",
#         "description": "primary muscle or muscle group targeted by the exercise, e.g. back, shoulders, arms, glutes, legs, chest, abs",
#     },
#     {
#         "key": "exercise.muscles_secondary",
#         "db": "exercise",
#         "table": "exercise_fdw.exercises",
#         "column": "muscles_secondary",
#         "description": "secondary muscles involved in an exercise",
#     },
#     {
#         "key": "exercise.category",
#         "db": "exercise",
#         "table": "exercise_fdw.exercises",
#         "column": "category",
#         "description": "exercise category / body region such as Back, Legs, Arms, Shoulders, Chest, Abs",
#     },
#     {
#         "key": "exercise.name",
#         "db": "exercise",
#         "table": "exercise_fdw.exercises",
#         "column": "name",
#         "description": "exercise name, e.g. deadlift, squat, bench press",
#     },
#     {
#         "key": "exercise.equipment",
#         "db": "exercise",
#         "table": "exercise_fdw.exercises",
#         "column": "equipment",
#         "description": "equipment used in exercise, e.g. barbell, dumbbell, kettlebell, bodyweight",
#     },

#     # ---- NUTRITION DB ----
#     {
#         "key": "nutrition.name",
#         "db": "nutrition",
#         "table": "nutrition_fdw.foods",
#         "column": "name",
#         "description": "food or product name, e.g. chicken breast, apple, protein shake, snack",
#     },
#     {
#         "key": "nutrition.protein_g",
#         "db": "nutrition",
#         "table": "nutrition_fdw.foods",
#         "column": "protein_g",
#         "description": "grams of protein in food per 100g or per serving",
#     },
#     {
#         "key": "nutrition.fat_g",
#         "db": "nutrition",
#         "table": "nutrition_fdw.foods",
#         "column": "fat_g",
#         "description": "grams of fat in food per 100g or per serving",
#     },
#     {
#         "key": "nutrition.energy_kcal",
#         "db": "nutrition",
#         "table": "nutrition_fdw.foods",
#         "column": "energy_kcal",
#         "description": "energy content in kilocalories per 100g or per serving; calories, kcal",
#     },
# ]


# # Keywords for muscles / food / exercise intent
# MUSCLE_KEYWORDS = [
#     "back", "shoulder", "shoulders", "arm", "arms", "bicep", "biceps",
#     "tricep", "triceps", "leg", "legs", "quad", "quads", "hamstring",
#     "hamstrings", "glute", "glutes", "chest", "abs", "core", "lats"
# ]

# FOOD_KEYWORDS = [
#     "food", "snack", "meal", "diet", "eat", "eating", "breakfast",
#     "lunch", "dinner", "protein", "calorie", "calories", "kcal", "fat"
# ]

# EXERCISE_KEYWORDS = [
#     "exercise", "exercises", "workout", "train", "training", "routine"
# ]

# # Concepts that we know are *not* in DB schema (gap candidates)
# GAP_CONCEPT_KEYWORDS = [
#     "gluten-free", "gluten free", "vegan", "vegetarian", "lactose-free",
#     "low sodium", "sugar-free", "sugar free"
# ]


# # ======================
# #  EMBEDDING INITIALIZATION
# # ======================

# _model: Optional[SentenceTransformer] = None
# _attr_embeddings = None


# def get_model() -> SentenceTransformer:
#     global _model
#     if _model is None:
#         _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
#     return _model


# def init_attr_embeddings():
#     """
#     Precompute embeddings for ATTR_CATALOG descriptions.
#     """
#     global _attr_embeddings
#     if _attr_embeddings is not None:
#         return

#     model = get_model()
#     texts = [attr["description"] for attr in ATTR_CATALOG]
#     emb = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
#     _attr_embeddings = emb


# def best_attr_for_phrase(phrase: str, threshold: float = 0.45) -> Optional[Dict[str, Any]]:
#     """
#     Map a short phrase like 'protein', 'calories', 'back', 'snack' to
#     the most semantically similar attribute in ATTR_CATALOG using
#     paraphrase-MiniLM-L12-v2 embeddings.
#     """
#     if not phrase or phrase.strip() == "":
#         return None

#     init_attr_embeddings()
#     model = get_model()

#     phrase_emb = model.encode(phrase, convert_to_tensor=True, normalize_embeddings=True)
#     scores = util.cos_sim(phrase_emb, _attr_embeddings)[0]
#     best_idx = int(scores.argmax().item())
#     best_score = float(scores[best_idx])

#     if best_score < threshold:
#         return None

#     return ATTR_CATALOG[best_idx]


# # ======================
# #  PARSING UTILITIES
# # ======================

# NUMERIC_FILTER_PATTERN = re.compile(
#     r"(?P<op>under|less than|below|more than|greater than|over|at least|at most|>=|<=|>|<)\s*"
#     r"(?P<val>\d+(\.\d+)?)\s*"
#     r"(?P<unit>kcal|calories?|g|grams?)?\s*"
#     r"(?P<attr>protein|proteins|fat|fats|carb|carbs|carbohydrates|calories?|kcal)?",
#     re.IGNORECASE,
# )


# def normalize_op(op_word: str) -> str:
#     """
#     Map natural language comparative word to SQL operator.
#     """
#     op_word = op_word.lower()
#     if op_word in [">", "more than", "greater than", "over"]:
#         return ">"
#     if op_word in ["<", "under", "less than", "below"]:
#         return "<"
#     if op_word in [">=", "at least"]:
#         return ">="
#     if op_word in ["<=", "at most"]:
#         return "<="
#     return "=="


# def detect_muscle(text: str) -> Optional[str]:
#     """
#     Detect a muscle/body-part keyword from the query.
#     """
#     low = text.lower()
#     for m in MUSCLE_KEYWORDS:
#         if m in low:
#             return m
#     return None


# def contains_any(text: str, keywords: List[str]) -> bool:
#     low = text.lower()
#     return any(k in low for k in keywords)


# def extract_numeric_filters(text: str) -> List[Dict[str, Any]]:
#     """
#     Extract numeric filters like 'more than 15g protein', 'under 150 kcal',
#     and map them to attributes via schema matching.
#     """
#     filters = []
#     for m in NUMERIC_FILTER_PATTERN.finditer(text):
#         op_word = m.group("op")
#         val_str = m.group("val")
#         unit = m.group("unit")
#         attr_token = m.group("attr")

#         op = normalize_op(op_word)
#         try:
#             value = float(val_str)
#         except ValueError:
#             continue

#         # Decide phrase for attribute matching
#         if attr_token:
#             phrase = attr_token
#         elif unit and unit.lower().startswith("kcal") or (unit and "calorie" in unit.lower()):
#             phrase = "calories"
#         else:
#             # No attribute cue, skip
#             continue

#         # Use embeddings to map phrase -> attribute
#         attr = best_attr_for_phrase(phrase)
#         if not attr:
#             continue

#         filters.append(
#             {
#                 "db": attr["db"],
#                 "table": attr["table"],
#                 "column": attr["column"],
#                 "op": op,
#                 "value": value,
#                 "unit": unit,
#                 "source_phrase": phrase,
#             }
#         )

#     return filters


# def detect_gap_concepts(text: str) -> List[str]:
#     """
#     Detect gap concepts like 'gluten-free' that don't have direct schema columns.
#     """
#     gaps = []
#     low = text.lower()
#     for k in GAP_CONCEPT_KEYWORDS:
#         if k in low:
#             gaps.append(k)
#     return list(sorted(set(gaps)))


# def parse_user_query(text: str) -> Dict[str, Any]:
#     """
#     Parse user query into a structured intent object, using:
#       - keyword rules for muscles & high-level intent
#       - numeric filter extraction
#       - schema matching via embeddings
#     """
#     text = text.strip()
#     low = text.lower()

#     muscle = detect_muscle(text)

#     # Exercise / nutrition intent detection
#     needs_exercise = muscle is not None or contains_any(text, EXERCISE_KEYWORDS + MUSCLE_KEYWORDS)
#     needs_nutrition = contains_any(text, FOOD_KEYWORDS)

#     # Extract numeric attribute filters (protein/fat/kcal/etc.)
#     attr_filters = extract_numeric_filters(text)

#     # If numeric filters target nutrition attributes, force nutrition
#     if any(f["db"] == "nutrition" for f in attr_filters):
#         needs_nutrition = True

#     # Ask protein flag (for user intent)
#     ask_protein = "protein" in low or "high-protein" in low

#     # Unique exercises (e.g. '10 unique exercise names')
#     ask_unique_exercises = "unique" in low and "exercise" in low

#     # Number of exercises requested (simple pattern: '10 exercise(s)')
#     n_exercises = None
#     m_ex = re.search(r"(\d+)\s+exercise", low)
#     if m_ex:
#         n_exercises = int(m_ex.group(1))

#     # Number of days requested (e.g. '3 day plan')
#     n_days = None
#     m_days = re.search(r"(\d+)\s+day", low)
#     if m_days:
#         n_days = int(m_days.group(1))

#     wants_plan = "plan" in low or "routine" in low or "schedule" in low

#     # Food name direct request (simple heuristic: 'tell me about X' not needed here; you can extend)
#     food_name = None

#     # Gap concepts (no schema column)
#     unsatisfied_concepts = detect_gap_concepts(text)

#     parsed = {
#         "muscle": muscle,                     # e.g. 'back'
#         "ask_protein": ask_protein,
#         "protein_filter": None,              # DEPRECATED: using attr_filters instead
#         "food_name": food_name,
#         "ask_unique_exercises": ask_unique_exercises,
#         "wants_plan": wants_plan,
#         "gluten_free": ("gluten-free" in low or "gluten free" in low),
#         "kcal_filter": None,                 # DEPRECATED: using attr_filters instead
#         "kcal_target": None,                 # DEPRECATED
#         "fat_filter": None,                  # DEPRECATED
#         "n_exercises": n_exercises,
#         "n_days": n_days,
#         "attr_filters": attr_filters,        # list of {db, table, column, op, value, ...}
#         "needs_exercise": needs_exercise,
#         "needs_nutrition": needs_nutrition,
#         "unsatisfied_concepts": unsatisfied_concepts,
#         "raw_text": text,
#     }

#     return parsed


# # ======================
# #  DB QUERIES
# # ======================

# def search_exercises(conn, parsed: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Query exercise_fdw.exercises based on parsed intent.
#     - Mainly uses parsed["muscle"] to filter muscles/category.
#     """
#     if not parsed.get("needs_exercise"):
#         return pd.DataFrame()

#     muscle = parsed.get("muscle")
#     limit = parsed.get("n_exercises") or 20

#     if not muscle:
#         # No muscle specified → just return some generic exercises
#         sql = """
#             SELECT DISTINCT ON (LOWER(name))
#                 id, name, category, muscles, muscles_secondary, equipment
#             FROM exercise_fdw.exercises
#             ORDER BY LOWER(name), id
#             LIMIT %s;
#         """
#         return pd.read_sql(sql, conn, params=(limit,))

#     muscle_like = f"%{muscle.lower()}%"
#     sql = """
#         SELECT DISTINCT ON (LOWER(name))
#             id, name, category, muscles, muscles_secondary, equipment
#         FROM exercise_fdw.exercises
#         WHERE
#             LOWER(category) LIKE %(muscle_like)s
#             OR LOWER(muscles::text) LIKE %(muscle_like)s
#             OR LOWER(muscles_secondary::text) LIKE %(muscle_like)s
#         ORDER BY LOWER(name), id
#         LIMIT %(limit)s;
#     """
#     params = {
#         "muscle_like": muscle_like,
#         "limit": limit,
#     }
#     return pd.read_sql(sql, conn, params=params)


# def search_nutrition(conn, parsed: Dict[str, Any]) -> pd.DataFrame:
#     """
#     Query nutrition_fdw.foods based on parsed attribute filters.
#     - Uses attr_filters where db == 'nutrition'.
#     """
#     if not parsed.get("needs_nutrition"):
#         return pd.DataFrame()

#     attr_filters = parsed.get("attr_filters") or []

#     where_clauses = ["1=1"]
#     params = {}

#     idx = 0
#     for f in attr_filters:
#         if f["db"] != "nutrition":
#             continue
#         col = f["column"]
#         op = f["op"]
#         val = f["value"]
#         param_name = f"v{idx}"
#         # only allow safe ops
#         if op not in [">", "<", ">=", "<="]:
#             continue
#         where_clauses.append(f"{col} {op} %({param_name})s")
#         params[param_name] = val
#         idx += 1

#     # If no numeric filters at all, just return something simple
#     where_sql = " AND ".join(where_clauses)
#     limit = 50

#     sql = f"""
#         SELECT
#             id,
#             name,
#             protein_g,
#             energy_kcal,
#             fat_g
#         FROM nutrition_fdw.foods
#         WHERE {where_sql}
#         ORDER BY energy_kcal ASC NULLS LAST, protein_g DESC NULLS LAST
#         LIMIT {limit};
#     """

#     return pd.read_sql(sql, conn, params=params)


# # ======================
# #  MAIN LOOP
# # ======================

# def main():
#     print("✅ Connected to trainer_dw")

#     user_query = input("\n💬 Enter your question: ").strip()
#     if not user_query:
#         print("No query provided. Exiting.")
#         return

#     print(f"\n📌 Your Query: {user_query}")

#     # 1) Parse
#     parsed = parse_user_query(user_query)
#     print(f"🔍 Parsed Query: {json.dumps(parsed, indent=2)}")

#     # 2) Decide source mode
#     needs_ex = parsed.get("needs_exercise", False)
#     needs_nu = parsed.get("needs_nutrition", False)

#     if needs_ex and needs_nu:
#         source = "hybrid"
#     elif needs_ex:
#         source = "exercise_only"
#     elif needs_nu:
#         source = "nutrition_only"
#     else:
#         source = "none"

#     # 3) Connect to DB and run appropriate queries
#     conn = psycopg2.connect(DB_DSN)

#     try:
#         exercises_df = pd.DataFrame()
#         nutrition_df = pd.DataFrame()

#         if needs_ex:
#             print("🟦 Running SQL for exercises based on parsed intent")
#             exercises_df = search_exercises(conn, parsed)

#         if needs_nu:
#             print("🟩 Running SQL for nutrition based on parsed filters")
#             nutrition_df = search_nutrition(conn, parsed)

#         # 4) Print structured results (for debug / demo)
#         print("\n🏋 Structured Exercise Results:")
#         if exercises_df.empty:
#             print("❌ None")
#         else:
#             print(exercises_df.to_string(index=False))

#         print("\n🍎 Structured Nutrition Results:")
#         if nutrition_df.empty:
#             print("❌ None")
#         else:
#             print(nutrition_df.to_string(index=False))

#         # 5) Gap detection summary (for LLM later)
#         unsatisfied = parsed.get("unsatisfied_concepts", [])
#         if unsatisfied:
#             print("\n⚠ Unsatisfied (schema-gap) concepts detected:", unsatisfied)
#             # Here is where, in the next step, you'd call your LLM gap-filler
#             # and integrate its output with the structured results.

#         # 6) Final JSON-style result that your front-end / next stage can use
#         final_json = {
#             "user_query": user_query,
#             "parsed": parsed,
#             "source": source,
#             "structured": {
#                 "exercises": exercises_df.to_dict(orient="records"),
#                 "nutrition": nutrition_df.to_dict(orient="records"),
#             },
#             "llm": None,  # placeholder for later integration
#             "materialized_fact_id": None,
#         }

#         print("\n======= FINAL JSON RESULT =======")
#         print(json.dumps(final_json, indent=2, ensure_ascii=False))

#         print("\n[RESULT]", json.dumps(final_json, ensure_ascii=False))

#     finally:
#         conn.close()
#         print("\n🔚 Connection closed.")


# if __name__ == "__main__":
#     main()

















# ===================BESTEST 1=========================
# import psycopg2
# import json
# import pandas as pd
# import numpy as np
# import re
# import torch
# from sentence_transformers import SentenceTransformer, util
# from datetime import datetime

# # MODEL LOAD
# embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L12-v2")

# # DB CONNECTION
# def get_connection():
#     return psycopg2.connect(
#         host="localhost",
#         database="trainer_dw",
#         user="postgres",
#         password="root",
#         port="5432"
#     )

# # SCHEMA DEFINITIONS
# EXERCISE_COLUMNS = {
#     "name": "name",
#     "category": "category",
#     "muscles": "muscles",
#     "muscles_secondary": "muscles_secondary",
#     "equipment": "equipment"
# }

# NUTRITION_COLUMNS = {
#     "name": "name",
#     "protein": "protein_g",
#     "protein_g": "protein_g",
#     "fat": "fat_g",
#     "fat_g": "fat_g",
#     "calories": "energy_kcal",
#     "kcal": "energy_kcal",
#     "energy": "energy_kcal",
# }

# TABLE_MAP = {
#     "exercise": EXERCISE_COLUMNS,
#     "nutrition": NUTRITION_COLUMNS
# }

# STOPWORDS = {"me","give","tell","a","an","the","for","any","plan","food","exercise","workout","day","days","high","low","and"}

# def embed(text):
#     return embedding_model.encode(text, convert_to_tensor=True)

# def semantic_match(term, candidates):
#     term_emb = embed(term)
#     cand_embs = embed(list(candidates))
#     scores = util.pytorch_cos_sim(term_emb, cand_embs)[0]
#     best_idx = torch.argmax(scores).item()
#     best_score = float(scores[best_idx])
#     return list(candidates)[best_idx], best_score

# def parse_query(q):
#     tokens = re.findall(r"[a-zA-Z0-9]+", q.lower())
#     valid_terms = [t for t in tokens if t not in STOPWORDS]

#     matched_filters = []
#     needs_exercise = False
#     needs_nutrition = False
#     unmatched_terms = []

#     for term in valid_terms:
#         exercise_match, ex_score = semantic_match(term, EXERCISE_COLUMNS.keys())
#         nutrition_match, nu_score = semantic_match(term, NUTRITION_COLUMNS.keys())

#         if ex_score > 0.45 and ex_score > nu_score:
#             needs_exercise = True

#         elif nu_score > 0.45:
#             mapped_col = NUTRITION_COLUMNS[nutrition_match]
#             op = None
#             num = None
#             if ">" in q or "more" in q:
#                 op = ">"
#             elif "<" in q or "less" in q:
#                 op = "<"

#             found_nums = re.findall(r"\d+\.?\d*", q)
#             if found_nums:
#                 num = float(found_nums[0])

#             matched_filters.append({
#                 "db": "nutrition",
#                 "table": "nutrition_fdw.foods",
#                 "column": mapped_col,
#                 "op": op,
#                 "value": num,
#                 "source": term
#             })
#             needs_nutrition = True

#         else:
#             unmatched_terms.append(term)

#     return {
#         "raw_text": q,
#         "needs_exercise": needs_exercise,
#         "needs_nutrition": needs_nutrition,
#         "attr_filters": matched_filters,
#         "unmatched": unmatched_terms
#     }

# def run_exercise_sql(conn):
#     sql = "SELECT id,name,category,muscles,muscles_secondary,equipment FROM exercise_fdw.exercises LIMIT 20;"
#     return pd.read_sql(sql, conn)

# def run_nutrition_sql(conn, filters):
#     base = "SELECT id,name,protein_g,energy_kcal,fat_g FROM nutrition_fdw.foods"
#     if filters:
#         where = []
#         params = {}
#         for idx, f in enumerate(filters):
#             if f["value"] is not None and f["op"] is not None:
#                 where.append(f"{f['column']} {f['op']} %(v{idx})s")
#                 params[f"v{idx}"] = f["value"]
#         if where:
#             base += " WHERE " + " AND ".join(where)

#         return pd.read_sql(base, conn, params=params)

#     return pd.read_sql(base + " LIMIT 50", conn)

# # Dummy LLM placeholder
# def call_llm(context):
#     return {
#         "llm_reason": "Query contained concepts not found in database.",
#         "suggestion": f"Advice for {context} — generated using LLM.",
#         "score": 0.87
#     }

# def integrate_results(query, parsed, ex_df, nu_df, llm_part):
#     return {
#         "user_query": query,
#         "parsed": parsed,
#         "structured": {
#             "exercises": json.loads(ex_df.to_json(orient="records")) if ex_df is not None else [],
#             "nutrition": json.loads(nu_df.to_json(orient="records")) if nu_df is not None else []
#         },
#         "llm": llm_part,
#         "source": ("hybrid" if llm_part else ("db_only" if parsed["needs_exercise"] or parsed["needs_nutrition"] else "llm_only"))
#     }

# def main():
#     print("Connected to trainer_dw")
#     q = input("Enter your question: ")
#     parsed = parse_query(q)
#     conn = get_connection()

#     ex_df = None
#     nu_df = None

#     if parsed["needs_exercise"]:
#         print("→ Running exercise SQL")
#         ex_df = run_exercise_sql(conn)

#     if parsed["needs_nutrition"]:
#         print("→ Running nutrition SQL")
#         nu_df = run_nutrition_sql(conn, parsed["attr_filters"])

#     llm_part = None
#     if parsed["unmatched"]:
#         print("⚠ Gap detected, calling LLM for:", parsed["unmatched"])
#         llm_part = call_llm(parsed["unmatched"])

#     result = integrate_results(q, parsed, ex_df, nu_df, llm_part)
#     print(json.dumps(result, indent=4))
#     conn.close()

# if __name__ == "__main__":
#     main()















# =============================BESTEST 🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥====================================
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

# # Groq client (None if key missing)
# groq_client: Optional[Groq] = None
# if GROQ_API_KEY:
#     groq_client = Groq(api_key=GROQ_API_KEY)
# else:
#     print("[WARN] GROQ_API_KEY not set in environment. LLM gap-filling will be skipped.")

# # Model to use (stable Groq model)
# GROQ_MODEL = "llama-3.1-8b-instant"

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
#         # rename to match logs you showed
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
#     "full": "Full Body"
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

#     # If neither detected, but user mentions 'plan' or 'diet plan', assume nutrition
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

#     # --- "low-carb" detection -> concept only (nutrition), we DON'T have carbs column ---
#     if "low" in tokens and ("carb" in tokens or "carbs" in tokens):
#         # treat as conceptual requirement, no direct SQL column filter
#         parsed.needs_nutrition = True
#         parsed.unsatisfied_concepts.append("low-carb snack (no direct carb column)")

#     # --- Ask protein explicitly without numeric filter ---
#     if "protein" in tokens and not parsed.protein_filter:
#         parsed.ask_protein = True  # user may just want protein info in answer

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
#     recognized_tokens |= {"day", "days", "beginner", "advanced", "intermediate",
#                           "full", "body", "routine", "using", "use",
#                           "after", "before", "match", "matches", "matchup"}

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

#     params = {}
#     where_clauses = []

#     if parsed.muscle:
#         # Try to match on category OR muscles array containing substring
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
#     # pandas warning is fine; psycopg2 connection works
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
# # LLM JSON PARSING (ROBUST)
# # ------------------------------------------------------------------------------------

# def safe_llm_json_parse(content: str) -> Dict[str, Any]:
#     """
#     Robustly parse JSON from LLM response.
#     - Strips code fences.
#     - Extracts first {...} block.
#     - Tries to fix trailing commas.
#     - Returns {'raw_response': content} if parsing fails.
#     """
#     if not content:
#         return {"raw_response": ""}

#     text = content.strip()

#     # Strip ``` fences if present
#     if text.startswith("```"):
#         # remove first line (``` or ```json)
#         parts = text.split("\n", 1)
#         if len(parts) == 2:
#             text = parts[1].strip()
#         else:
#             text = text.strip("`").strip()
#     if text.endswith("```"):
#         text = text[:-3].strip()

#     # Extract JSON block from first { to last }
#     start = text.find("{")
#     end = text.rfind("}")
#     if start != -1 and end != -1 and end > start:
#         candidate = text[start : end + 1]
#     else:
#         candidate = text

#     def try_parse(s: str):
#         try:
#             return json.loads(s)
#         except Exception:
#             # fix trailing commas before } or ]
#             s2 = re.sub(r",(\s*[\]}])", r"\1", s)
#             try:
#                 return json.loads(s2)
#             except Exception:
#                 return None

#     parsed = try_parse(candidate)
#     if parsed is not None:
#         return parsed

#     # last resort: try full content
#     parsed = try_parse(text)
#     if parsed is not None:
#         return parsed

#     # give up: return raw response for debugging
#     return {"raw_response": content}


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
#         "protein_g": 0.0,
#         "fat_g": 0.0,
#         "energy_kcal": 0.0
#       }},
#       "gluten_free": null,
#       "serving_size": ""
#     }}
#   ],
#   "summary": "<plain-language summary for the user>"
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

#     prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

#     try:
#         completion = groq_client.chat.completions.create(
#             model=GROQ_MODEL,
#             messages=[
#                 {"role": "system", "content": "You are a precise, honest fitness and nutrition assistant."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,
#             max_tokens=900
#         )
#         content = completion.choices[0].message.content.strip()

#         # Debug: show a bit of raw content
#         print("\n🧠 Raw LLM response (first 400 chars):")
#         print(content[:400], "\n")

#         # Robust JSON parse
#         llm_json = safe_llm_json_parse(content)
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

#             if gap_tokens:
#                 print(f"⚠ Gap detected, calling LLM for: {gap_tokens}")
#                 llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
#                 if llm_result is not None:
#                     source = "hybrid"
#             else:
#                 # If absolutely no exercise/nutrition needed, but query exists, still call LLM
#                 if not parsed.needs_exercise and not parsed.needs_nutrition:
#                     print("⚠ Query seems out-of-scope for both databases; calling LLM anyway.")
#                     llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, [])
#                     if llm_result is not None:
#                         source = "llm_only"

#             # 6) Build final JSON result
#             result = {
#                 "user_query": user_query,
#                 "parsed": parsed.to_public_dict(),
#                 "source": source,
#                 "structured": {
#                     "exercises": df_ex.to_dict(orient="records") if df_ex is not None else [],
#                     "nutrition": df_nu.to_dict(orient="records") if df_nu is not None else [],
#                 },
#                 "llm": llm_result,
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

# ------------------------------------------------------------------------------------
# ENV + GLOBALS
# ------------------------------------------------------------------------------------

# Load .env BEFORE reading any env vars
load_dotenv(override=True)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "trainer_dw")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "1234")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Groq client (None if key missing)
groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    print("[WARN] GROQ_API_KEY not set in environment. LLM gap-filling will be skipped.")

# ------------------------------------------------------------------------------------
# PARSED QUERY DATACLASS
# ------------------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    raw_text: str

    muscle: Optional[str] = None  # primary muscle/category
    ask_protein: bool = False
    protein_filter: Optional[Dict[str, Any]] = None  # {op, value}
    food_name: Optional[str] = None

    ask_unique_exercises: bool = False
    wants_plan: bool = False
    gluten_free: bool = False  # kept for compatibility, but NOT applied as SQL
    kcal_filter: Optional[Dict[str, Any]] = None
    kcal_target: Optional[float] = None
    fat_filter: Optional[Dict[str, Any]] = None

    n_exercises: Optional[int] = None
    n_days: Optional[int] = None

    # attr-level filters for SQL building
    attr_filters: List[Dict[str, Any]] = None

    # flags telling which DBs are needed
    needs_exercise: bool = False
    needs_nutrition: bool = False

    # tokens that didn't map to any structured concept
    unsatisfied_concepts: List[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # rename to match logs you showed
        d["unmatched"] = d.pop("unsatisfied_concepts")
        return d


# ------------------------------------------------------------------------------------
# SIMPLE UTILITIES
# ------------------------------------------------------------------------------------

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
    "full": "Full Body"
}

def tokenize(text: str) -> List[str]:
    # simple word tokenizer
    return re.findall(r"[a-zA-Z0-9%]+", text.lower())


def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(title)
    print("-" * len(title))
    if df is None or df.empty:
        print("❌ None\n")
        return
    if len(df) > max_rows:
        display_df = df.head(max_rows)
    else:
        display_df = df
    print(display_df.to_string(index=False))
    print()


# ------------------------------------------------------------------------------------
# QUERY PARSING
# ------------------------------------------------------------------------------------

def parse_user_query(q: str) -> ParsedQuery:
    text = q.strip()
    tokens = tokenize(text)

    parsed = ParsedQuery(
        raw_text=text,
        attr_filters=[],
        unsatisfied_concepts=[]
    )

    # --- Decide if user needs exercise / nutrition ---
    exercise_markers = {"exercise", "exercises", "workout", "routine", "plan"}
    nutrition_markers = {
        "food", "foods", "snack", "snacks", "diet", "protein",
        "calories", "kcal", "carb", "carbs", "fat", "fats", "meal", "meals"
    }

    if any(t in exercise_markers or t in MUSCLE_KEYWORDS for t in tokens):
        parsed.needs_exercise = True
    if any(t in nutrition_markers for t in tokens):
        parsed.needs_nutrition = True

    # If neither detected, but user mentions 'plan' or 'diet plan', assume nutrition
    if not parsed.needs_exercise and not parsed.needs_nutrition:
        if "diet" in tokens or "meal" in tokens or "plan" in tokens:
            parsed.needs_nutrition = True

    # --- days / number of days ---
    m_days = re.search(r"(\d+)\s*day", text.lower())
    if m_days:
        parsed.n_days = int(m_days.group(1))
        parsed.wants_plan = True

    # --- unique exercises requested? ---
    if "different" in tokens or "unique" in tokens or "variety" in tokens:
        parsed.ask_unique_exercises = True

    # --- muscle / body-part detection (single main muscle) ---
    muscle_found = None
    for t in tokens:
        if t in MUSCLE_KEYWORDS:
            muscle_found = MUSCLE_KEYWORDS[t]
    parsed.muscle = muscle_found

    # --- Protein filter: e.g. "more than 15g protein", "at least 20g protein" ---
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

    # --- "low-carb" detection -> concept only (nutrition), we DON'T have carbs column ---
    if "low" in tokens and ("carb" in tokens or "carbs" in tokens):
        # treat as conceptual requirement, no direct SQL column filter
        parsed.needs_nutrition = True
        parsed.unsatisfied_concepts.append("low-carb snack (no direct carb column)")

    # --- Ask protein explicitly without numeric filter ---
    if "protein" in tokens and not parsed.protein_filter:
        parsed.ask_protein = True  # user may just want protein info in answer

    # --- Compute unmatched concept tokens for GAP ---
    recognized_tokens = set()

    # numbers, days
    for t in tokens:
        if t.isdigit():
            recognized_tokens.add(t)

    # recognized markers
    recognized_tokens |= exercise_markers
    recognized_tokens |= nutrition_markers
    recognized_tokens |= set(MUSCLE_KEYWORDS.keys())
    recognized_tokens |= {"day", "days", "beginner", "advanced", "intermediate",
                          "full", "body", "routine", "using", "use",
                          "after", "before", "match", "matches", "matchup"}

    for t in tokens:
        if t in STOPWORDS:
            continue
        if t in recognized_tokens:
            continue
        # Everything else is a candidate unmatched concept
        parsed.unsatisfied_concepts.append(t)

    return parsed


# ------------------------------------------------------------------------------------
# DB CONNECTION + SQL HELPERS
# ------------------------------------------------------------------------------------

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
    """
    Very simple exercise query:
    - If muscle detected: filter by category or muscles containing that word
    - Else: just return some generic exercises
    """
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

    params = {}
    where_clauses = []

    if parsed.muscle:
        # Try to match on category OR muscles array containing substring
        where_clauses.append(
            "(LOWER(category) = %(muscle_category)s OR %(muscle_like)s = ANY(ARRAY[LOWER(category)]) OR %(muscle_like)s = ANY(muscles))"
        )
        params["muscle_category"] = parsed.muscle.lower()
        params["muscle_like"] = parsed.muscle.lower()

    sql = base
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " LIMIT 50"

    print("→ Running exercise SQL")
    # pandas warning is fine; psycopg2 connection works
    return pd.read_sql(sql, conn, params=params)


def run_nutrition_sql(conn, parsed: ParsedQuery) -> pd.DataFrame:
    """
    Simple nutrition query:
    - If protein filter exists: apply it
    - Otherwise, just return some rows
    """
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


# ------------------------------------------------------------------------------------
# GAP DETECTION + LLM CALL
# ------------------------------------------------------------------------------------

def detect_gap_tokens(parsed: ParsedQuery) -> List[str]:
    # unsatisfied_concepts already built during parsing
    # Plus conceptual flags like "low-carb snack (no direct carb column)"
    return parsed.unsatisfied_concepts or []


def build_llm_prompt(
    user_query: str,
    parsed: ParsedQuery,
    df_ex: pd.DataFrame,
    df_nu: pd.DataFrame,
    gap_tokens: List[str]
) -> str:
    """
    Prompt that gives the LLM:
    - original user query
    - parsed intent (as JSON)
    - structured DB rows
    - unmatched / gap tokens
    and asks it to synthesize a final fitness + diet plan.
    """
    parsed_json = json.dumps(parsed.to_public_dict(), indent=2)
    ex_json = df_ex.head(20).to_dict(orient="records") if df_ex is not None else []
    nu_json = df_nu.head(20).to_dict(orient="records") if df_nu is not None else []

    prompt = f"""
You are a fitness + nutrition expert that extends a virtual table combining exercise and foods.

USER QUERY:
{user_query}

PARSED INTENT (JSON):
{parsed_json}

STRUCTURED EXERCISE ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
{json.dumps(ex_json, indent=2)}

STRUCTURED NUTRITION ROWS FROM DB (use them as ground truth, do NOT invent new DB facts):
{json.dumps(nu_json, indent=2)}

UNMATCHED / GAP TOKENS FROM THE QUERY (concepts not directly mapped to DB columns):
{gap_tokens}

TASK:
1. Interpret the user's goal.
2. Use ONLY the structured rows above as factual database items.
3. For parts that are not covered by the DB (the gap tokens), you may use your general knowledge to:
   - build a beginner-friendly or goal-specific workout structure,
   - choose which of the given DB rows are best for the request,
   - decide ordering, sets, reps, per-day structure, etc.
4. Return a rich but concise combined plan in **JSON** with this shape:

{{
  "reason": "<short reasoning>",
  "exercise_plan": [
    {{
      "day": 1,
      "exercise_name": "<name from exercises DB>",
      "sets": 3,
      "reps": 10,
      "notes": "<how to perform, level info, etc>"
    }}
  ],
  "diet_plan": [
    {{
      "name": "<name from nutrition DB>",
      "description": "<why this snack or food fits>",
      "macros": {{
        "protein_g": <float>,
        "fat_g": <float>,
        "energy_kcal": <float>
      }},
      "gluten_free": null,
      "serving_size": ""
    }}
  ],
  "summary": "<plain-language summary for the user>"
}}

If the query is completely outside the scope of both databases (no useful DB rows), you MUST still:
- explain that no DB rows matched,
- but still give a reasonable plan using your general knowledge,
- and clearly state that this part is purely LLM-generated, not from the DB.

Important:
- Use only exercises and foods that actually appear in the structured DB rows above.
- It's OK if some fields like gluten_free or serving_size remain empty/approximate.
"""
    return prompt


def call_llm_with_groq(
    user_query: str,
    parsed: ParsedQuery,
    df_ex: pd.DataFrame,
    df_nu: pd.DataFrame,
    gap_tokens: List[str]
) -> Optional[Dict[str, Any]]:
    """
    Two modes:
    1) In-domain (fitness/nutrition): use DB-aware JSON plan prompt.
    2) Out-of-domain (no exercise & no nutrition detected): answer generically
       but clearly say it's outside the project domain.
    """
    if groq_client is None:
        print("[WARN] GROQ client not initialised (no API key). Skipping LLM call.")
        return None

    # ---------- OUT-OF-DOMAIN MODE ----------
    if not parsed.needs_exercise and not parsed.needs_nutrition:
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful general assistant. "
                            "IMPORTANT: First, clearly state that this question is outside "
                            "the fitness/nutrition database domain of the project, "
                            "then still fully answer the user's question using your general knowledge."
                        ),
                    },
                    {"role": "user", "content": user_query},
                ],
                temperature=0.3,
                max_tokens=900,
            )
            content = completion.choices[0].message.content.strip()
            print("\n🧠 Raw LLM response (first 400 chars):")
            print(content[:400])
            # For out-of-domain, we don't force JSON, just return raw text
            return {
                "mode": "out_of_domain",
                "raw_response": content,
            }
        except Exception as e:
            print(f"[ERROR] LLM call (out-of-domain) failed: {e}")
            return None

    # ---------- HYBRID FITNESS + NUTRITION MODE ----------
    prompt = build_llm_prompt(user_query, parsed, df_ex, df_nu, gap_tokens)

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise, honest fitness and nutrition assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        content = completion.choices[0].message.content.strip()

        print("\n🧠 Raw LLM response (first 400 chars):")
        print(content[:400])

        # Try to locate JSON in the response; if it's plain JSON, parse directly.
        json_text = None
        if content.startswith("{"):
            json_text = content
        else:
            m = re.search(r"\{.*\}", content, flags=re.S)
            if m:
                json_text = m.group(0)

        if not json_text:
            print("[WARN] LLM response did not contain JSON. Returning raw text.")
            return {"raw_response": content}

        try:
            llm_json = json.loads(json_text)
            return llm_json
        except Exception as e:
            print(f"[WARN] Failed to parse LLM JSON, returning raw text instead: {e}")
            return {"raw_response": content}

    except Exception as e:
        print(f"[ERROR] LLM call (hybrid) failed: {e}")
        return None


# ------------------------------------------------------------------------------------
# MAIN MEDIATOR LOOP
# ------------------------------------------------------------------------------------

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

            # 1) Parse
            parsed = parse_user_query(user_query)
            print("🔍 Parsed Query:", json.dumps(parsed.to_public_dict(), indent=2))

            df_ex = pd.DataFrame()
            df_nu = pd.DataFrame()

            # 2) Run SQL for exercises (if requested)
            if parsed.needs_exercise:
                df_ex = run_exercise_sql(conn, parsed)

            # 3) Run SQL for nutrition (if requested)
            if parsed.needs_nutrition:
                df_nu = run_nutrition_sql(conn, parsed)

            # 4) Pretty-print structured results (tabular)
            pretty_print_df("🏋 Structured Exercise Results:", df_ex)
            pretty_print_df("🍎 Structured Nutrition Results:", df_nu)

            # 5) GAP detection
            gap_tokens = detect_gap_tokens(parsed)
            llm_result = None
            source = "db_only"

            # ---------- OUT-OF-DOMAIN PATH (no exercise & no nutrition needed) ----------
            if not parsed.needs_exercise and not parsed.needs_nutrition:
                print("ℹ Query is outside exercise/nutrition domain; calling LLM in general mode.")
                llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
                if llm_result is not None:
                    source = "llm_only"

            # ---------- IN-DOMAIN / HYBRID PATH ----------
            else:
                if gap_tokens:
                    print(f"⚠ Gap detected, calling LLM for: {gap_tokens}")
                    llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, gap_tokens)
                    if llm_result is not None:
                        source = "hybrid"
                else:
                    # Even if no explicit "gap tokens", user might still want a plan
                    # (e.g., wants_plan == True). We can still call LLM with DB rows.
                    if parsed.wants_plan or df_ex.empty or df_nu.empty:
                        print("ℹ Calling LLM to synthesize plan from structured DB rows.")
                        llm_result = call_llm_with_groq(user_query, parsed, df_ex, df_nu, [])
                        if llm_result is not None:
                            source = "hybrid"

            # 6) Build final JSON result
            result = {
                "user_query": user_query,
                "parsed": parsed.to_public_dict(),
                "source": source,
                "structured": {
                    "exercises": df_ex.to_dict(orient="records") if df_ex is not None else [],
                    "nutrition": df_nu.to_dict(orient="records") if df_nu is not None else [],
                },
                "llm": llm_result,
            }

            print("\n======= FINAL JSON RESULT =======")
            print(json.dumps(result, indent=4, ensure_ascii=False))
            print("\n")

    finally:
        conn.close()
        print("🔚 Connection closed.")


if __name__ == "__main__":
    main()
