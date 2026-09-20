import asyncio
import base64
import io
import json
import os
import random
import re
import secrets
import time
import sqlite3
import urllib.parse
from copy import deepcopy

import aiohttp
import discord
from dotenv import load_dotenv

from dnd_rules import (
    get_ability_modifier,
    get_proficiency_bonus,
    get_level_one_hp,
    get_armor_class,
    get_class_saving_throws,
    get_skill_ability_idx,
    get_skill_name,
    get_skill_modifier as rules_skill_modifier,
    get_monster_attack_data,
    get_monster_combat_data,
    get_initiative_modifier,
    resolve_attack_roll,
    resolve_attack_advantage_disadvantage,
    can_take_action,
    resolve_monster_attack_damage,
    resolve_weapon_attack,
    resolve_attack_damage,
    get_weapon_damage_dice,
    get_class_proficiencies,
    get_class,
    resolve_class_starting_equipment,
    get_starting_equipment_presets,
    get_manual_starting_equipment_options,
    resolve_manual_starting_equipment,
    get_weapon_data,
    get_condition,
    get_condition_effects,
    get_condition_effects,
)

def normalize_class_idx(character_class):
    """
    Normaliza una clase D&D al índice interno de la base de reglas.

    Acepta tanto el índice interno (ej. 'fighter') como
    el nombre en español (ej. 'Guerrero').
    """
    if not isinstance(character_class, str):
        return None

    value = character_class.strip()
    if not value:
        return None

    # Primero aceptar directamente el índice interno.
    direct = get_class(value.lower())
    if direct:
        return direct.get("idx")

    # Si no es un índice, buscar por nombre de clase.
    normalized = value.casefold()

    for class_idx in (
        "barbarian",
        "bard",
        "cleric",
        "druid",
        "fighter",
        "monk",
        "paladin",
        "ranger",
        "rogue",
        "sorcerer",
        "warlock",
        "wizard",
    ):
        data = get_class(class_idx)
        if not data:
            continue

        name = str(data.get("name", "")).strip()
        if name and name.casefold() == normalized:
            return class_idx

    return None

load_dotenv()

# ============================================================
# BOT MASTER D&D 5e — ARQUITECTURA HÍBRIDA
# ============================================================
# Motores de texto:
#   1) Groq GPT-OSS 120B
#   2) Groq Qwen 3.8 27B
#   3) Groq GPT-OSS 20B como tercera ruta gratuita
#   4) Gemini como respaldo multimodal
#   5) Gemma 4 E2B local vía LiteRT-LM, sin cuota de API
#
# IMPORTANTE:
# - No pongas claves reales dentro de este archivo.
# - Configúralas mediante variables de entorno en Termux.
# - El bot NO inventa resultados de dados: todas las tiradas se delegan a
#   Dice Golem. Si Dice Golem no responde, la acción queda pendiente.
# ============================================================

MEMORY_FILE = os.getenv("MEMORY_FILE", "partida_memoria.json")
FRESH_CAMPAIGN = not os.path.exists(MEMORY_FILE)
IMAGE_EVERY_ACTIONS = max(1, int(os.getenv("IMAGE_EVERY_ACTIONS", "5")))
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
POLLINATIONS_IMAGE_MODEL = os.getenv("POLLINATIONS_IMAGE_MODEL", "flux").strip() or "flux"

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

GEMINI_API_KEYS = [
    x.strip()
    for x in [
        os.getenv("GEMINI_API_KEY_1", ""),
        os.getenv("GEMINI_API_KEY_2", ""),
        os.getenv("GEMINI_API_KEY_3", ""),
    ]
    if x.strip()
]

# Modelos que tu clave de Groq debe mostrar en /models.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
]

# Gemini 3.6 Flash es actualmente un modelo estable según la documentación
# oficial de Google; se mantiene como respaldo multimodal.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# ------------------------------------------------------------
# Integración opcional con un bot externo de dados
# ------------------------------------------------------------
# Ejemplo en Termux:
# export DICE_GOLEM_ID="ID_DEL_BOT_DICE_GOLEM"
#
# {expression} será reemplazado por, por ejemplo, "1d20+5".
DICE_GOLEM_NAME = os.getenv("DICE_GOLEM_NAME", "Dice Golem").strip()
DICE_GOLEM_ID = os.getenv("DICE_GOLEM_ID", "").strip()
DICE_GOLEM_TIMEOUT = float(os.getenv("DICE_GOLEM_TIMEOUT", "30"))

# El modelo puede pedir estos niveles: low / medium / high.
DEFAULT_REASONING = os.getenv("DM_REASONING", "medium").lower()
if DEFAULT_REASONING not in {"low", "medium", "high"}:
    DEFAULT_REASONING = "medium"

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "40"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "6000"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1600"))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "5200"))
RECENT_HISTORY_MESSAGES = int(os.getenv("RECENT_HISTORY_MESSAGES", "10"))
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:9379/v1").strip().rstrip("/")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "gemma-4-E2B-it.litertlm").strip() or "gemma-4-E2B-it.litertlm"
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "").strip()
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "12"))

# Proveedores adicionales opcionales.
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b").strip() or "gpt-oss-120b"
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b").strip() or "nvidia/nemotron-3.5-lightning-30b-a3b"
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-120b:free").strip() or "openai/gpt-oss-120b:free"
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
AI_USAGE_FILE = os.getenv("AI_USAGE_FILE", "ai_usage.json")
DATABASE_FILE = os.getenv("DATABASE_FILE", "botmaster.db")
LOCAL_TASK_URL = os.getenv("LOCAL_TASK_URL", LOCAL_LLM_URL).strip().rstrip("/")
LOCAL_TASK_MODEL = os.getenv("LOCAL_TASK_MODEL", LOCAL_LLM_MODEL).strip() or "local-task"
LOCAL_TASK_API_KEY = os.getenv("LOCAL_TASK_API_KEY", LOCAL_LLM_API_KEY).strip()
LOCAL_TASK_TIMEOUT = float(os.getenv("LOCAL_TASK_TIMEOUT", "8"))

# Presupuesto local para evitar golpear los límites de Groq innecesariamente.
GROQ_SAFE_TPM_RESERVE = int(os.getenv("GROQ_SAFE_TPM_RESERVE", "1200"))
GROQ_SAFE_TPD_RESERVE = int(os.getenv("GROQ_SAFE_TPD_RESERVE", "12000"))
GROQ_DAILY_BUDGET_PER_MODEL = int(os.getenv("GROQ_DAILY_BUDGET_PER_MODEL", "180000"))
GROQ_USAGE_FILE = os.getenv("GROQ_USAGE_FILE", "groq_usage.json")

sheet_messages = {}
gemini_key_index = 0
channel_locks = {}

groq_limits = {}


def load_groq_usage():
    try:
        with open(GROQ_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("date") == time.strftime("%Y-%m-%d"):
            return data
    except Exception:
        pass
    return {"date": time.strftime("%Y-%m-%d"), "models": {}}


groq_usage = load_groq_usage()


def load_ai_usage():
    try:
        with open(AI_USAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("date") == time.strftime("%Y-%m-%d"):
            return data
    except Exception:
        pass
    return {"date": time.strftime("%Y-%m-%d"), "providers": {}}

ai_usage = load_ai_usage()

def record_ai_usage(provider, model, usage=None, ok=True, error=None):
    global ai_usage
    today = time.strftime("%Y-%m-%d")
    if ai_usage.get("date") != today:
        ai_usage = {"date": today, "providers": {}}
    bucket = ai_usage.setdefault("providers", {}).setdefault(provider, {}).setdefault(model, {"requests":0,"errors":0,"input_tokens":0,"output_tokens":0,"total_tokens":0})
    bucket["requests" if ok else "errors"] += 1
    if isinstance(usage, dict):
        inp=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total=int(usage.get("total_tokens") or inp+out)
        bucket["input_tokens"] += inp; bucket["output_tokens"] += out; bucket["total_tokens"] += total
    if error: bucket["last_error"] = str(error)[:300]
    try:
        tmp=AI_USAGE_FILE+".tmp"
        with open(tmp,"w",encoding="utf-8") as f: json.dump(ai_usage,f,ensure_ascii=False,indent=2)
        os.replace(tmp,AI_USAGE_FILE)
    except Exception as exc: print(f"[AI USAGE] No se pudo guardar: {exc}")

def groq_daily_used(model):
    return int(groq_usage.setdefault("models", {}).get(model, 0))


def record_groq_usage(model, tokens):
    if tokens <= 0:
        return
    current_date = time.strftime("%Y-%m-%d")
    if groq_usage.get("date") != current_date:
        groq_usage.clear()
        groq_usage.update({"date": current_date, "models": {}})
    groq_usage.setdefault("models", {})[model] = groq_daily_used(model) + int(tokens)
    try:
        tmp = GROQ_USAGE_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(groq_usage, f, ensure_ascii=False, indent=2)
        os.replace(tmp, GROQ_USAGE_FILE)
    except Exception as exc:
        print(f"[GROQ USAGE] No se pudo guardar: {exc}")


def get_channel_lock(channel_id):
    return channel_locks.setdefault(channel_id, asyncio.Lock())

# ============================================================
# ESTADO PERSISTENTE
# ============================================================
def default_character_sheet(name=""):
    """
    Estructura reutilizable para cualquier personaje de una campaña.
    Los valores mecánicos reales se definirán al crear/configurar
    el personaje; no se inventan aquí.
    """
    return {
        "identity": {
            "name": name,
            "race": "",
            "class": "",
            "subclass": "",
            "level": None,
        },

        "abilities": {
            "strength": None,
            "dexterity": None,
            "constitution": None,
            "intelligence": None,
            "wisdom": None,
            "charisma": None,
        },

        "combat": {
            "hp_current": 0,
            "hp_max": 0,
            "ac": 10,
            "initiative": 0,
            "speed": 30,
            "conditions": [],
        },

        "proficiency": {
            "bonus": None,
            "skills": [],
            "saving_throws": [],
        },

        "inventory": [],
        "currency": {
            "pp": 0,
            "gp": 0,
            "ep": 0,
            "sp": 0,
            "cp": 0,
        },
        "spells": [],
        "notes": [],
    }

def default_campaign_state():
    return {
        "campaign_id": secrets.token_hex(8),
        "player_bindings": {
            "622776087050453005": "Jugador 1",
            "790733674823811073": "Jugador 2",
        },
        "character_creation": None,
        "campaign": {
            "title": "Campaña D&D",
            "current_scene": "",
            "weather": "",
            "time": "",
            "location": "",
        },
        "characters": {},
        "npcs": {},
        "locations": {},
        "quests": {
            "main": "",
            "clues": [],
            "active": [],
            "completed": [],
        },
        "factions": {},
        "loot": {
            "po": 0,
            "pp": 0,
            "pc": 0,
            "items": [],
        },
        "combat": {
            "active": False,
            "round": 0,
            "turn": None,
            "combatants": [],
            "initiative": [],
            "monster_state": {},
            "conditions": {},
            "concentration": {},
            "reaction_used": {},
        },
        "clocks": [],
        "secrets": [],
        "facts": [],
        "last_roll": None,
        "image_counter": 0,
        "discord_panels": {
            "channel_id": None,
            "message_ids": {
                "fichas": None,
                "botin": None,
                "facciones": None,
                "misiones": None,
            },
        },
        "updated_at": time.time(),
    }


def load_data():
    default = default_campaign_state()
    if not os.path.exists(MEMORY_FILE):
        return "", [], default

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Compatibilidad con el formato antiguo.
            summary = data.get("summary", "")
            history = data.get("history", [])

            state = data.get("state", default)
            if not isinstance(state, dict):
                state = default

            merged = default
            merged.update(state)

            if not merged.get("campaign_id"):
                merged["campaign_id"] = secrets.token_hex(8)

            return (
                summary,
                history if isinstance(history, list) else [],
                merged,
            )

        if isinstance(data, list):
            return "", data, default

    except Exception as exc:
        print(f"[MEMORIA] No se pudo cargar: {exc}")

        recovered_state = load_latest_sqlite_snapshot()

        if recovered_state:
            print(
                "[RECOVERY] Estado recuperado desde el último snapshot SQLite."
            )
            return "", [], recovered_state

    return "", [], default


def load_latest_sqlite_snapshot(campaign_id=None):
    """Recupera el último snapshot SQLite válido de una campaña."""
    if not os.path.exists(DATABASE_FILE):
        return None

    try:
        with sqlite3.connect(DATABASE_FILE) as conn:
            if campaign_id:
                row = conn.execute(
                    """
                    SELECT state_json, campaign_id
                    FROM state_snapshots
                    WHERE campaign_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (campaign_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT state_json, campaign_id
                    FROM state_snapshots
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()

        if not row or not row[0]:
            return None

        state = json.loads(row[0])

        if not isinstance(state, dict):
            return None

        snapshot_campaign_id = row[1]

        if snapshot_campaign_id:
            state["campaign_id"] = snapshot_campaign_id
        elif campaign_id:
            state["campaign_id"] = campaign_id

        return state

    except Exception as exc:
        print(f"[SQLite] No se pudo recuperar snapshot: {exc}")
        return None

def save_data(summary, history, state):
    state = deepcopy(state)
    state["updated_at"] = time.time()
    tmp_file = f"{MEMORY_FILE}.tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "history": history[-MAX_HISTORY_MESSAGES:], "state": state},
            f,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(tmp_file, MEMORY_FILE)


long_term_summary, chat_history, campaign_state = load_data()

# ============================================================
# SQLITE: HISTORIAL PERMANENTE Y SNAPSHOTS
# ============================================================

def init_database():
    """Crea y migra la base local sin depender de servicios externos."""
    db_dir = os.path.dirname(os.path.abspath(DATABASE_FILE))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    campaign_id = campaign_state.get("campaign_id")
    if not campaign_id:
        raise RuntimeError("No hay campaign_id disponible para inicializar SQLite")

    with sqlite3.connect(DATABASE_FILE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS campaign_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT UNIQUE NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                state_json TEXT NOT NULL,
                summary TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS important_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT UNIQUE NOT NULL,
                timestamp REAL NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'botmaster'
            )
        """)

        migrations = {
            "campaign_history": "campaign_id",
            "state_snapshots": "campaign_id",
            "important_memory": "campaign_id",
        }

        for table, column in migrations.items():
            columns = [
                row[1]
                for row in conn.execute(f"PRAGMA table_info({table})")
            ]

            if column not in columns:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} TEXT"
                )

            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE {column} IS NULL",
                (campaign_id,),
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_timestamp "
            "ON campaign_history(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_timestamp "
            "ON important_memory(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_campaign "
            "ON campaign_history(campaign_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_snapshots_campaign "
            "ON state_snapshots(campaign_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_campaign "
            "ON important_memory(campaign_id)"
        )

        conn.commit()

def _history_event_key(msg, index=0, campaign_id=None):
    """Genera una clave estable y aislada por campaña para no duplicar eventos históricos."""
    event_id = msg.get("event_id") if isinstance(msg, dict) else None
    if event_id:
        raw = f"{campaign_id or ''}|event|{event_id}"
    else:
        role = str(msg.get("role", "user")) if isinstance(msg, dict) else "user"
        content = str(msg.get("content", "")) if isinstance(msg, dict) else str(msg)
        timestamp = msg.get("timestamp") if isinstance(msg, dict) else None
        raw = f"{campaign_id or ''}|legacy|{index}|{timestamp}|{role}|{content}"
    return __import__("hashlib").sha256(raw.encode("utf-8")).hexdigest()

def archive_history_events(history):
    """Archiva el historial completo sin eliminarlo al compactar el contexto."""
    if not history:
        return 0

    campaign_id = campaign_state.get("campaign_id")
    if not campaign_id:
        return 0

    inserted = 0

    with sqlite3.connect(DATABASE_FILE) as conn:
        for index, msg in enumerate(history):
            if not isinstance(msg, dict):
                continue

            content = str(msg.get("content", "")).strip()
            if not content:
                continue

            try:
                timestamp = float(msg.get("timestamp", 0) or 0) or time.time()
            except (TypeError, ValueError):
                timestamp = time.time()

            role = str(msg.get("role", "user"))
            key = _history_event_key(msg, index, campaign_id)

            cur = conn.execute(
                """
                INSERT OR IGNORE INTO campaign_history(
                    event_key, timestamp, role, content, campaign_id
                ) VALUES(?,?,?,?,?)
                """,
                (key, timestamp, role, content, campaign_id),
            )

            inserted += cur.rowcount

        conn.commit()

    return inserted

def save_state_snapshot():
    """Guarda una fotografía estructurada del estado actual de la campaña."""
    campaign_id = campaign_state.get("campaign_id")
    if not campaign_id:
        return

    try:
        payload = json.dumps(
            campaign_state,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        with sqlite3.connect(DATABASE_FILE) as conn:
            conn.execute(
                """
                INSERT INTO state_snapshots(
                    timestamp, state_json, summary, campaign_id
                ) VALUES(?,?,?,?)
                """,
                (
                    time.time(),
                    payload,
                    long_term_summary[-MAX_SUMMARY_CHARS:],
                    campaign_id,
                ),
            )
            conn.commit()

    except Exception as exc:
        print(f"[SQLite] Error guardando snapshot: {exc}")


def save_important_fact_memory():
    """Replica hechos estructurados en memoria permanente, aislados por campaña."""
    campaign_id = campaign_state.get("campaign_id")
    if not campaign_id:
        return

    facts = campaign_state.get("facts", [])
    if not isinstance(facts, list) or not facts:
        return

    with sqlite3.connect(DATABASE_FILE) as conn:
        for fact in facts:
            text = str(fact).strip()[:1000]
            if not text:
                continue

            key = __import__("hashlib").sha256(
                f"{campaign_id}|fact|{text}".encode("utf-8")
            ).hexdigest()

            conn.execute(
                """
                INSERT OR IGNORE INTO important_memory(
                    event_key, timestamp, content, source, campaign_id
                ) VALUES(?,?,?,?,?)
                """,
                (
                    key,
                    time.time(),
                    text,
                    "campaign_state.facts",
                    campaign_id,
                ),
            )

        conn.commit()


try:
    init_database()
    archive_history_events(chat_history)
    save_important_fact_memory()
    print(f"[SQLite] Base inicializada: {os.path.abspath(DATABASE_FILE)}")
except Exception as exc:
    print(f"[SQLite] ERROR inicializando la base: {exc}")

# ============================================================
# UTILIDADES DE MEMORIA Y ESTADO
# ============================================================

def _clip_text(value, max_chars):
    text = str(value or "")
    return text if len(text) <= max_chars else text[:max_chars] + "…"

def campaign_state_text(max_chars=7000, sections=None):
    """Genera solo las secciones de estado solicitadas."""
    combat = campaign_state.get("combat", {})
    factions = campaign_state.get("factions", {})
    loot = campaign_state.get("loot", {})
    quests = campaign_state.get("quests", {})
    chars = campaign_state.get("characters", {})
    npcs = campaign_state.get("npcs", {})

    if sections is None:
        sections = {
            "campaign",
            "characters",
            "npcs",
            "quests",
            "factions",
            "loot",
            "combat",
            "clocks",
            "facts",
            "last_roll",
        }

    payload = {}

    if "campaign" in sections:
        payload["campaign"] = campaign_state.get("campaign", {})

    if "characters" in sections:
        payload["characters"] = chars

    if "npcs" in sections:
        npc_lines = []
        if isinstance(npcs, dict):
            for name, npc in list(npcs.items())[-8:]:
                if not isinstance(npc, dict):
                    continue
                npc_lines.append({
                    "name": name,
                    "attitude": npc.get("attitude", "?"),
                    "goal": npc.get("goal", "?"),
                    "knows": (
                        npc.get("knows", [])[-5:]
                        if isinstance(npc.get("knows"), list)
                        else []
                    ),
                    "relationships": npc.get("relationships", {}),
                })
        payload["npcs_relevantes"] = npc_lines

    if "quests" in sections:
        payload["quests"] = {
            "main": quests.get("main"),
            "active": quests.get("active", [])[-8:],
            "clues": quests.get("clues", [])[-8:],
            "completed": quests.get("completed", [])[-5:],
        }

    if "factions" in sections:
        payload["factions"] = factions

    if "loot" in sections:
        payload["loot"] = loot

    if "combat" in sections:
        combatants = (
            combat.get("combatants", [])
            if isinstance(combat, dict)
            else []
        )

        if isinstance(combat, dict) and combat.get("active"):
            payload["combat"] = {
                "active": True,
                "round": combat.get("round", 0),
                "turn": combat.get("turn"),
                "combatants": combatants,
            }
        else:
            payload["combat"] = "INACTIVO"

    if "clocks" in sections:
        payload["clocks"] = campaign_state.get("clocks", [])[-8:]

    if "facts" in sections:
        payload["facts"] = campaign_state.get("facts", [])[-15:]

    if "last_roll" in sections:
        payload["last_roll"] = campaign_state.get("last_roll")

    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return _clip_text(text, max_chars)


def build_context_message(user_text):
    """Construye contexto selectivo según el tipo de acción."""
    text_lower = str(user_text or "").strip().lower()

    # Presupuesto de entrada independiente del límite de salida.
    dynamic_budget_chars = max(7000, (MAX_CONTEXT_TOKENS - 1500) * 4)

    summary = _clip_text(
        long_term_summary or "Inicio de la aventura.",
        min(2600, dynamic_budget_chars // 5),
    )

    # --------------------------------------------------------
    # Selección conservadora de contexto
    # --------------------------------------------------------

    sections = {
        "campaign",
        "characters",
        "npcs",
        "quests",
        "combat",
        "facts",
        "last_roll",
    }

    # Combate / acciones físicas.
    if re.search(
        r"\b(ataco|atacar|ataque|golpeo|golpear|disparo|disparar|"
        r"lanzo|lanzar|esquivo|esquivar|bloqueo|bloquear|peleo|"
        r"pelear|combato|combatir|daño|danio|herido|iniciativa)\b",
        text_lower,
    ):
        sections.update({"npcs", "combat", "last_roll", "campaign"})

    # Inventario / recursos / botín.
    elif re.search(
        r"\b(inventario|llevo|objeto|objetos|equipo|equipamiento|"
        r"botín|botin|tesoro|monedas|dinero|oro|po|pp|pc|"
        r"comprar|vendo|vender|tomo|recoger|recogerlo)\b",
        text_lower,
    ):
        sections = {
            "campaign",
            "characters",
            "quests",
            "loot",
        }

    # NPC / diálogo / interacción social.
    elif re.search(
        r"\b(hablo|hablar|decir|pregunto|preguntar|respondo|"
        r"responder|convencer|persuadir|intimidar|amenazo|"
        r"amenazar|npc|personaje|rey|reina|guardia|mercader|"
        r"aldeano|soldado)\b",
        text_lower,
    ):
        sections = {
            "campaign",
            "characters",
            "npcs",
            "quests",
            "factions",
            "facts",
        }

    # Misiones / objetivos / pistas.
    elif re.search(
        r"\b(misión|mision|misiones|objetivo|objetivos|pista|"
        r"pistas|investigo|investigar|búsqueda|busqueda|"
        r"encargo|encargos)\b",
        text_lower,
    ):
        sections = {
            "campaign",
            "characters",
            "quests",
            "npcs",
            "facts",
        }

    # Ubicación / exploración / desplazamiento.
    elif re.search(
        r"\b(dónde|donde|ubicación|ubicacion|lugar|camino|"
        r"viajo|viajar|voy|ir|exploro|explorar|entro|salgo|"
        r"puerta|habitación|habitacion|cueva|bosque|ciudad|"
        r"aldea|mazmorra)\b",
        text_lower,
    ):
        sections = {
            "campaign",
            "characters",
            "npcs",
            "quests",
            "facts",
        }

    state = campaign_state_text(
        min(7000, dynamic_budget_chars // 2),
        sections=sections,
    )

    action = _clip_text(user_text, 1800)

    text = (
        "CONTEXTO DINÁMICO DEL DM (usa solo lo relevante para esta acción):\n"
        f"MEMORIA: {summary}\n"
        f"ESTADO: {state}\n"
        f"ACCIÓN ACTUAL: {action}"
    )

    return _clip_text(text, dynamic_budget_chars)


def compact_recent_history(history, max_messages=RECENT_HISTORY_MESSAGES, max_chars=9000):
    """Conserva solo las últimas interacciones y limita su tamaño."""
    selected = []
    total = 0
    for msg in reversed(history):
        if not isinstance(msg, dict):
            continue
        content = _clip_text(msg.get("content", ""), 1600)
        if not content:
            continue
        item = {"role": msg.get("role", "user"), "content": content}
        size = len(content)
        if selected and (len(selected) >= max_messages or total + size > max_chars):
            break
        selected.append(item)
        total += size
    return list(reversed(selected))


def add_fact(text):
    if not text:
        return
    facts = campaign_state.setdefault("facts", [])
    text = text.strip()[:500]
    if text and text not in facts:
        facts.append(text)
        del facts[:-50]


def compact_history_if_needed():
    """No elimina memoria arbitrariamente: conserva un resumen y estado estructurado."""
    global chat_history, long_term_summary
    if len(chat_history) <= MAX_HISTORY_MESSAGES:
        return

    old = chat_history[:-MAX_HISTORY_MESSAGES]
    snippets = []
    for msg in old:
        content = str(msg.get("content", "")).strip()
        if content:
            role = msg.get("role", "")
            snippets.append(f"{role}: {content[:300]}")

    if snippets:
        long_term_summary += "\n" + "\n".join(snippets[-20:])
        long_term_summary = long_term_summary[-MAX_SUMMARY_CHARS:]

    chat_history = chat_history[-MAX_HISTORY_MESSAGES:]


def persist():
    # SQLite recibe el historial antes de compactar la copia activa.
    try:
        archive_history_events(chat_history)
        save_state_snapshot()
        save_important_fact_memory()
    except Exception as exc:
        print(f"[SQLite] No se pudo persistir: {exc}")
    compact_history_if_needed()
    save_data(long_term_summary, chat_history, campaign_state)

# ============================================================
# D&D: MOTOR DE DADOS + OPCIONAL BOT EXTERNO
# ============================================================

def validate_dice_expression(expression):
    expression = expression.lower().replace(" ", "")
    if not re.fullmatch(r"(?:\d+)?d\d+(?:kh\d+|kl\d+)?(?:[+-]\d+)?", expression):
        raise ValueError(f"Expresión de dados no permitida: {expression}")
    m = re.fullmatch(r"(\d*)d(\d+)(?:(kh|kl)(\d+))?([+-]\d+)?", expression)
    count, sides = int(m.group(1) or 1), int(m.group(2))
    keep, modifier = int(m.group(4) or 0), int(m.group(5) or 0)
    if not (1 <= count <= 100 and 2 <= sides <= 1000): raise ValueError("Dados fuera de rango")
    if keep and not (1 <= keep <= count): raise ValueError("kh/kl fuera de rango")
    if abs(modifier) > 1000: raise ValueError("Modificador fuera de rango")
    return expression


def extract_dice_result(message):
    """Extrae el total de Dice Golem sin confundirlo con la expresión solicitada."""
    chunks = [getattr(message, "content", "") or ""]

    for e in getattr(message, "embeds", []) or []:
        chunks += [
            str(x)
            for x in (
                getattr(e, "title", None),
                getattr(e, "description", None),
                getattr(getattr(e, "footer", None), "text", None),
            )
            if x
        ]
        for f in getattr(e, "fields", []) or []:
            chunks += [
                str(x)
                for x in (getattr(f, "name", None), getattr(f, "value", None))
                if x
            ]

    raw_text = "\n".join(chunks)
    normalized = raw_text.replace("\u200b", " ").strip()

    patterns = (
        r"(?i)\b(?:total|resultado|result|sum|suma)\s*[:=]\s*(-?\d+)\b",
        r"(?i)\b(?:total|resultado|result|sum|suma)\s+(-?\d+)\b",
        r"(?<!\w)=\s*(-?\d+)\s*(?:$|\n)",
    )

    for pattern in patterns:
        matches = re.findall(pattern, normalized)
        if matches:
            return int(matches[-1]), normalized[:2000]

    nums = re.findall(r"(?<!\w)-?\d+(?!\w)", normalized)
    return (int(nums[-1]), normalized[:2000]) if nums else (None, normalized[:2000])
async def find_dice_golem(guild):
    if not guild:
        return None

    # Primero intentar desde la caché.
    if DICE_GOLEM_ID:
        try:
            member_id = int(DICE_GOLEM_ID)
            member = guild.get_member(member_id)

            if member and member.bot:
                return member

            # Si no está en caché, consultar directamente a Discord.
            member = await guild.fetch_member(member_id)

            if member and member.bot:
                return member

        except (ValueError, TypeError, discord.NotFound, discord.HTTPException):
            pass

    # Fallback por nombre.
    target = DICE_GOLEM_NAME.lower()

    return next(
        (
            member
            for member in guild.members
            if member.bot
            and (
                member.name.lower() == target
                or member.display_name.lower() == target
            )
        ),
        None,
    )


async def roll_with_dice_golem(channel, expression, label=""):
    """
    Envía la tirada a Dice Golem y espera específicamente su respuesta.
    El ID del mensaje de la orden sirve como punto de corte para no aceptar
    una respuesta antigua.
    """
    if not getattr(channel, "guild", None):
        raise RuntimeError("Dice Golem requiere un canal de servidor.")

    expression = validate_dice_expression(expression)
    dg = await find_dice_golem(channel.guild)

    if not dg:
        raise RuntimeError("No encuentro a Dice Golem. Configura DICE_GOLEM_ID.")

    command = f"{dg.mention} {expression}"
    if label:
        command += f" # {label[:120]}"

    sent = await channel.send(
        command,
        allowed_mentions=discord.AllowedMentions(
            users=True,
            roles=False,
            everyone=False,
        ),
    )

    print(f"[DADOS] Orden enviada a Dice Golem: {command}")

    def check(msg):
        if msg.channel.id != channel.id:
            return False
        if msg.author.id != dg.id:
            return False
        if msg.id <= sent.id:
            return False

        total, raw = extract_dice_result(msg)
        if total is None:
            print(
                "[DADOS] Respuesta de Dice Golem sin total reconocible: "
                f"{raw[:500]}"
            )
            return False
        return True

    try:
        response = await bot.wait_for(
            "message",
            timeout=DICE_GOLEM_TIMEOUT,
            check=check,
        )
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Dice Golem no respondió en {DICE_GOLEM_TIMEOUT:.0f}s."
        )

    total, raw = extract_dice_result(response)
    if total is None:
        raise RuntimeError(
            "Dice Golem respondió, pero no pude identificar el resultado."
        )

    print(
        f"[DADOS] Resultado recibido: {expression} => {total} | "
        f"mensaje Dice Golem={response.id} | raw={raw[:500]}"
    )

    return {
        "expression": expression,
        "total": total,
        "source": "Dice Golem",
        "raw": raw,
        "message_id": response.id,
    }



async def roll_attack_d20_with_conditions(
    channel,
    attack_mode="normal",
    label="",
):
    """
    Realiza la tirada de ataque de d20 respetando ventaja o desventaja.

    Dice Golem genera exclusivamente los resultados de los dados.
    Python determina cuál resultado utilizar:
    - normal: un d20.
    - advantage: conserva el mayor de dos d20.
    - disadvantage: conserva el menor de dos d20.
    """
    if attack_mode not in {
        "normal",
        "advantage",
        "disadvantage",
    }:
        return None

    first_roll = await roll_with_dice_golem(
        channel,
        "1d20",
        label,
    )

    if not isinstance(first_roll, dict):
        return None

    first_total = first_roll.get("total")

    try:
        first_total = int(first_total)
    except (TypeError, ValueError):
        return None

    if attack_mode == "normal":
        return {
            "mode": "normal",
            "selected": first_total,
            "rolls": [first_total],
        }

    second_roll = await roll_with_dice_golem(
        channel,
        "1d20",
        label,
    )

    if not isinstance(second_roll, dict):
        return None

    second_total = second_roll.get("total")

    try:
        second_total = int(second_total)
    except (TypeError, ValueError):
        return None

    if attack_mode == "advantage":
        selected = max(first_total, second_total)
    else:
        selected = min(first_total, second_total)

    return {
        "mode": attack_mode,
        "selected": selected,
        "rolls": [
            first_total,
            second_total,
        ],
    }


async def perform_roll(channel, expression, label=""):
    result=await roll_with_dice_golem(channel,expression,label)
    campaign_state["last_roll"]=result
    persist()
    print(f"[DADOS] {expression} => {result['total']} (Dice Golem)")
    return result


# ============================================================
# MEMORIA DE NPC / MISIONES / ESTADO
# ============================================================

def ensure_npc(name):
    npcs = campaign_state.setdefault("npcs", {})
    if name not in npcs:
        npcs[name] = {
            "attitude": "Neutral",
            "goal": "",
            "fear": "",
            "relationships": {},
            "knows": [],
            "secrets": [],
            "last_seen": "",
        }
    return npcs[name]

def skill_modifier(character, skill_name):
    """
    Calcula el modificador total de una habilidad usando dnd_rules.py
    como autoridad mecánica.
    """
    if not isinstance(character, dict):
        return None

    if not isinstance(skill_name, str):
        return None

    normalized_skill = (
        skill_name.strip()
        .lower()
        .translate(str.maketrans("áéíóúü", "aeiouu"))
    )

    skill_idx = next(
        (
            idx
            for idx in (
                "acrobatics",
                "animal-handling",
                "arcana",
                "athletics",
                "deception",
                "history",
                "insight",
                "intimidation",
                "investigation",
                "medicine",
                "nature",
                "perception",
                "performance",
                "persuasion",
                "religion",
                "sleight-of-hand",
                "stealth",
                "survival",
            )
            if get_skill_name(idx).strip().lower().translate(
                str.maketrans("áéíóúü", "aeiouu")
            ) == normalized_skill
        ),
        None,
    )

    if skill_idx is None:
        return None

    abilities = character.get("abilities", {})
    if not isinstance(abilities, dict):
        return None

    ability_scores = {
        "str": abilities.get("strength"),
        "dex": abilities.get("dexterity"),
        "con": abilities.get("constitution"),
        "int": abilities.get("intelligence"),
        "wis": abilities.get("wisdom"),
        "cha": abilities.get("charisma"),
    }

    identity = character.get("identity", {})
    if not isinstance(identity, dict):
        return None

    class_idx = identity.get("class")
    level = identity.get("level")

    if not class_idx or level is None:
        return None

    proficiency_data = character.get("proficiency", {})
    skills = (
        proficiency_data.get("skills", [])
        if isinstance(proficiency_data, dict)
        else []
    )

    normalized_skills = {
        str(skill).strip().lower().translate(
            str.maketrans("áéíóúü", "aeiouu")
        )
        for skill in skills
    }

    proficient_skills = []

    for idx in (
        "acrobatics",
        "animal-handling",
        "arcana",
        "athletics",
        "deception",
        "history",
        "insight",
        "intimidation",
        "investigation",
        "medicine",
        "nature",
        "perception",
        "performance",
        "persuasion",
        "religion",
        "sleight-of-hand",
        "stealth",
        "survival",
    ):
        translated = get_skill_name(idx).strip().lower().translate(
            str.maketrans("áéíóúü", "aeiouu")
        )
        if translated in normalized_skills or idx in normalized_skills:
            proficient_skills.append(idx)

    return rules_skill_modifier(
        skill_idx,
        ability_scores,
        class_idx,
        level,
        proficient_skills,
    )



def resolve_saving_throw_advantage_disadvantage(
    conditions,
    ability_idx,
):
    """
    Determina el modo de una tirada de salvación según las condiciones.

    Devuelve:
    - advantage: True/False
    - disadvantage: True/False
    - automatic_failure: True/False
    - mode: normal, advantage, disadvantage o automatic_failure

    Las condiciones y sus efectos provienen exclusivamente de Magical20.
    """
    if not isinstance(conditions, list):
        conditions = []

    if not isinstance(ability_idx, str):
        return None

    ability_idx = ability_idx.strip().lower()

    if ability_idx not in {
        "str",
        "dex",
        "con",
        "int",
        "wis",
        "cha",
    }:
        return None

    advantage = False
    disadvantage = False
    automatic_failure = False

    for condition in conditions:
        if isinstance(condition, str):
            condition_idx = condition.strip().lower()
        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")
            if isinstance(condition_idx, str):
                condition_idx = condition_idx.strip().lower()
            else:
                condition_idx = ""
        else:
            continue

        if not condition_idx:
            continue

        condition_data = get_condition_effects(condition_idx)

        if not isinstance(condition_data, dict):
            continue

        effects = condition_data.get("effects")

        if not isinstance(effects, dict):
            continue

        if (
            ability_idx == "str"
            and effects.get("strength_save_auto_fail") is True
        ):
            automatic_failure = True

        if (
            ability_idx == "dex"
            and effects.get("dexterity_save_auto_fail") is True
        ):
            automatic_failure = True

        if (
            ability_idx == "dex"
            and effects.get("dexterity_save_disadvantage") is True
        ):
            disadvantage = True

    if automatic_failure:
        mode = "automatic_failure"
    elif advantage and disadvantage:
        mode = "normal"
    elif advantage:
        mode = "advantage"
    elif disadvantage:
        mode = "disadvantage"
    else:
        mode = "normal"

    return {
        "advantage": advantage,
        "disadvantage": disadvantage,
        "automatic_failure": automatic_failure,
        "mode": mode,
    }

def resolve_saving_throw(character, ability_idx, dc):
    """
    Resuelve la parte mecánica de una tirada de salvación.

    Python determina:
    - modificador de la característica;
    - competencia de la clase;
    - bonificador total de la salvación;
    - d20 requerido.

    Dice Golem se encargará posteriormente del d20 real.
    """
    if not isinstance(character, dict):
        return None

    if not isinstance(ability_idx, str):
        return None

    ability_idx = ability_idx.strip().lower()

    if ability_idx not in {
        "str",
        "dex",
        "con",
        "int",
        "wis",
        "cha",
    }:
        return None

    try:
        dc = int(dc)
    except (TypeError, ValueError):
        return None

    identity = character.get("identity", {})
    abilities = character.get("abilities", {})

    if not isinstance(identity, dict):
        return None

    if not isinstance(abilities, dict):
        return None

    ability_map = {
        "str": "strength",
        "dex": "dexterity",
        "con": "constitution",
        "int": "intelligence",
        "wis": "wisdom",
        "cha": "charisma",
    }

    ability_name = ability_map[ability_idx]
    score = abilities.get(ability_name)

    modifier = get_ability_modifier(score)

    if modifier is None:
        return None

    combat = character.get("combat")

    if not isinstance(combat, dict):
        combat = {}

    conditions = combat.get("conditions")

    if not isinstance(conditions, list):
        conditions = []

    condition_keys = set()

    for condition in conditions:
        if isinstance(condition, str):
            key = condition.strip().lower()
            if key:
                condition_keys.add(key)

        elif isinstance(condition, dict):
            idx = condition.get("idx")
            if isinstance(idx, str) and idx.strip():
                condition_keys.add(idx.strip().lower())

    save_mode = resolve_saving_throw_advantage_disadvantage(
        conditions,
        ability_idx,
    )

    if not isinstance(save_mode, dict):
        return None

    automatic_failure = save_mode.get(
        "automatic_failure",
        False,
    )

    class_idx = identity.get("class")
    level = identity.get("level")

    if not isinstance(class_idx, str) or not class_idx.strip():
        return None

    proficiency_bonus = get_proficiency_bonus(
        class_idx,
        level,
    )

    if proficiency_bonus is None:
        return None

    saving_throws = get_class_saving_throws(class_idx)

    if not isinstance(saving_throws, list):
        saving_throws = []

    proficient = ability_idx in {
        str(save).strip().lower()
        for save in saving_throws
        if isinstance(save, str)
    }

    total_modifier = modifier

    if proficient:
        total_modifier += proficiency_bonus

    return {
        "ability": ability_idx,
        "dc": dc,
        "modifier": total_modifier,
        "ability_modifier": modifier,
        "proficiency_bonus": proficiency_bonus,
        "proficient": proficient,
        "automatic_failure": automatic_failure,
        "advantage": save_mode.get("advantage", False),
        "disadvantage": save_mode.get("disadvantage", False),
        "mode": save_mode.get("mode", "normal"),
        "dice_expression": "1d20",
    }


def apply_saving_throw_result(save, d20_result):
    """
    Completa una tirada de salvación usando el d20 real.

    No genera azar.
    Python calcula exclusivamente el total y el éxito/fallo.
    """
    if not isinstance(save, dict):
        return None

    automatic_failure = save.get(
        "automatic_failure",
        False,
    )

    if automatic_failure:
        return {
            **save,
            "d20": None,
            "total": None,
            "success": False,
        }

    try:
        d20_result = int(d20_result)
    except (TypeError, ValueError):
        return None

    if d20_result < 1 or d20_result > 20:
        return None

    try:
        dc = int(save.get("dc"))
        modifier = int(save.get("modifier"))
    except (TypeError, ValueError):
        return None

    total = d20_result + modifier
    success = total >= dc

    return {
        **save,
        "d20": d20_result,
        "total": total,
        "success": success,
    }

def resolve_skill_check(character, skill_name, dc):
    """
    Resuelve la parte mecánica de una prueba de habilidad D&D 5e.

    Python determina:
    - modificador de la habilidad
    - d20 requerido
    - total
    - éxito o fallo

    La tirada real se añadirá después mediante Dice Golem.
    """
    if not isinstance(character, dict):
        return None

    try:
        dc = int(dc)
    except (TypeError, ValueError):
        return None

    modifier = skill_modifier(character, skill_name)

    if modifier is None:
        return None

    return {
        "skill": str(skill_name).strip(),
        "dc": dc,
        "modifier": modifier,
        "dice_expression": "1d20",
    }


def apply_skill_check_result(check, d20_result):
    """
    Completa una prueba de habilidad usando el resultado real del d20.
    No genera azar; únicamente aplica las reglas mecánicas.
    """
    if not isinstance(check, dict):
        return None

    try:
        d20_result = int(d20_result)
    except (TypeError, ValueError):
        return None

    if not 1 <= d20_result <= 20:
        return None

    try:
        dc = int(check["dc"])
        modifier = int(check["modifier"])
    except (KeyError, TypeError, ValueError):
        return None

    total = d20_result + modifier

    # Regla básica de pruebas de habilidad:
    # éxito si el total alcanza o supera la CD.
    success = total >= dc

    return {
        **check,
        "d20": d20_result,
        "total": total,
        "success": success,
    }



async def resolve_monster_attack_with_dice_golem(
    monster_idx,
    action_name,
    attacker_combatant_id,
    target_combatant_id,
    channel,
):
    """
    Resuelve un ataque individual de un monstruo usando Dice Golem.

    Python:
    - obtiene el ataque desde Magical20;
    - determina el resultado del ataque;
    - aplica las reglas de combate;
    - resuelve los modificadores de daño.

    Dice Golem:
    - tira exclusivamente los dados.
    """

    attack = get_monster_attack_data(
        monster_idx,
        action_name,
    )

    if attack is None:
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    attacker = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == attacker_combatant_id:
            attacker = combatant
            break

    if attacker is None:
        return None

    if attacker.get("type") != "monster":
        return None

    if not is_combatant_turn(attacker_combatant_id):
        return None

    attacker_conditions = get_combatant_conditions(
        attacker_combatant_id
    )

    if attacker_conditions is None:
        return None

    if not can_take_action(attacker_conditions):
        return None

    target = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == target_combatant_id:
            target = combatant
            break

    if target is None:
        return None

    if target.get("type") != "character":
        return None

    if not can_attack_target(
        attacker_combatant_id,
        attacker_conditions,
        target_combatant_id,
    ):
        return None

    character_name = target.get("character_name")

    if not isinstance(character_name, str):
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    character_combat = character.get("combat")

    if not isinstance(character_combat, dict):
        return None

    try:
        target_ac = int(character_combat.get("ac"))
    except (TypeError, ValueError):
        return None

    attacker_conditions = get_combatant_conditions(
        attacker_combatant_id
    )

    target_conditions = get_combatant_conditions(
        target_combatant_id
    )

    if attacker_conditions is None or target_conditions is None:
        return None

    attack_mode = resolve_attack_advantage_disadvantage(
        attacker_conditions,
        target_conditions,
        distance_ft=None,
    )

    if not isinstance(attack_mode, dict):
        return None

    attack_roll = await roll_attack_d20_with_conditions(
        channel,
        attack_mode.get("mode", "normal"),
        f"{attack['action']} ataque",
    )

    if not isinstance(attack_roll, dict):
        return None

    selected_roll = attack_roll.get("selected")

    try:
        selected_roll = int(selected_roll)
    except (TypeError, ValueError):
        return None

    resolved_attack = resolve_attack_roll(
        selected_roll,
        attack["attack_bonus"],
        target_ac,
    )

    if resolved_attack is None:
        return None

    if resolved_attack["outcome"] == "miss":
        turn_result = advance_turn_after_action()

        return {
            "monster": monster_idx,
            "action": attack["action"],
            "target": target_combatant_id,
            "attack": resolved_attack,
            "damage": 0,
            "damages": [],
            "turn": turn_result,
        }

    damage_entries = attack["damages"]

    damage_roll_results = []

    for damage_entry in damage_entries:
        damage_dice = damage_entry["damage_dice"]

        if resolved_attack["critical"]:
            parts = damage_dice.split("d", 1)

            if len(parts) == 2:
                try:
                    number_of_dice = int(parts[0])
                except (TypeError, ValueError):
                    return None

                if number_of_dice < 1:
                    return None

                damage_dice = f"{number_of_dice * 2}d{parts[1]}"

        damage_result = await roll_with_dice_golem(
            channel,
            damage_dice,
            f"{attack['action']} daño",
        )

        if not damage_result:
            return None

        damage_roll_results.append(
            damage_result["total"]
        )

    target_conditions = get_combatant_conditions(
        target_combatant_id,
    )

    if target_conditions is None:
        return None

    target_damage_resistances = get_condition_damage_resistances(
        target_conditions,
    )

    target_damage_immunities = get_condition_damage_immunities(
        target_conditions,
    )

    resolved_damage = resolve_monster_attack_damage(
        resolved_attack,
        damage_entries,
        damage_roll_results,
        damage_resistances=target_damage_resistances,
        damage_vulnerabilities=[],
        damage_immunities=target_damage_immunities,
    )

    if resolved_damage is None:
        return None

    applied = apply_damage_to_character(
        character,
        resolved_damage["damage"],
    )

    if not isinstance(applied, dict):
        return None

    removed = None

    if applied.get("unconscious") is True:
        removed = remove_dead_combatant(
            target_combatant_id,
        )

        if not isinstance(removed, dict):
            return None

        turn_result = {
            "round": removed.get("round"),
            "turn": removed.get("turn"),
        }
    else:
        turn_result = advance_turn_after_action()

    persist()

    return {
        "monster": monster_idx,
        "action": attack["action"],
        "target": target_combatant_id,
        "character": character_name,
        "attack": resolved_attack,
        "damage": resolved_damage["damage"],
        "damages": resolved_damage["damages"],
        "critical": resolved_damage["critical"],
        "applied": applied,
        "removed": removed,
        "turn": turn_result,
    }



async def resolve_saving_throw_with_dice_golem(
    character,
    ability_idx,
    dc,
    channel,
):
    """
    Resuelve una tirada de salvación completa usando Dice Golem.

    Python determina:
    - modificador;
    - competencia;
    - condiciones;
    - ventaja/desventaja;
    - fallo automático;
    - éxito o fallo final.

    Dice Golem determina exclusivamente los d20.
    """
    save = resolve_saving_throw(
        character,
        ability_idx,
        dc,
    )

    if not isinstance(save, dict):
        return None

    if save.get("automatic_failure") is True:
        return apply_saving_throw_result(
            save,
            None,
        )

    mode = save.get("mode", "normal")

    result = await roll_attack_d20_with_conditions(
        channel,
        mode,
        f"Salvación {save['ability']} CD {save['dc']}",
    )

    if not isinstance(result, dict):
        return None

    selected = result.get("selected")

    try:
        selected = int(selected)
    except (TypeError, ValueError):
        return None

    return apply_saving_throw_result(
        save,
        selected,
    )

async def resolve_skill_check_with_dice_golem(
    character,
    skill_name,
    dc,
    channel,
):
    """
    Resuelve una prueba de habilidad completa usando Dice Golem.

    Python determina el modificador y la ventaja/desventaja.
    Dice Golem determina exclusivamente los d20.
    Python selecciona el resultado y calcula el total y el éxito/fallo.
    """
    check = resolve_skill_check(character, skill_name, dc)

    if check is None:
        return None

    combat = character.get("combat")

    if not isinstance(combat, dict):
        combat = {}

    conditions = combat.get("conditions")

    if not isinstance(conditions, list):
        conditions = []

    check_mode = resolve_skill_check_advantage_disadvantage(
        conditions,
    )

    if not isinstance(check_mode, dict):
        return None

    mode = check_mode.get("mode", "normal")

    result = await roll_attack_d20_with_conditions(
        channel,
        mode,
        f"{check['skill']} CD {check['dc']}",
    )

    if not isinstance(result, dict):
        return None

    selected = result.get("selected")

    try:
        selected = int(selected)
    except (TypeError, ValueError):
        return None

    return apply_skill_check_result(
        check,
        selected,
    )



def parse_save_marker(text):
    """
    Extrae una solicitud de tirada de salvación emitida por el LLM.

    Formato:
    <<<SAVE:Fuerza|15>>>

    Devuelve (característica, CD) o None si no hay una solicitud válida.
    """
    if not text:
        return None

    match = re.search(
        r"<<<SAVE:\s*([^|>]+)\|\s*(\d+)\s*>>>",
        str(text),
        re.IGNORECASE,
    )

    if not match:
        return None

    ability_name = match.group(1).strip()

    try:
        dc = int(match.group(2))
    except (TypeError, ValueError):
        return None

    if not ability_name or dc < 0:
        return None

    return ability_name, dc


def parse_check_marker(text):
    """
    Extrae una solicitud de prueba emitida por el LLM.

    Formato:
    <<<CHECK:Percepción|15>>>

    Devuelve (habilidad, CD) o None si no hay una solicitud válida.
    """
    if not text:
        return None

    match = re.search(
        r"<<<CHECK:\s*([^|>]+)\|\s*(\d+)\s*>>>",
        str(text),
        re.IGNORECASE,
    )

    if not match:
        return None

    skill_name = match.group(1).strip()

    try:
        dc = int(match.group(2))
    except (TypeError, ValueError):
        return None

    if not skill_name or dc < 0:
        return None

    return skill_name, dc



def parse_attack_marker(text):
    """
    Extrae una solicitud de ataque emitida por el LLM.

    Formato:
    <<<ATTACK:Espada larga|Goblin>>>

    Devuelve (arma, objetivo) o None si no hay una solicitud válida.
    """
    if not text:
        return None

    match = re.search(
        r"<<<ATTACK:\s*([^|>]+)\|\s*([^>]+?)\s*>>>",
        str(text),
        re.IGNORECASE,
    )

    if not match:
        return None

    weapon_name = match.group(1).strip()
    target_name = match.group(2).strip()

    if not weapon_name or not target_name:
        return None

    return weapon_name, target_name


def get_character_for_player(user_id):
    """
    Obtiene el personaje asociado al jugador mediante player_bindings.

    La asociación pertenece al estado de la campaña actual.
    No inventa personajes ni asociaciones.
    """
    bindings = campaign_state.get("player_bindings", {})
    characters = campaign_state.get("characters", {})

    if not isinstance(bindings, dict):
        return None

    if not isinstance(characters, dict):
        return None

    character_name = bindings.get(str(user_id))

    if not character_name:
        return None

    character = characters.get(character_name)

    if not isinstance(character, dict):
        return None

    return character



def normalize_attack_name(value):
    """
    Normaliza texto utilizado para identificar armas.
    """
    if not isinstance(value, str):
        return ""

    value = value.strip().lower()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    return re.sub(r"\s+", " ", value)


def get_attack_weapon_aliases():
    """
    Nombres visibles aceptados para armas.
    El idx interno sigue siendo la autoridad mecánica.
    """
    return {
        "battleaxe": ["hacha de batalla", "battleaxe"],
        "club": ["garrote", "club"],
        "dagger": ["daga", "dagger"],
        "flail": ["mangual", "flail"],
        "glaive": ["glaive"],
        "greataxe": ["hacha a dos manos", "greataxe"],
        "greatclub": ["garrote grande", "greatclub"],
        "greatsword": ["espadon", "greatsword"],
        "halberd": ["alabarda", "halberd"],
        "handaxe": ["hacha de mano", "handaxe"],
        "javelin": ["jabalina", "javelin"],
        "lance": ["lanza de caballeria", "lance"],
        "light-hammer": ["martillo ligero", "light hammer", "light-hammer"],
        "longsword": ["espada larga", "longsword"],
        "mace": ["maza", "mace"],
        "maul": ["maza a dos manos", "maul"],
        "morningstar": ["lucero del alba", "morningstar"],
        "pike": ["pica", "pike"],
        "quarterstaff": ["baston", "quarterstaff"],
        "rapier": ["estoque", "rapier"],
        "scimitar": ["cimitarra", "scimitar"],
        "shortsword": ["espada corta", "shortsword"],
        "sickle": ["hoz", "sickle"],
        "spear": ["lanza", "spear"],
        "trident": ["tridente", "trident"],
        "war-pick": ["pico de guerra", "war pick", "war-pick"],
        "warhammer": ["martillo de guerra", "warhammer"],
        "whip": ["latigo", "whip"],
        "blowgun": ["cerbatana", "blowgun"],
        "crossbow-hand": ["ballesta de mano", "hand crossbow", "crossbow-hand"],
        "crossbow-heavy": ["ballesta pesada", "heavy crossbow", "crossbow-heavy"],
        "crossbow-light": ["ballesta ligera", "light crossbow", "crossbow-light"],
        "longbow": ["arco largo", "longbow"],
        "shortbow": ["arco corto", "shortbow"],
        "net": ["red", "net"],
        "sling": ["honda", "sling"],
    }


def resolve_weapon_for_attack(character_name, weapon_name):
    """
    Resuelve el nombre declarado por el jugador a un weapon_idx real.

    El arma debe existir en el inventario del personaje.
    """
    character = get_character_by_name(character_name)

    if character is None:
        return None

    inventory = character.get("inventory", [])

    if not isinstance(inventory, list):
        return None

    requested = normalize_attack_name(weapon_name)

    if not requested:
        return None

    aliases = get_attack_weapon_aliases()

    for entry in inventory:
        if isinstance(entry, str):
            idx = entry.strip()
            name = entry.strip()
            quantity = 1
        elif isinstance(entry, dict):
            idx = entry.get("idx")
            name = entry.get("name")
            quantity = entry.get("quantity", 0)
        else:
            continue

        if not isinstance(idx, str):
            continue

        idx = idx.strip()

        if not idx:
            continue

        if not isinstance(quantity, int) or quantity <= 0:
            continue

        normalized_idx = normalize_attack_name(idx)
        normalized_name = normalize_attack_name(name)

        candidates = {
            normalized_idx,
            normalized_name,
        }

        for alias in aliases.get(normalized_idx, []):
            candidates.add(normalize_attack_name(alias))

        if requested in candidates:
            weapon_data = get_weapon_data(idx)

            if not isinstance(weapon_data, dict):
                continue

            return idx

    return None



def resolve_attack_target(target_name):
    """
    Resuelve el nombre declarado por el jugador contra un único monstruo
    actualmente presente en combate.

    combat["combatants"] es una lista.
    Si hay cero o más de un objetivo compatible, no elige arbitrariamente.
    """
    if not isinstance(target_name, str):
        return None

    requested = normalize_attack_name(target_name)

    if not requested:
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    if not combat.get("active"):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    matches = []

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("type") != "monster":
            continue

        monster_idx = combatant.get("monster_idx")

        if monster_idx is None:
            continue

        monster_data = get_monster_combat_data(monster_idx)

        if not isinstance(monster_data, dict):
            continue

        possible_names = {
            normalize_attack_name(str(monster_idx)),
            normalize_attack_name(str(monster_data.get("name", ""))),
            normalize_attack_name(str(combatant.get("name", ""))),
        }

        if requested in possible_names:
            combatant_id = combatant.get("id")

            if isinstance(combatant_id, str) and combatant_id:
                matches.append(combatant_id)

    if len(matches) != 1:
        return None

    return matches[0]


async def resolve_character_attack_with_dice_golem(
    character_name,
    weapon_idx,
    target_combatant_id,
    channel,
):
    """
    Resuelve un ataque de un personaje contra una instancia de monstruo.

    Autoridad:
    - Magical20/dnd_rules.py: reglas y datos mecánicos.
    - Dice Golem: resultados aleatorios.
    - bot_master.py: aplicación del daño y persistencia.

    No depende del LLM.
    """

    global campaign_state

    if not isinstance(character_name, str):
        return None

    if not isinstance(weapon_idx, str):
        return None

    if not isinstance(target_combatant_id, str):
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    identity = character.get("identity")
    abilities = character.get("abilities")

    if not isinstance(identity, dict):
        return None

    if not isinstance(abilities, dict):
        return None

    class_idx = identity.get("class")
    level = identity.get("level")
    race_idx = identity.get("race") or None
    subrace_idx = identity.get("subclass") or None

    if not isinstance(class_idx, str):
        return None

    if not class_idx.strip():
        return None

    try:
        level = int(level)
    except (TypeError, ValueError):
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    attacker_combatant_id = f"character:{character_name}"

    if not is_combatant_turn(attacker_combatant_id):
        return None

    attacker_conditions = get_combatant_conditions(
        attacker_combatant_id
    )

    if attacker_conditions is None:
        return None

    if not can_take_action(attacker_conditions):
        return None

    target = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == target_combatant_id:
            target = combatant
            break

    if target is None:
        return None

    if target.get("type") != "monster":
        return None

    if not can_attack_target(
        attacker_combatant_id,
        attacker_conditions,
        target_combatant_id,
    ):
        return None

    if not has_item_in_inventory(character_name, weapon_idx):
        return None

    monster_idx = target.get("monster_idx")

    if not isinstance(monster_idx, str):
        return None

    monster_data = get_monster_combat_data(monster_idx)

    if not isinstance(monster_data, dict):
        return None

    target_ac = monster_data.get("ac")

    try:
        target_ac = int(target_ac)
    except (TypeError, ValueError):
        return None

    attack_data = resolve_weapon_attack(
        weapon_idx,
        abilities,
        class_idx,
        level,
        race_idx,
        subrace_idx,
    )

    if not isinstance(attack_data, dict):
        return None

    attack_modifier = attack_data.get("attack_modifier")

    try:
        attack_modifier = int(attack_modifier)
    except (TypeError, ValueError):
        return None

    ammunition_idx = get_ammunition_for_weapon(weapon_idx)

    if ammunition_idx is not None:
        if not has_item_in_inventory(character_name, ammunition_idx):
            return None

    attacker_conditions = get_combatant_conditions(
        attacker_combatant_id
    )

    target_conditions = get_combatant_conditions(
        target_combatant_id
    )

    if attacker_conditions is None or target_conditions is None:
        return None

    attack_mode = resolve_attack_advantage_disadvantage(
        attacker_conditions,
        target_conditions,
        distance_ft=None,
    )

    if not isinstance(attack_mode, dict):
        return None

    attack_roll = await roll_attack_d20_with_conditions(
        channel,
        attack_mode.get("mode", "normal"),
        f"{character_name} ataque con {weapon_idx}",
    )

    if not isinstance(attack_roll, dict):
        return None

    d20_result = attack_roll.get("selected")

    try:
        d20_result = int(d20_result)
    except (TypeError, ValueError):
        return None

    attack_result = resolve_attack_roll(
        d20_result,
        attack_modifier,
        target_ac,
    )

    if not isinstance(attack_result, dict):
        return None

    if ammunition_idx is not None:
        consumed = remove_item_from_character(
            character_name,
            ammunition_idx,
            1,
        )

        if not isinstance(consumed, dict):
            return None

    outcome = attack_result.get("outcome")

    if outcome == "miss":
        turn_result = advance_turn_after_action()

        return {
            "character": character_name,
            "weapon": weapon_idx,
            "target": target_combatant_id,
            "monster": monster_idx,
            "attack": attack_result,
            "damage": None,
            "turn": turn_result,
        }

    damage_dice = get_weapon_damage_dice(
        weapon_idx,
        critical=outcome == "critical",
    )

    if not isinstance(damage_dice, str):
        return None

    damage_roll = await roll_with_dice_golem(
        channel,
        damage_dice,
        f"{character_name} daño con {weapon_idx}",
    )

    if not isinstance(damage_roll, dict):
        return None

    damage_roll_result = damage_roll.get("total")

    try:
        damage_roll_result = int(damage_roll_result)
    except (TypeError, ValueError):
        return None

    target_conditions = get_combatant_conditions(
        target_combatant_id,
    )

    if target_conditions is None:
        return None

    condition_resistances = get_condition_damage_resistances(
        target_conditions,
    )

    monster_resistances = monster_data.get("damage_resistances") or []

    combined_resistances = list(
        set(monster_resistances) | set(condition_resistances)
    )

    condition_immunities = get_condition_damage_immunities(
        target_conditions,
    )

    monster_immunities = monster_data.get("damage_immunities") or []

    combined_immunities = list(
        set(monster_immunities) | set(condition_immunities)
    )

    damage_result = resolve_attack_damage(
        weapon_idx,
        abilities,
        attack_result,
        damage_roll_result,
        damage_resistances=combined_resistances,
        damage_vulnerabilities=monster_data.get("damage_vulnerabilities"),
        damage_immunities=combined_immunities,
    )

    if not isinstance(damage_result, dict):
        return None

    applied = apply_damage_to_monster(
        target_combatant_id,
        damage_result.get("damage"),
    )

    if not isinstance(applied, dict):
        return None

    removed = None

    if applied.get("dead") is True:
        removed = remove_dead_combatant(
            target_combatant_id
        )

        if not isinstance(removed, dict):
            return None

        turn_result = {
            "round": removed.get("round"),
            "turn": removed.get("turn"),
        }
    else:
        turn_result = advance_turn_after_action()

    persist()

    return {
        "character": character_name,
        "weapon": weapon_idx,
        "target": target_combatant_id,
        "monster": monster_idx,
        "attack": attack_result,
        "damage_roll": damage_roll_result,
        "damage": damage_result,
        "applied": applied,
        "removed": removed,
    }


def advance_turn_after_action():
    """
    Avanza el turno después de una acción de combate válida.

    No hace nada si el combate terminó o no existe un turno válido.
    """
    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    if combat.get("active") is not True:
        return None

    return advance_combat_turn()


def remove_dead_combatant(combatant_id):
    """
    Retira un combatiente muerto del combate actual.

    Personajes:
        Se considera muerto cuando hp_current == 0.

    Monstruos:
        Se considera muerto cuando su hp_current en monster_state == 0.

    Si el combatiente era el turno actual, el turno pasa al siguiente
    combatiente válido.

    Si no quedan combatientes, el combate termina automáticamente.

    No persiste el estado.
    """

    global campaign_state

    if not isinstance(combatant_id, str):
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    target = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == combatant_id:
            target = combatant
            break

    if target is None:
        return None

    combatant_type = target.get("type")

    if combatant_type == "character":
        character_name = target.get("character_name")
        character = get_character_by_name(character_name)

        if character is None:
            return None

        character_combat = character.get("combat")

        if not isinstance(character_combat, dict):
            return None

        try:
            hp_current = int(character_combat.get("hp_current"))
        except (TypeError, ValueError):
            return None

    elif combatant_type == "monster":
        monster_state = combat.get("monster_state")

        if not isinstance(monster_state, dict):
            return None

        state = monster_state.get(combatant_id)

        if not isinstance(state, dict):
            return None

        try:
            hp_current = int(state.get("hp_current"))
        except (TypeError, ValueError):
            return None

    else:
        return None

    if hp_current != 0:
        return None

    initiative = combat.get("initiative")

    if not isinstance(initiative, list):
        initiative = []

    current_turn = combat.get("turn")
    was_current_turn = current_turn == combatant_id

    combat["combatants"] = [
        combatant
        for combatant in combatants
        if isinstance(combatant, dict)
        and combatant.get("id") != combatant_id
    ]

    combat["initiative"] = [
        combatant_id_value
        for combatant_id_value in initiative
        if combatant_id_value != combatant_id
    ]

    if combatant_type == "monster":
        monster_state = combat.get("monster_state")

        if isinstance(monster_state, dict):
            monster_state.pop(combatant_id, None)

    remaining = combat["initiative"]

    if not remaining:
        combat["active"] = False
        combat["round"] = 0
        combat["turn"] = None

        return {
            "removed": combatant_id,
            "combat_ended": True,
            "active": False,
            "round": 0,
            "turn": None,
        }

    if was_current_turn:
        try:
            removed_index = initiative.index(combatant_id)
        except ValueError:
            removed_index = -1

        if removed_index >= 0:
            next_index = removed_index % len(initiative)

            while next_index < len(initiative):
                candidate = initiative[next_index]

                if candidate in remaining:
                    combat["turn"] = candidate
                    break

                next_index += 1

            else:
                combat["turn"] = remaining[0]
        else:
            combat["turn"] = remaining[0]

        return {
            "removed": combatant_id,
            "combat_ended": False,
            "active": combat.get("active") is True,
            "round": combat.get("round"),
            "turn": combat["turn"],
        }

    return {
        "removed": combatant_id,
        "combat_ended": False,
        "active": combat.get("active") is True,
        "round": combat.get("round"),
        "turn": combat.get("turn"),
    }


def get_character_by_name(character_name):
    """
    Obtiene un personaje de la campaña actual por su nombre.
    """
    characters = campaign_state.get("characters", {})

    if not isinstance(characters, dict):
        return None

    if not isinstance(character_name, str):
        return None

    character_name = character_name.strip()

    if not character_name:
        return None

    character = characters.get(character_name)

    if not isinstance(character, dict):
        return None

    return character


def apply_damage_to_monster(combatant_id, damage):
    """
    Aplica daño a una instancia de monstruo del combate actual.

    Modifica únicamente el HP dinámico almacenado en monster_state.
    No persiste el estado ni depende del LLM o Dice Golem.

    Devuelve un resumen del cambio o None si los datos no son válidos.
    """
    global campaign_state

    if not isinstance(combatant_id, str):
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    monster_state = combat.get("monster_state")

    if not isinstance(monster_state, dict):
        return None

    state = monster_state.get(combatant_id)

    if not isinstance(state, dict):
        return None

    try:
        damage = int(damage)
        hp_current = int(state.get("hp_current"))
        hp_max = int(state.get("hp_max"))
    except (TypeError, ValueError):
        return None

    if damage < 0:
        return None

    if hp_max < 0 or hp_current < 0:
        return None

    if hp_current > hp_max:
        hp_current = hp_max

    hp_before = hp_current
    hp_after = max(0, hp_current - damage)

    state["hp_current"] = hp_after
    state["hp_max"] = hp_max

    return {
        "combatant_id": combatant_id,
        "damage": damage,
        "hp_before": hp_before,
        "hp_after": hp_after,
        "hp_max": hp_max,
        "dead": hp_after == 0,
    }



def get_condition_data(condition_idx):
    """
    Devuelve los datos de una condición usando exclusivamente
    el registro correspondiente de Magical20.
    """
    if not isinstance(condition_idx, str):
        return None

    condition_idx = condition_idx.strip().lower()

    if not condition_idx:
        return None

    condition = get_condition(condition_idx)

    if not isinstance(condition, dict):
        return None

    name = condition.get("name")

    if not isinstance(name, str) or not name.strip():
        return None

    return {
        "idx": condition_idx,
        "name": name.strip(),
    }



def get_condition_source(conditions, condition_idx):
    """
    Devuelve el combatiente que originó una condición.

    Las condiciones antiguas almacenadas como strings no tienen origen.
    Las condiciones modernas pueden almacenarse como:
    {"idx": "charmed", "source": "character:Hechicero"}
    """
    if not isinstance(conditions, list):
        return None

    if not isinstance(condition_idx, str):
        return None

    condition_key = condition_idx.strip().lower()

    if not condition_key:
        return None

    for condition in conditions:
        if isinstance(condition, str):
            existing_idx = condition.strip().lower()

            if existing_idx == condition_key:
                return None

        elif isinstance(condition, dict):
            existing_idx = condition.get("idx")

            if (
                isinstance(existing_idx, str)
                and existing_idx.strip().lower() == condition_key
            ):
                source = condition.get("source")

                if isinstance(source, str) and source.strip():
                    return source.strip()

                return None

    return None


def can_attack_target(attacker_combatant_id, attacker_conditions, target_combatant_id):
    """
    Determina si un combatiente puede atacar a un objetivo concreto
    según sus condiciones actuales.

    Charmed:
    El objetivo encantado no puede atacar al combatiente que originó
    la condición.
    """
    if not isinstance(attacker_combatant_id, str):
        return False

    if not isinstance(target_combatant_id, str):
        return False

    if not isinstance(attacker_conditions, list):
        return False

    for condition in attacker_conditions:
        condition_idx = None

        if isinstance(condition, str):
            condition_idx = condition.strip().lower()

        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")

            if isinstance(condition_idx, str):
                condition_idx = condition_idx.strip().lower()

        if condition_idx != "charmed":
            continue

        source = get_condition_source(
            [condition],
            "charmed",
        )

        if source == target_combatant_id:
            return False

    return True


def apply_condition_to_monster(combatant_id, condition_idx, source_combatant_id=None):
    """
    Aplica una condición válida a una instancia de monstruo.

    La condición se valida exclusivamente contra Magical20.
    No permite duplicados y persiste el cambio.
    """
    if not isinstance(combatant_id, str):
        return None

    combatant_id = combatant_id.strip()

    if not combatant_id:
        return None

    if not combatant_id.startswith("monster:"):
        return None

    condition_data = get_condition_data(condition_idx)

    if not isinstance(condition_data, dict):
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    target = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == combatant_id:
            target = combatant
            break

    if target is None:
        return None

    if target.get("type") != "monster":
        return None

    monster_state = combat.get("monster_state")

    if not isinstance(monster_state, dict):
        return None

    state = monster_state.get(combatant_id)

    if not isinstance(state, dict):
        return None

    conditions = state.get("conditions")

    if not isinstance(conditions, list):
        conditions = []
        state["conditions"] = conditions

    condition_key = condition_data["idx"].lower()

    if (
        condition_key == "poisoned"
        and condition_grants_immunity(
            conditions,
            "poison_immunity",
        )
    ):
        return {
            "combatant_id": combatant_id,
            "condition": condition_data,
            "applied": False,
            "immune": True,
            "conditions": conditions,
        }

    for existing in conditions:
        if isinstance(existing, str):
            if existing.strip().lower() == condition_key:
                return {
                    "combatant_id": combatant_id,
                    "condition": condition_data,
                    "applied": False,
                    "already_present": True,
                    "conditions": conditions,
                }

        elif isinstance(existing, dict):
            existing_idx = existing.get("idx")

            if (
                isinstance(existing_idx, str)
                and existing_idx.strip().lower() == condition_key
            ):
                return {
                    "combatant_id": combatant_id,
                    "condition": condition_data,
                    "applied": False,
                    "already_present": True,
                    "conditions": conditions,
                }

    condition_entry = condition_data["idx"]

    if isinstance(source_combatant_id, str) and source_combatant_id.strip():
        condition_entry = {
            "idx": condition_data["idx"],
            "source": source_combatant_id.strip(),
        }

    conditions.append(condition_entry)
    persist()

    return {
        "combatant_id": combatant_id,
        "condition": condition_data,
        "applied": True,
        "already_present": False,
        "conditions": conditions,
    }


def remove_condition_from_monster(combatant_id, condition_idx):
    """
    Elimina una condición de una instancia de monstruo.

    La condición se valida exclusivamente contra Magical20.
    Si no está presente, no modifica el estado.
    """
    if not isinstance(combatant_id, str):
        return None

    combatant_id = combatant_id.strip()

    if not combatant_id:
        return None

    if not combatant_id.startswith("monster:"):
        return None

    condition_data = get_condition_data(condition_idx)

    if not isinstance(condition_data, dict):
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    target = None

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        if combatant.get("id") == combatant_id:
            target = combatant
            break

    if target is None:
        return None

    if target.get("type") != "monster":
        return None

    monster_state = combat.get("monster_state")

    if not isinstance(monster_state, dict):
        return None

    state = monster_state.get(combatant_id)

    if not isinstance(state, dict):
        return None

    conditions = state.get("conditions")

    if not isinstance(conditions, list):
        return None

    condition_key = condition_data["idx"].lower()

    for index, existing in enumerate(conditions):
        existing_idx = None

        if isinstance(existing, str):
            existing_idx = existing.strip()

        elif isinstance(existing, dict):
            existing_idx = existing.get("idx")

        if (
            isinstance(existing_idx, str)
            and existing_idx.lower() == condition_key
        ):
            conditions.pop(index)
            persist()

            return {
                "combatant_id": combatant_id,
                "condition": condition_data,
                "removed": True,
                "conditions": conditions,
            }

    return {
        "combatant_id": combatant_id,
        "condition": condition_data,
        "removed": False,
        "conditions": conditions,
    }

def has_item_in_inventory(character_name, item):
    """
    Comprueba si un personaje posee un objeto en su inventario.

    Acepta tanto entradas antiguas de texto como objetos con cantidad.
    En el formato moderno compara tanto idx como nombre visible.
    La comparación ignora mayúsculas/minúsculas.
    """
    if not isinstance(character_name, str):
        return False

    if not isinstance(item, str):
        return False

    item = item.strip()

    if not item:
        return False

    character = get_character_by_name(character_name)

    if character is None:
        return False

    inventory = character.get("inventory", [])

    if not isinstance(inventory, list):
        return False

    item_normalized = item.lower()

    for entry in inventory:
        if isinstance(entry, str):
            if entry.strip().lower() == item_normalized:
                return True

        elif isinstance(entry, dict):
            quantity = entry.get("quantity", 0)

            if not isinstance(quantity, int) or quantity <= 0:
                continue

            idx = entry.get("idx")
            name = entry.get("name")

            if (
                isinstance(idx, str)
                and idx.strip().lower() == item_normalized
            ):
                return True

            if (
                isinstance(name, str)
                and name.strip().lower() == item_normalized
            ):
                return True

    return False


def get_ammunition_for_weapon(weapon_idx):
    """
    Devuelve el idx de la munición requerida por un arma.

    La propiedad 'ammunition' se determina exclusivamente
    mediante los datos mecánicos de Magical20.
    """
    if not isinstance(weapon_idx, str):
        return None

    weapon_idx = weapon_idx.strip().lower()

    if not weapon_idx:
        return None

    weapon_data = get_weapon_data(weapon_idx)

    if not isinstance(weapon_data, dict):
        return None

    properties = weapon_data.get("properties", [])

    if not isinstance(properties, list):
        return None

    if "ammunition" not in properties:
        return None

    ammunition_map = {
        "longbow": "arrow",
        "crossbow-light": "crossbow-bolt",
        "blowgun": "blowgun-needle",
        "sling": "sling-bullet",
    }

    return ammunition_map.get(weapon_idx)


def normalize_inventory(inventory):
    """
    Normaliza un inventario al formato moderno.

    - Convierte entradas de texto antiguas en objetos.
    - Fusiona entradas con el mismo idx.
    - Suma sus cantidades.
    - Descarta cantidades inválidas o no positivas.
    """
    if not isinstance(inventory, list):
        return []

    normalized = []
    positions = {}

    for entry in inventory:
        idx = None
        name = None
        quantity = 1

        if isinstance(entry, str):
            value = entry.strip()

            if not value:
                continue

            idx = value
            name = value

        elif isinstance(entry, dict):
            raw_idx = entry.get("idx")
            raw_name = entry.get("name")
            raw_quantity = entry.get("quantity", 1)

            if isinstance(raw_idx, str) and raw_idx.strip():
                idx = raw_idx.strip()
            elif isinstance(raw_name, str) and raw_name.strip():
                idx = raw_name.strip()

            if isinstance(raw_name, str) and raw_name.strip():
                name = raw_name.strip()
            elif idx:
                name = idx

            if isinstance(raw_quantity, int):
                quantity = raw_quantity

        if not isinstance(idx, str) or not idx:
            continue

        if not isinstance(name, str) or not name:
            name = idx

        if not isinstance(quantity, int) or quantity <= 0:
            continue

        key = idx.lower()

        if key in positions:
            normalized[positions[key]]["quantity"] += quantity
        else:
            positions[key] = len(normalized)
            normalized.append({
                "idx": idx,
                "name": name,
                "quantity": quantity,
            })

    return normalized


def add_item_to_character(character_name, item, quantity=1):
    """
    Añade una cantidad de un objeto al inventario.

    Reconoce tanto el idx como el nombre visible.
    Mantiene el formato moderno con idx, name y quantity.
    """
    if not isinstance(character_name, str):
        return None

    if not isinstance(item, str):
        return None

    item = item.strip()

    if not item:
        return None

    if not isinstance(quantity, int) or quantity <= 0:
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    inventory = character.get("inventory")

    if not isinstance(inventory, list):
        inventory = []
        character["inventory"] = inventory

    item_normalized = item.lower()

    for index, entry in enumerate(inventory):
        if isinstance(entry, str):
            if entry.strip().lower() == item_normalized:
                inventory[index] = {
                    "idx": item,
                    "name": entry.strip(),
                    "quantity": 1 + quantity,
                }
                persist()
                return {
                    "character": character_name,
                    "item": entry.strip(),
                    "quantity": 1 + quantity,
                    "inventory": inventory,
                }

        elif isinstance(entry, dict):
            current_quantity = entry.get("quantity", 0)

            if not isinstance(current_quantity, int) or current_quantity <= 0:
                continue

            idx = entry.get("idx")
            name = entry.get("name")

            matches = (
                isinstance(idx, str)
                and idx.strip().lower() == item_normalized
            ) or (
                isinstance(name, str)
                and name.strip().lower() == item_normalized
            )

            if matches:
                entry["quantity"] = current_quantity + quantity

                if not isinstance(entry.get("idx"), str):
                    entry["idx"] = item

                if not isinstance(entry.get("name"), str):
                    entry["name"] = item

                persist()

                return {
                    "character": character_name,
                    "item": entry["name"],
                    "quantity": entry["quantity"],
                    "inventory": inventory,
                }

    new_entry = {
        "idx": item,
        "name": item,
        "quantity": quantity,
    }

    inventory.append(new_entry)
    persist()

    return {
        "character": character_name,
        "item": item,
        "quantity": quantity,
        "inventory": inventory,
    }


def remove_item_from_character(character_name, item, quantity=1):
    """
    Elimina una cantidad de un objeto del inventario.

    Reconoce tanto el idx como el nombre visible.
    Si la cantidad llega a cero, elimina completamente el objeto.
    """
    if not isinstance(character_name, str):
        return None

    if not isinstance(item, str):
        return None

    item = item.strip()

    if not item:
        return None

    if not isinstance(quantity, int) or quantity <= 0:
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    inventory = character.get("inventory")

    if not isinstance(inventory, list):
        return None

    item_normalized = item.lower()

    for index, entry in enumerate(inventory):
        if isinstance(entry, str):
            if entry.strip().lower() == item_normalized:
                current_quantity = 1

                if quantity > current_quantity:
                    return None

                inventory.pop(index)
                persist()

                return {
                    "character": character_name,
                    "item": entry.strip(),
                    "quantity": 0,
                    "inventory": inventory,
                }

        elif isinstance(entry, dict):
            current_quantity = entry.get("quantity", 0)

            if not isinstance(current_quantity, int) or current_quantity <= 0:
                continue

            idx = entry.get("idx")
            name = entry.get("name")

            matches = (
                isinstance(idx, str)
                and idx.strip().lower() == item_normalized
            ) or (
                isinstance(name, str)
                and name.strip().lower() == item_normalized
            )

            if not matches:
                continue

            if quantity > current_quantity:
                return None

            new_quantity = current_quantity - quantity
            display_name = (
                name.strip()
                if isinstance(name, str) and name.strip()
                else item
            )

            if new_quantity == 0:
                inventory.pop(index)
            else:
                entry["quantity"] = new_quantity

            persist()

            return {
                "character": character_name,
                "item": display_name,
                "quantity": new_quantity,
                "inventory": inventory,
            }

    return None




def get_combatant_conditions(combatant_id):
    """
    Devuelve las condiciones actuales de un combatiente.

    Un personaje obtiene sus condiciones desde su ficha.
    Un monstruo obtiene sus condiciones desde monster_state.
    """
    if not isinstance(combatant_id, str):
        return None

    combatant_id = combatant_id.strip()

    if not combatant_id:
        return None

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    if combatant_id.startswith("character:"):
        character_name = combatant_id[len("character:"):].strip()

        if not character_name:
            return None

        character = get_character_by_name(character_name)

        if not isinstance(character, dict):
            return None

        character_combat = character.get("combat")

        if not isinstance(character_combat, dict):
            return []

        conditions = character_combat.get("conditions")

        if not isinstance(conditions, list):
            return []

        return conditions

    if combatant_id.startswith("monster:"):
        monster_state = combat.get("monster_state")

        if not isinstance(monster_state, dict):
            return None

        state = monster_state.get(combatant_id)

        if not isinstance(state, dict):
            return None

        conditions = state.get("conditions")

        if not isinstance(conditions, list):
            return []

        return conditions

    return None


def get_condition_damage_resistances(conditions):
    """
    Devuelve los tipos de daño contra los que las condiciones
    actuales otorgan resistencia.

    Petrified otorga resistencia contra todo daño.
    """
    if not isinstance(conditions, list):
        return []

    damage_types = {
        "acid",
        "bludgeoning",
        "cold",
        "fire",
        "force",
        "lightning",
        "necrotic",
        "piercing",
        "poison",
        "psychic",
        "radiant",
        "slashing",
        "thunder",
    }

    resistances = set()

    for condition in conditions:
        condition_idx = None

        if isinstance(condition, str):
            condition_idx = condition.strip().lower()

        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")

            if isinstance(condition_idx, str):
                condition_idx = condition_idx.strip().lower()

        if condition_idx == "petrified":
            resistances.update(damage_types)

    return sorted(resistances)


def get_condition_damage_immunities(conditions):
    """
    Devuelve los tipos de daño contra los que las condiciones
    actuales otorgan inmunidad.

    Petrified otorga inmunidad al daño de veneno.
    """
    if not isinstance(conditions, list):
        return []

    immunities = set()

    for condition in conditions:
        condition_idx = None

        if isinstance(condition, str):
            condition_idx = condition.strip().lower()

        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")

            if isinstance(condition_idx, str):
                condition_idx = condition_idx.strip().lower()

        if condition_idx == "petrified":
            immunities.add("poison")

    return sorted(immunities)



def resolve_skill_check_advantage_disadvantage(conditions):
    """
    Determina ventaja o desventaja en una prueba de habilidad
    según las condiciones actuales del personaje.

    Devuelve:
    - advantage: True/False
    - disadvantage: True/False
    - mode: advantage, disadvantage o normal

    Si existen simultáneamente ventaja y desventaja, se cancelan.
    """
    if not isinstance(conditions, list):
        conditions = []

    condition_keys = set()

    for condition in conditions:
        if isinstance(condition, str):
            key = condition.strip().lower()
            if key:
                condition_keys.add(key)

        elif isinstance(condition, dict):
            idx = condition.get("idx")

            if isinstance(idx, str) and idx.strip():
                condition_keys.add(idx.strip().lower())

    advantage = False
    disadvantage = False

    if "frightened" in condition_keys:
        disadvantage = True

    if "poisoned" in condition_keys:
        disadvantage = True

    if advantage and disadvantage:
        mode = "normal"
    elif advantage:
        mode = "advantage"
    elif disadvantage:
        mode = "disadvantage"
    else:
        mode = "normal"

    return {
        "advantage": advantage,
        "disadvantage": disadvantage,
        "mode": mode,
    }

def condition_grants_immunity(conditions, immunity_effect):
    """
    Determina si alguna condición actual concede una inmunidad
    mecánica concreta.
    """
    if not isinstance(conditions, list):
        return False

    if not isinstance(immunity_effect, str):
        return False

    immunity_effect = immunity_effect.strip()

    if not immunity_effect:
        return False

    for condition in conditions:
        condition_idx = None

        if isinstance(condition, str):
            condition_idx = condition.strip().lower()

        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")

            if isinstance(condition_idx, str):
                condition_idx = condition_idx.strip().lower()

        if not condition_idx:
            continue

        condition_effects = get_condition_effects(condition_idx)

        if not isinstance(condition_effects, dict):
            continue

        effects = condition_effects.get("effects")

        if not isinstance(effects, dict):
            continue

        if effects.get(immunity_effect) is True:
            return True

    return False


def apply_condition_to_character(character_name, condition_idx, source_combatant_id=None):
    """
    Aplica una condición válida a un personaje.

    La condición se valida exclusivamente contra Magical20.
    No permite duplicados y persiste el cambio.
    """
    if not isinstance(character_name, str):
        return None

    character_name = character_name.strip()

    if not character_name:
        return None

    condition_data = get_condition_data(condition_idx)

    if not isinstance(condition_data, dict):
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    combat = character.get("combat")

    if not isinstance(combat, dict):
        combat = {}
        character["combat"] = combat

    conditions = combat.get("conditions")

    if not isinstance(conditions, list):
        conditions = []
        combat["conditions"] = conditions

    condition_key = condition_data["idx"].lower()

    if (
        condition_key == "poisoned"
        and condition_grants_immunity(
            conditions,
            "poison_immunity",
        )
    ):
        return {
            "character": character_name,
            "condition": condition_data,
            "applied": False,
            "immune": True,
            "conditions": conditions,
        }

    for existing in conditions:
        if isinstance(existing, str):
            if existing.strip().lower() == condition_key:
                return {
                    "character": character_name,
                    "condition": condition_data,
                    "applied": False,
                    "already_present": True,
                    "conditions": conditions,
                }

        elif isinstance(existing, dict):
            existing_idx = existing.get("idx")

            if (
                isinstance(existing_idx, str)
                and existing_idx.strip().lower() == condition_key
            ):
                return {
                    "character": character_name,
                    "condition": condition_data,
                    "applied": False,
                    "already_present": True,
                    "conditions": conditions,
                }

    condition_entry = condition_data["idx"]

    if isinstance(source_combatant_id, str) and source_combatant_id.strip():
        condition_entry = {
            "idx": condition_data["idx"],
            "source": source_combatant_id.strip(),
        }

    conditions.append(condition_entry)
    persist()

    return {
        "character": character_name,
        "condition": condition_data,
        "applied": True,
        "already_present": False,
        "conditions": conditions,
    }


def remove_condition_from_character(character_name, condition_idx):
    """
    Elimina una condición de un personaje.

    La condición se valida exclusivamente contra Magical20.
    Si no está presente, no modifica el estado.
    """
    if not isinstance(character_name, str):
        return None

    character_name = character_name.strip()

    if not character_name:
        return None

    condition_data = get_condition_data(condition_idx)

    if not isinstance(condition_data, dict):
        return None

    character = get_character_by_name(character_name)

    if character is None:
        return None

    combat = character.get("combat")

    if not isinstance(combat, dict):
        return None

    conditions = combat.get("conditions")

    if not isinstance(conditions, list):
        return None

    condition_key = condition_data["idx"].lower()

    for index, existing in enumerate(conditions):
        existing_idx = None

        if isinstance(existing, str):
            existing_idx = existing.strip()

        elif isinstance(existing, dict):
            existing_idx = existing.get("idx")

        if (
            isinstance(existing_idx, str)
            and existing_idx.lower() == condition_key
        ):
            conditions.pop(index)
            persist()

            return {
                "character": character_name,
                "condition": condition_data,
                "removed": True,
                "conditions": conditions,
            }

    return {
        "character": character_name,
        "condition": condition_data,
        "removed": False,
        "conditions": conditions,
    }


def apply_damage_to_character_by_name(character_name, damage):
    """
    Busca un personaje por nombre, aplica daño y persiste el estado.

    La resolución mecánica del daño ocurre en apply_damage_to_character().
    Esta función únicamente conecta esa resolución con campaign_state.
    """
    global campaign_state

    character = get_character_by_name(character_name)

    if character is None:
        return None

    result = apply_damage_to_character(character, damage)

    if result is None:
        return None

    persist()

    return result


async def resolve_player_check(
    user_id,
    skill_name,
    dc,
    channel,
):
    """
    Resuelve una prueba completa para el jugador indicado.

    Flujo:
    jugador -> personaje -> modificador -> Dice Golem -> resultado.
    """
    character = get_character_for_player(user_id)

    if character is None:
        return None

    result = await resolve_skill_check_with_dice_golem(
        character,
        skill_name,
        dc,
        channel,
    )

    if result is None:
        return None

    character_name = character.get("identity", {}).get("name", "Personaje")

    result["character"] = character_name
    result["user_id"] = str(user_id)

    return result



async def resolve_player_save(
    user_id,
    ability_name,
    dc,
    channel,
):
    """
    Resuelve una tirada de salvación completa para el jugador indicado.

    El LLM proporciona la característica y la CD.
    Python determina el personaje, modificadores, competencia,
    condiciones y reglas de la salvación.
    Dice Golem determina exclusivamente el d20.
    """
    character = get_character_for_player(user_id)

    if character is None:
        return None

    ability_map = {
        "fuerza": "str",
        "destreza": "dex",
        "constitución": "con",
        "constitucion": "con",
        "inteligencia": "int",
        "sabiduría": "wis",
        "sabiduria": "wis",
        "carisma": "cha",
    }

    if not isinstance(ability_name, str):
        return None

    ability_idx = ability_map.get(
        ability_name.strip().lower()
    )

    if ability_idx is None:
        return None

    result = await resolve_saving_throw_with_dice_golem(
        character,
        ability_idx,
        dc,
        channel,
    )

    if result is None:
        return None

    character_name = character.get(
        "identity",
        {},
    ).get(
        "name",
        "Personaje",
    )

    result["character"] = character_name
    result["user_id"] = str(user_id)

    return result


def validate_character_sheet(character):
    """
    Comprueba si una ficha tiene los datos mecánicos mínimos
    necesarios para resolver pruebas de habilidad.
    """
    if not isinstance(character, dict):
        return False

    identity = character.get("identity", {})
    abilities = character.get("abilities", {})
    proficiency = character.get("proficiency", {})

    if not isinstance(identity, dict):
        return False

    if not isinstance(abilities, dict):
        return False

    if not isinstance(proficiency, dict):
        return False

    level = identity.get("level")

    identity_class = identity.get("class")
    if not identity_class:
        return False

    class_idx = normalize_class_idx(identity_class)
    if not class_idx:
        return False

    if get_proficiency_bonus(class_idx, level) is None:
        return False

    required_abilities = (
        "strength",
        "dexterity",
        "constitution",
        "intelligence",
        "wisdom",
        "charisma",
    )

    ability_idx_map = {
        "strength": "str",
        "dexterity": "dex",
        "constitution": "con",
        "intelligence": "int",
        "wisdom": "wis",
        "charisma": "cha",
    }

    for ability in required_abilities:
        ability_idx = ability_idx_map[ability]
        if get_ability_modifier(abilities.get(ability)) is None:
            return False

    skills = proficiency.get("skills", [])

    if not isinstance(skills, list):
        return False

    return True


def normalize_equipment_mode(equipment_mode):
    """
    Normaliza el modo de selección de equipamiento.

    Modos válidos:
    - random: selección aleatoria
    - preset: configuración predefinida
    - manual: selección explícita del jugador

    Si no se especifica un modo, se mantiene el comportamiento
    histórico: selección aleatoria.
    """
    if equipment_mode is None:
        return "random"

    if not isinstance(equipment_mode, str):
        return None

    mode = equipment_mode.strip().lower()

    if mode in ("random", "preset", "manual"):
        return mode

    return None


def configure_character_sheet(
    name,
    race,
    character_class,
    level,
    abilities,
    skills=None,
    saving_throws=None,
    equipment_mode=None,
    equipment_choices=None,
):
    """
    Construye una ficha de personaje completa sin modificar campaign_state.
    Los modificadores derivados se calculan posteriormente, no se almacenan.
    """
    if not name or not isinstance(name, str):
        return None

    try:
        level = int(level)
    except (TypeError, ValueError):
        return None

    if not 1 <= level <= 20:
        return None

    if not isinstance(abilities, dict):
        return None

    equipment_mode = normalize_equipment_mode(equipment_mode)
    if equipment_mode is None:
        return None

    required_abilities = (
        "strength",
        "dexterity",
        "constitution",
        "intelligence",
        "wisdom",
        "charisma",
    )

    normalized_abilities = {}

    for ability in required_abilities:
        try:
            score = int(abilities.get(ability))
        except (TypeError, ValueError):
            return None

        if not 1 <= score <= 30:
            return None

        normalized_abilities[ability] = score

    if skills is None:
        skills = []

    if saving_throws is None:
        saving_throws = []

    if not isinstance(skills, list):
        return None

    if not isinstance(saving_throws, list):
        return None

    character = default_character_sheet(name)

    character["identity"].update({
        "name": name.strip(),
        "race": str(race or "").strip(),
        "class": str(character_class or "").strip(),
        "level": level,
    })

    character["abilities"] = normalized_abilities

    class_idx = normalize_class_idx(character["identity"]["class"])
    if not class_idx:
        return None

    proficiency_bonus = get_proficiency_bonus(class_idx, level)
    if proficiency_bonus is None:
        return None

    hp_max = get_level_one_hp(
        class_idx,
        normalized_abilities["constitution"],
    )
    if hp_max is None:
        return None

    character["combat"]["hp_current"] = hp_max
    character["combat"]["hp_max"] = hp_max

    character["proficiency"] = {
        "bonus": proficiency_bonus,

        "skills": [
            str(skill).strip()
            for skill in skills
            if str(skill).strip()
        ],
        "saving_throws": [
            str(save).strip()
            for save in saving_throws
            if str(save).strip()
        ],
    }

    class_idx = normalize_class_idx(character["identity"]["class"])
    if not class_idx:
        return None

    class_proficiencies = get_class_proficiencies(class_idx)

    if equipment_mode == "manual":
        starting_equipment = resolve_manual_starting_equipment(
            class_idx,
            equipment_choices,
            proficiencies=class_proficiencies,
        )

        if not starting_equipment:
            return None

    elif equipment_mode == "preset":
        try:
            preset_number = int(equipment_choices)
        except (TypeError, ValueError):
            return None

        presets = get_starting_equipment_presets(
            class_idx,
            proficiencies=class_proficiencies,
            preset_count=3,
        )

        selected_preset = next(
            (
                preset
                for preset in presets
                if preset.get("preset") == preset_number
            ),
            None,
        )

        if not selected_preset:
            return None

        starting_equipment = selected_preset.get("equipment", [])

        if not starting_equipment:
            return None

    else:
        # random mantiene el comportamiento histórico.
        starting_equipment = resolve_class_starting_equipment(
            class_idx,
            proficiencies=class_proficiencies,
            rng=random,
        )

    character["inventory"] = [
        {
            "idx": item["idx"],
            "name": item["name"],
            "quantity": item["quantity"],
        }
        for item in starting_equipment
        if isinstance(item, dict)
        and item.get("idx")
        and item.get("name")
        and isinstance(item.get("quantity"), int)
        and item["quantity"] > 0
    ]

    character["inventory"] = normalize_inventory(
        character["inventory"]
    )

    if not validate_character_sheet(character):
        return None

    return character


def migrate_character_sheet(character_name, character):
    """
    Añade la estructura moderna a un personaje existente sin sobrescribir
    datos de campaña ya registrados.
    """
    if not isinstance(character, dict):
        character = {}

    defaults = default_character_sheet(character_name)

    # Añadir únicamente campos que todavía no existen.
    for key, default_value in defaults.items():
        if key not in character:
            character[key] = deepcopy(default_value)

    # Mantener compatibilidad con los campos antiguos.
    if "inventory" not in character or not isinstance(character["inventory"], list):
        character["inventory"] = []

    character["inventory"] = normalize_inventory(
        character["inventory"]
    )

    if "notes" not in character or not isinstance(character["notes"], list):
        character["notes"] = []

    # Migrar HP antiguo: "actual/máximo" -> combat.hp_current / hp_max.
    old_hp = character.get("hp")
    combat = character.get("combat")

    if isinstance(combat, dict) and isinstance(old_hp, str) and "/" in old_hp:
        try:
            current_hp, max_hp = old_hp.split("/", 1)
            combat["hp_current"] = int(current_hp.strip())
            combat["hp_max"] = int(max_hp.strip())
        except (ValueError, TypeError):
            pass

    # Migrar CA antigua al nuevo modelo.
    old_ac = character.get("ac")

    if isinstance(combat, dict):
        if isinstance(old_ac, int):
            combat["ac"] = old_ac

    return character
def create_character_in_campaign(
    user_id,
    name,
    race,
    character_class,
    level,
    abilities,
    skills=None,
    saving_throws=None,
    equipment_mode=None,
    equipment_choices=None,
):
    """
    Crea y registra una ficha en la campaña actual.

    La propuesta recibida no modifica directamente campaign_state.
    La ficha debe pasar primero por configure_character_sheet()
    y validate_character_sheet().
    """
    global campaign_state

    user_id = str(user_id)

    bindings = campaign_state.get("player_bindings", {})
    characters = campaign_state.get("characters", {})

    if not isinstance(bindings, dict):
        return None, "Las asociaciones de jugadores no son válidas."

    if not isinstance(characters, dict):
        return None, "El registro de personajes no es válido."

    if not user_id.isdigit():
        return None, "El identificador del jugador no es válido."

    character = configure_character_sheet(
        name=name,
        race=race,
        character_class=character_class,
        level=level,
        abilities=abilities,
        skills=skills,
        saving_throws=saving_throws,
        equipment_mode=equipment_mode,
        equipment_choices=equipment_choices,
    )

    if character is None:
        return None, "La ficha propuesta no supera la validación mecánica."

    character_name = character["identity"]["name"]

    if character_name in characters:
        return None, f"Ya existe un personaje llamado {character_name}."

    characters[character_name] = character
    bindings[user_id] = character_name

    campaign_state["characters"] = characters
    campaign_state["player_bindings"] = bindings

    sanitize_campaign_state()

    persist()

    return character, None



def apply_damage_to_character(character, damage):
    """
    Aplica daño a los puntos de golpe de una ficha.

    Esta función modifica únicamente la ficha recibida.
    No persiste campaign_state ni depende del LLM o Dice Golem.

    Devuelve un resumen del cambio o None si los datos no son válidos.
    """
    if not isinstance(character, dict):
        return None

    combat = character.get("combat")

    if not isinstance(combat, dict):
        return None

    try:
        damage = int(damage)
        hp_current = int(combat.get("hp_current"))
        hp_max = int(combat.get("hp_max"))
    except (TypeError, ValueError):
        return None

    if damage < 0:
        return None

    if hp_max < 0 or hp_current < 0:
        return None

    if hp_current > hp_max:
        hp_current = hp_max

    hp_before = hp_current
    hp_after = max(0, hp_current - damage)

    combat["hp_current"] = hp_after
    combat["hp_max"] = hp_max

    return {
        "damage": damage,
        "hp_before": hp_before,
        "hp_after": hp_after,
        "hp_max": hp_max,
        "unconscious": hp_after == 0,
    }



def add_character_to_combat(character_name):
    """
    Añade un personaje existente al combate actual.

    El combatante mantiene una referencia a la ficha real.
    No duplica HP, CA ni otras estadísticas.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        combatants = []
        combat["combatants"] = combatants

    character = get_character_by_name(character_name)

    if character is None:
        return None

    character_name = character_name.strip()
    combatant_id = f"character:{character_name}"

    for combatant in combatants:
        if (
            isinstance(combatant, dict)
            and combatant.get("id") == combatant_id
        ):
            return combatant

    combatant = {
        "id": combatant_id,
        "type": "character",
        "character_name": character_name,
        "initiative": None,
    }

    combatants.append(combatant)

    return combatant



def add_monster_to_combat(monster_idx):
    """
    Añade una instancia de monstruo al combate actual.

    Magical20 proporciona las estadísticas base del monstruo.
    El combatiente guarda únicamente la referencia al monstruo.
    El HP dinámico de cada instancia se guarda en monster_state.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        combatants = []
        combat["combatants"] = combatants

    if not isinstance(monster_idx, str):
        return None

    monster_idx = monster_idx.strip()

    if not monster_idx:
        return None

    monster_data = get_monster_combat_data(monster_idx)

    if monster_data is None:
        return None

    hp_max = monster_data.get("hp")

    try:
        hp_max = int(hp_max)
    except (TypeError, ValueError):
        return None

    if hp_max < 0:
        return None

    existing_numbers = set()

    prefix = f"monster:{monster_idx}:"

    for combatant in combatants:
        if not isinstance(combatant, dict):
            continue

        combatant_id = combatant.get("id", "")

        if not isinstance(combatant_id, str):
            continue

        if not combatant_id.startswith(prefix):
            continue

        suffix = combatant_id[len(prefix):]

        try:
            existing_numbers.add(int(suffix))
        except (TypeError, ValueError):
            continue

    instance_number = 1

    while instance_number in existing_numbers:
        instance_number += 1

    combatant_id = f"{prefix}{instance_number}"

    combatant = {
        "id": combatant_id,
        "type": "monster",
        "monster_idx": monster_idx,
        "initiative": None,
    }

    combatants.append(combatant)

    monster_state = combat.get("monster_state")

    if not isinstance(monster_state, dict):
        monster_state = {}
        combat["monster_state"] = monster_state

    monster_state[combatant_id] = {
        "hp_current": hp_max,
        "hp_max": hp_max,
        "conditions": [],
    }

    return combatant


async def resolve_combat_initiative(channel):
    """
    Tira iniciativa para todos los combatientes actuales.

    Dice Golem:
    - tira exclusivamente 1d20.

    Python:
    - obtiene el modificador;
    - calcula el total;
    - ordena los combatientes.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list) or not combatants:
        return None

    resolved = []

    for combatant in combatants:
        if not isinstance(combatant, dict):
            return None

        combatant_id = combatant.get("id")
        combatant_type = combatant.get("type")

        if not isinstance(combatant_id, str):
            return None

        initiative_modifier = None

        if combatant_type == "character":
            character_name = combatant.get("character_name")
            character = get_character_by_name(character_name)

            if character is None:
                return None

            abilities = character.get("abilities", {})

            if not isinstance(abilities, dict):
                return None

            initiative_modifier = get_initiative_modifier(
                abilities.get("dexterity")
            )

        elif combatant_type == "monster":
            monster_idx = combatant.get("monster_idx")

            if not isinstance(monster_idx, str):
                return None

            monster_data = get_monster_combat_data(monster_idx)

            if monster_data is None:
                return None

            initiative_modifier = monster_data.get(
                "initiative_modifier"
            )

        else:
            return None

        if initiative_modifier is None:
            return None

        roll_result = await roll_with_dice_golem(
            channel,
            "1d20",
            f"Iniciativa {combatant_id}",
        )

        if not roll_result:
            return None

        d20_result = roll_result.get("total")

        try:
            d20_result = int(d20_result)
            initiative_modifier = int(initiative_modifier)
        except (TypeError, ValueError):
            return None

        if d20_result < 1 or d20_result > 20:
            return None

        total = d20_result + initiative_modifier

        combatant["initiative"] = total

        resolved.append({
            "id": combatant_id,
            "d20": d20_result,
            "initiative_modifier": initiative_modifier,
            "total": total,
        })

    resolved.sort(
        key=lambda entry: (
            entry["total"],
            entry["d20"],
        ),
        reverse=True,
    )

    combat["initiative"] = [
        entry["id"]
        for entry in resolved
    ]

    return resolved

def is_combatant_turn(combatant_id):
    """
    Comprueba si un combatiente tiene actualmente el turno.

    No modifica el estado del combate.
    """
    if not isinstance(combatant_id, str):
        return False

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return False

    if combat.get("active") is not True:
        return False

    turn = combat.get("turn")

    if not isinstance(turn, str):
        return False

    return turn == combatant_id


def start_combat_turn():
    """
    Inicia el primer turno del combate usando el orden de iniciativa.

    No realiza acciones ni modifica estadísticas del combatiente.
    Solo establece el turno actual y la ronda.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    initiative = combat.get("initiative")

    if not isinstance(initiative, list) or not initiative:
        return None

    if not isinstance(combat.get("combatants"), list):
        return None

    current_turn = initiative[0]

    valid_ids = {
        combatant.get("id")
        for combatant in combat["combatants"]
        if isinstance(combatant, dict)
    }

    if current_turn not in valid_ids:
        return None

    combat["round"] = 1
    combat["turn"] = current_turn

    return {
        "round": combat["round"],
        "turn": combat["turn"],
    }


def advance_combat_turn():
    """
    Avanza al siguiente combatiente según el orden de iniciativa.

    Si termina la ronda, vuelve al primer combatiente e incrementa la ronda.
    No realiza acciones ni modifica estadísticas del combatiente.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    initiative = combat.get("initiative")

    if not isinstance(initiative, list) or not initiative:
        return None

    current_turn = combat.get("turn")

    if current_turn not in initiative:
        return None

    try:
        current_round = int(combat.get("round"))
    except (TypeError, ValueError):
        return None

    current_index = initiative.index(current_turn)

    if current_index + 1 >= len(initiative):
        next_turn = initiative[0]
        current_round += 1
    else:
        next_turn = initiative[current_index + 1]

    combat["round"] = current_round
    combat["turn"] = next_turn

    return {
        "round": combat["round"],
        "turn": combat["turn"],
    }


async def start_combat(channel):
    """
    Inicia formalmente el combate actual.

    - Requiere al menos un combatiente.
    - No permite iniciar un combate ya activo.
    - Resuelve la iniciativa mediante Dice Golem.
    - Establece la ronda 1 y el primer turno.
    - No realiza acciones de combate.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    if combat.get("active") is True:
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list) or not combatants:
        return None

    combat["initiative"] = []
    combat["round"] = 0
    combat["turn"] = None

    initiative = await resolve_combat_initiative(channel)

    if not isinstance(initiative, list) or not initiative:
        combat["initiative"] = []
        combat["round"] = 0
        combat["turn"] = None
        combat["active"] = False
        return None

    combat["active"] = True

    turn = start_combat_turn()

    if turn is None:
        combat["active"] = False
        combat["initiative"] = []
        combat["round"] = 0
        combat["turn"] = None
        return None

    return {
        "active": combat["active"],
        "initiative": initiative,
        "round": combat["round"],
        "turn": combat["turn"],
    }


def end_combat():
    """
    Finaliza el combate actual.

    Conserva los combatientes registrados, pero limpia el estado
    específico de la ronda y la iniciativa.

    No modifica HP, inventario ni fichas.
    """
    global campaign_state

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combat["active"] = False
    combat["round"] = 0
    combat["turn"] = None
    combat["initiative"] = []

    return {
        "active": combat["active"],
        "round": combat["round"],
        "turn": combat["turn"],
    }


def sanitize_campaign_state():

    """Evita que un JSON del LLM pueda romper los paneles o el contexto."""
    global campaign_state
    defaults = default_campaign_state()

    # Secciones que deben ser diccionarios.
    for key in ("campaign", "characters", "npcs", "locations", "quests", "factions", "loot", "combat", "discord_panels"):
        if not isinstance(campaign_state.get(key), dict):
            campaign_state[key] = deepcopy(defaults[key])

    # Entradas de personajes/NPC/facciones siempre deben ser objetos.
    for key in ("characters", "npcs", "factions"):
        value = campaign_state.get(key, {})
    
        if key == "characters":
            campaign_state[key] = {
                str(name): migrate_character_sheet(str(name), obj)
                for name, obj in value.items()
                if isinstance(obj, dict)
            }
        else:
            campaign_state[key] = {
                str(name): obj
                for name, obj in value.items()
                if isinstance(obj, dict)
            }

    # Asociaciones entre usuarios de Discord y personajes.
    bindings = campaign_state.get("player_bindings", {})
    if not isinstance(bindings, dict):
        bindings = deepcopy(defaults["player_bindings"])

    campaign_state["player_bindings"] = {
        str(discord_id): str(character_name)
        for discord_id, character_name in bindings.items()
        if str(discord_id).isdigit() and str(character_name).strip()
    }

    # Estado temporal de creación de personaje.
    character_creation = campaign_state.get("character_creation")

    if character_creation is not None:
        if not isinstance(character_creation, dict):
            campaign_state["character_creation"] = None
        else:
            user_id = character_creation.get("user_id")

            if not isinstance(user_id, str) or not user_id.isdigit():
                campaign_state["character_creation"] = None
            else:
                stage = character_creation.get("stage", "mode")

                if stage not in ("mode", "preset", "manual"):
                    character_creation["stage"] = "mode"

                for field in (
                    "guild_id",
                    "channel_id",
                    "message_id",
                ):
                    value = character_creation.get(field)

                    if value is not None:
                        character_creation[field] = str(value)

                try:
                    current_group = int(
                        character_creation.get(
                            "current_group",
                            1,
                        )
                    )
                except (TypeError, ValueError):
                    current_group = 1

                character_creation["current_group"] = max(
                    1,
                    current_group,
                )

                if not isinstance(
                    character_creation.get("selected_choices"),
                    list,
                ):
                    character_creation["selected_choices"] = []

                if not isinstance(
                    character_creation.get("selected_alternatives"),
                    dict,
                ):
                    character_creation["selected_alternatives"] = {}

                if stage == "preset" and not isinstance(
                    character_creation.get("presets"),
                    list,
                ):
                    character_creation["presets"] = []

    # Tipos de colecciones.
    for key in ("clocks", "secrets", "facts"):
        if not isinstance(campaign_state.get(key), list):
            campaign_state[key] = deepcopy(defaults[key])

    quests = campaign_state.get("quests", {})
    for key in ("active", "completed", "clues"):
        if not isinstance(quests.get(key), list):
            quests[key] = deepcopy(defaults["quests"][key])

    loot = campaign_state.get("loot", {})
    if not isinstance(loot.get("items"), list):
        loot["items"] = []

    combat = campaign_state.get("combat", {})
    for key in ("combatants", "initiative"):
        if not isinstance(combat.get(key), list):
            combat[key] = []
    for key in ("conditions", "concentration", "reaction_used"):
        if not isinstance(combat.get(key), dict):
            combat[key] = {}

    try:
        campaign_state["image_counter"] = max(0, int(campaign_state.get("image_counter", 0)))
    except (TypeError, ValueError):
        campaign_state["image_counter"] = 0


sanitize_campaign_state()



async def generate_character_proposal(request_text):
    """
    Genera una propuesta de personaje mediante el LLM.

    Esta función NO modifica campaign_state.
    El resultado debe pasar posteriormente por
    configure_character_sheet() antes de registrarse.
    """
    proposal_prompt = """
Actúa como diseñador de personajes para una campaña de D&D 5e.

Debes convertir la solicitud del jugador en una propuesta COMPLETA
de personaje.

Reglas:
- Si faltan datos, invéntalos de forma coherente con la solicitud.
- Si el jugador especifica raza, clase, nivel u orientación, respétalos.
- El nivel debe estar entre 1 y 20.
- Debes proporcionar las seis características.
- Las características deben ser números enteros entre 1 y 30.
- Las habilidades deben utilizar los nombres de habilidades de D&D en español.
- Los tiros de salvación deben utilizar nombres de características en español.
- No inventes modificadores, HP, CA ni iniciativa como valores mecánicos finales.
  Python calculará esos valores posteriormente.
- No escribas explicaciones.
- No escribas Markdown.
- Devuelve EXCLUSIVAMENTE un objeto JSON válido.

Formato obligatorio:

{
  "name": "Nombre del personaje",
  "race": "Raza",
  "class": "Clase",
  "level": 1,
  "abilities": {
    "strength": 10,
    "dexterity": 10,
    "constitution": 10,
    "intelligence": 10,
    "wisdom": 10,
    "charisma": 10
  },
  "skills": [],
  "saving_throws": [],
  "equipment_mode": "random",
  "equipment_choices": null
}

EQUIPAMIENTO:
- equipment_mode debe ser exactamente "random" o "preset".
- Si equipment_mode es "random", equipment_choices debe ser null.
- Si equipment_mode es "preset", equipment_choices debe ser un entero entre 1 y 3.
- No inventes objetos ni IDs de equipamiento.
- No escribas una lista manual de objetos.
"""

    try:
        raw_reply, model_used = await ask_text_llm(
            proposal_prompt,
            "",
            [],
            str(request_text or "").strip(),
        )
    except Exception as exc:
        print(f"[PERSONAJE] Error generando propuesta: {exc}")
        return None

    if not raw_reply:
        return None

    text = str(raw_reply).strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        proposal = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            print("[PERSONAJE] El LLM no devolvió JSON válido.")
            return None

        try:
            proposal = json.loads(match.group(0))
        except json.JSONDecodeError:
            print("[PERSONAJE] No se pudo interpretar la propuesta.")
            return None

    if not isinstance(proposal, dict):
        return None

    required_keys = (
        "name",
        "race",
        "class",
        "level",
        "abilities",
        "skills",
        "saving_throws",
        "equipment_mode",
        "equipment_choices",
    )

    if any(key not in proposal for key in required_keys):
        print("[PERSONAJE] Propuesta incompleta.")
        return None

    if not isinstance(proposal.get("abilities"), dict):
        return None

    if not isinstance(proposal.get("skills"), list):
        return None

    if not isinstance(proposal.get("saving_throws"), list):
        return None

    equipment_mode = proposal.get("equipment_mode")

    if not isinstance(equipment_mode, str):
        return None

    equipment_mode = equipment_mode.strip().lower()

    if equipment_mode not in ("random", "preset"):
        return None

    proposal["equipment_mode"] = equipment_mode

    if equipment_mode == "random":
        proposal["equipment_choices"] = None
    else:
        try:
            preset_number = int(proposal.get("equipment_choices"))
        except (TypeError, ValueError):
            return None

        if preset_number not in (1, 2, 3):
            return None

        proposal["equipment_choices"] = preset_number

    return proposal

def deep_patch(dst, patch):
    if not isinstance(dst,dict) or not isinstance(patch,dict): return
    for k,v in patch.items():
        if k=="new_facts": continue
        # El modelo no puede borrar accidentalmente estado persistente con null.
        if v is None: continue
        if isinstance(v,dict) and isinstance(dst.get(k),dict): deep_patch(dst[k],v)
        else: dst[k]=v


def update_state_from_model(raw_reply):
    match=re.search(r"<<<CAMPAÑA_JSON>>>(.*?)<<<FIN_CAMPAÑA_JSON>>>",raw_reply,re.DOTALL)
    if not match: return
    try:
        patch=json.loads(match.group(1).strip())
        if not isinstance(patch,dict): return
        allowed={"campaign","characters","npcs","locations","quests","factions","loot","combat","clocks","secrets"}
        for key in allowed:
            if key not in patch: continue
            if key=="clocks":
                if isinstance(patch[key],list): campaign_state[key]=patch[key]
            elif isinstance(patch[key],dict): deep_patch(campaign_state.setdefault(key,{}),patch[key])
        for fact in patch.get("new_facts",[]):
            if isinstance(fact,str): add_fact(fact)
        sanitize_campaign_state()
    except Exception as exc: print(f"[ESTADO] JSON de campaña inválido: {exc}")


# ============================================================
# PANELES DISCORD
# ============================================================

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)



class CharacterEquipmentModeView(discord.ui.View):
    """
    Pantalla inicial para elegir el modo de equipamiento.
    """

    def __init__(self, user_id, proposal, timeout=900):
        super().__init__(timeout=timeout)

        self.user_id = int(user_id)
        self.proposal = deepcopy(proposal)
        self.message = None

        self.add_item(
            CharacterEquipmentModeSelect(self)
        )

    async def interaction_check(self, interaction):
        return interaction.user.id == self.user_id

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True

        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass


class CharacterEquipmentModeSelect(discord.ui.Select):
    """
    Selector de modo:
    random / preset / manual.
    """

    def __init__(self, view):
        self.creation_view = view

        options = [
            discord.SelectOption(
                label="Aleatorio",
                description="El sistema elige automáticamente.",
                value="random",
                emoji="🎲",
            ),
            discord.SelectOption(
                label="Preset",
                description="Elegí una configuración predefinida.",
                value="preset",
                emoji="📋",
            ),
            discord.SelectOption(
                label="Manual",
                description="Elegí cada opción del equipamiento.",
                value="manual",
                emoji="🛠️",
            ),
        ]

        super().__init__(
            placeholder="Elegí cómo obtener el equipamiento...",
            min_values=1,
            max_values=1,
            options=options,
        )

    async def callback(self, interaction):
        if interaction.user.id != self.creation_view.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        mode = self.values[0]
        proposal = deepcopy(self.creation_view.proposal)

        if mode == "random":
            character, error = create_character_in_campaign(
                user_id=self.creation_view.user_id,
                name=proposal.get("name"),
                race=proposal.get("race"),
                character_class=proposal.get("class"),
                level=proposal.get("level"),
                abilities=proposal.get("abilities"),
                skills=proposal.get("skills"),
                saving_throws=proposal.get("saving_throws"),
                equipment_mode="random",
                equipment_choices=None,
            )

            if character is None:
                await interaction.response.send_message(
                    f"No pude crear la ficha: {error}",
                    ephemeral=True,
                )
                return

            campaign_state["character_creation"] = None
            persist()
            self.creation_view.stop()

            identity = character["identity"]
            abilities = character["abilities"]

            await interaction.response.edit_message(
                content=(
                    f"**Ficha creada:** {identity['name']}\n"
                    f"**Raza:** {identity['race']}\n"
                    f"**Clase:** {identity['class']}\n"
                    f"**Nivel:** {identity['level']}\n"
                    f"**Características:** "
                    f"FUE {abilities['strength']} · "
                    f"DES {abilities['dexterity']} · "
                    f"CON {abilities['constitution']} · "
                    f"INT {abilities['intelligence']} · "
                    f"SAB {abilities['wisdom']} · "
                    f"CAR {abilities['charisma']}"
                ),
                view=None,
            )
            return

        class_idx = normalize_class_idx(
            proposal.get("class")
        )

        if not class_idx:
            await interaction.response.send_message(
                "No pude identificar la clase del personaje.",
                ephemeral=True,
            )
            return

        proficiencies = get_class_proficiencies(class_idx)

        if mode == "preset":
            presets = get_starting_equipment_presets(
                class_idx,
                proficiencies=proficiencies,
                preset_count=3,
            )

            if not presets:
                await interaction.response.send_message(
                    "No hay presets válidos para esta clase.",
                    ephemeral=True,
                )
                return

            preset_view = CharacterEquipmentPresetView(
                self.creation_view.user_id,
                proposal,
                presets,
            )

            creation_state = campaign_state.get(
                "character_creation"
            )

            if not isinstance(creation_state, dict):
                creation_state = {}

            creation_state.update({
                "user_id": str(self.creation_view.user_id),
                "guild_id": str(
                    getattr(interaction.guild, "id", "")
                ),
                "channel_id": str(interaction.channel.id),
                "message_id": str(
                    interaction.message.id
                ),
                "stage": "preset",
                "proposal": deepcopy(proposal),
                "equipment_options": deepcopy(
                    creation_state.get(
                        "equipment_options",
                        [],
                    )
                ),
                "selected_choices": [],
                "selected_alternatives": {},
                "current_group": 1,
                "presets": deepcopy(presets),
            })

            campaign_state["character_creation"] = creation_state
            persist()

            await interaction.response.edit_message(
                content="Elegí un preset de equipamiento:",
                embed=None,
                view=preset_view,
            )

            preset_view.message = await interaction.original_response()
            return

        equipment_options = get_manual_starting_equipment_options(
            class_idx,
            proficiencies=proficiencies,
        )

        if not equipment_options:
            await interaction.response.send_message(
                "No encontré opciones de equipamiento manual para esta clase.",
                ephemeral=True,
            )
            return

        campaign_state["character_creation"] = {
            "user_id": str(self.creation_view.user_id),
            "guild_id": str(
                getattr(interaction.guild, "id", "")
            ),
            "channel_id": str(interaction.channel.id),
            "message_id": str(interaction.message.id),
            "stage": "manual",
            "proposal": deepcopy(proposal),
            "equipment_options": deepcopy(equipment_options),
            "selected_choices": [],
            "selected_alternatives": {},
            "current_group": 1,
        }

        persist()

        view = CharacterCreationView(
            self.creation_view.user_id,
            equipment_options,
            proposal=proposal,
        )

        await interaction.response.edit_message(
            content="Elegí el equipamiento inicial:",
            embed=view.build_embed(),
            view=view,
        )

        view.message = await interaction.original_response()

        creation_state = campaign_state.get(
            "character_creation"
        )

        if isinstance(creation_state, dict):
            creation_state["guild_id"] = str(
                getattr(interaction.guild, "id", "")
            )
            creation_state["channel_id"] = str(
                interaction.channel.id
            )
            creation_state["message_id"] = str(
                view.message.id
            )
            creation_state["stage"] = "manual"
            persist()


class CharacterEquipmentPresetView(discord.ui.View):
    """
    Selección de preset de equipamiento.
    """

    def __init__(self, user_id, proposal, presets, timeout=900):
        super().__init__(timeout=timeout)

        self.user_id = int(user_id)
        self.proposal = deepcopy(proposal)
        self.presets = presets
        self.message = None

        options = []

        for preset in presets:
            number = preset.get("preset")

            if number is None:
                continue

            equipment = preset.get("equipment", [])

            names = []

            for item in equipment:
                if not isinstance(item, dict):
                    continue

                name = item.get("name")
                quantity = item.get("quantity", 1)

                if name:
                    names.append(
                        f"{name} x{quantity}"
                        if quantity != 1
                        else str(name)
                    )

            options.append(
                discord.SelectOption(
                    label=f"Preset {number}",
                    description=", ".join(names)[:100] or "Configuración disponible",
                    value=str(number),
                )
            )

        self.add_item(
            CharacterEquipmentPresetSelect(self, options)
        )

    async def interaction_check(self, interaction):
        return interaction.user.id == self.user_id

    async def on_timeout(self):
        for item in self.children:
            item.disabled = True

        if self.message:
            try:
                await self.message.edit(view=self)
            except Exception:
                pass


class CharacterEquipmentPresetSelect(discord.ui.Select):
    def __init__(self, view, options):
        self.creation_view = view

        super().__init__(
            placeholder="Elegí un preset...",
            min_values=1,
            max_values=1,
            options=options[:25],
        )

    async def callback(self, interaction):
        if interaction.user.id != self.creation_view.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        try:
            preset_number = int(self.values[0])
        except (TypeError, ValueError):
            await interaction.response.send_message(
                "Preset inválido.",
                ephemeral=True,
            )
            return

        preset = next(
            (
                item
                for item in self.creation_view.presets
                if item.get("preset") == preset_number
            ),
            None,
        )

        if not preset:
            await interaction.response.send_message(
                "No encontré ese preset.",
                ephemeral=True,
            )
            return

        character, error = create_character_in_campaign(
            user_id=self.creation_view.user_id,
            name=self.creation_view.proposal.get("name"),
            race=self.creation_view.proposal.get("race"),
            character_class=self.creation_view.proposal.get("class"),
            level=self.creation_view.proposal.get("level"),
            abilities=self.creation_view.proposal.get("abilities"),
            skills=self.creation_view.proposal.get("skills"),
            saving_throws=self.creation_view.proposal.get("saving_throws"),
            equipment_mode="preset",
            equipment_choices=preset_number,
        )

        if character is None:
            await interaction.response.send_message(
                f"No pude crear la ficha: {error}",
                ephemeral=True,
            )
            return

        campaign_state["character_creation"] = None
        persist()
        self.creation_view.stop()

        identity = character["identity"]
        abilities = character["abilities"]

        await interaction.response.edit_message(
            content=(
                f"**Ficha creada:** {identity['name']}\n"
                f"**Raza:** {identity['race']}\n"
                f"**Clase:** {identity['class']}\n"
                f"**Nivel:** {identity['level']}\n"
                f"**Características:** "
                f"FUE {abilities['strength']} · "
                f"DES {abilities['dexterity']} · "
                f"CON {abilities['constitution']} · "
                f"INT {abilities['intelligence']} · "
                f"SAB {abilities['wisdom']} · "
                f"CAR {abilities['charisma']}"
            ),
            view=None,
        )


class EquipmentAlternativeSelect(discord.ui.Select):
    """
    Selector de la alternativa de un grupo de equipamiento.

    Los elementos fijos de la alternativa se incorporan
    automáticamente. Solo se muestran controles para las
    elecciones que realmente debe realizar el jugador.
    """

    def __init__(self, view, group_index, alternatives):
        self.creation_view = view
        self.group_index = group_index
        self.alternatives = alternatives

        options = []

        for index, alternative in enumerate(alternatives):
            if not isinstance(alternative, dict):
                continue

            description = self._describe_option(alternative)

            options.append(
                discord.SelectOption(
                    label=f"Opción {index + 1}",
                    description=description[:100] if description else None,
                    value=str(index),
                )
            )

        if not options:
            options.append(
                discord.SelectOption(
                    label="Sin alternativas",
                    value="__invalid__",
                )
            )

        super().__init__(
            placeholder="Elegí una alternativa...",
            min_values=1,
            max_values=1,
            options=options[:25],
        )

    @staticmethod
    def _describe_option(option):
        if not isinstance(option, dict):
            return ""

        option_type = option.get("type")

        if option_type == "item":
            return str(
                option.get("name")
                or option.get("idx")
                or "Objeto"
            )

        if option_type == "category":
            category = str(
                option.get("category")
                or "categoría"
            )

            try:
                choose = int(
                    option.get("choose", 1) or 1
                )
            except (TypeError, ValueError):
                choose = 1

            return (
                f"{choose} × {category}"
                if choose > 1
                else category
            )

        if option_type == "multiple":
            items = option.get("items", [])

            if not isinstance(items, list):
                return ""

            parts = []

            for item in items:
                text = EquipmentAlternativeSelect._describe_option(item)

                if text:
                    parts.append(text)

            return " + ".join(parts)

        if option_type == "counted_reference":
            return str(
                option.get("name")
                or option.get("idx")
                or "Objeto"
            )

        if option_type == "choice":
            options = option.get("options", [])

            if isinstance(options, list):
                parts = []

                for item in options:
                    text = EquipmentAlternativeSelect._describe_option(item)

                    if text:
                        parts.append(text)

                return " / ".join(parts)

        return str(
            option.get("name")
            or option.get("idx")
            or ""
        )

    @staticmethod
    def _collect_fixed_values(option):
        """
        Devuelve solamente los componentes que ya están
        determinados por la alternativa.

        item/count_reference -> automático
        category/choice      -> requiere selección
        multiple             -> recursivo
        """

        if not isinstance(option, dict):
            return []

        option_type = option.get("type")

        if option_type in ("item", "counted_reference"):
            idx = option.get("idx")

            if isinstance(idx, str):
                return [idx]

            return []

        if option_type == "multiple":
            values = []

            items = option.get("items", [])

            if isinstance(items, list):
                for item in items:
                    values.extend(
                        EquipmentAlternativeSelect._collect_fixed_values(
                            item
                        )
                    )

            return values

        return []

    async def callback(self, interaction):
        if interaction.user.id != self.creation_view.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        if "__invalid__" in self.values:
            await interaction.response.send_message(
                "No hay alternativas válidas disponibles.",
                ephemeral=True,
            )
            return

        try:
            alternative_index = int(self.values[0])
        except (TypeError, ValueError):
            await interaction.response.send_message(
                "La alternativa seleccionada no es válida.",
                ephemeral=True,
            )
            return

        if not (
            0 <= alternative_index < len(self.alternatives)
        ):
            await interaction.response.send_message(
                "La alternativa seleccionada no es válida.",
                ephemeral=True,
            )
            return

        alternative = self.alternatives[alternative_index]

        self.creation_view.selected_alternatives[
            self.group_index
        ] = alternative_index

        # Al cambiar de alternativa, se eliminan las
        # selecciones anteriores de ese grupo.
        self.creation_view.selections = [
            selection
            for selection in self.creation_view.selections
            if selection.get("group") != self.group_index
        ]

        # Los componentes fijos se agregan automáticamente.
        fixed_values = self._collect_fixed_values(alternative)

        if fixed_values:
            self.creation_view.register_selection(
                self.group_index,
                fixed_values,
            )
        else:
            self.creation_view._persist_state()

        self.creation_view.rebuild()

        await interaction.response.edit_message(
            embed=self.creation_view.build_embed(),
            view=self.creation_view,
        )


class EquipmentChoiceSelect(discord.ui.Select):
    """
    Selector de equipamiento generado a partir de las opciones de Magical20.
    """

    def __init__(self, view, group_index, options, choose=1):
        self.creation_view = view
        self.group_index = group_index
        self.choose = max(1, int(choose or 1))

        select_options = []

        for option_index, option in enumerate(options):
            if not isinstance(option, dict):
                continue

            option_type = option.get("type")

            if option_type == "item":
                idx = option.get("idx")
                name = option.get("name") or idx

                if not isinstance(idx, str):
                    continue

                display_name = (
                    self.creation_view._display_selection_value(
                        idx
                    )
                )

                select_options.append(
                    discord.SelectOption(
                        label=str(display_name)[:100],
                        value=idx,
                    )
                )

            elif option_type == "category":
                category = option.get("category")
                category_options = option.get("options", [])

                if not isinstance(category, str):
                    continue

                if not isinstance(category_options, list):
                    continue

                for item in category_options:
                    if not isinstance(item, dict):
                        continue

                    idx = item.get("idx")
                    name = item.get("name") or idx

                    if not isinstance(idx, str):
                        continue

                    display_name = (
                        self.creation_view._display_selection_value(
                            idx
                        )
                    )

                    select_options.append(
                        discord.SelectOption(
                            label=str(display_name)[:100],
                            value=idx,
                        )
                    )

            elif option_type in ("multiple", "alternatives"):
                # Las estructuras compuestas se manejan mediante
                # subselectores en la vista; no se deben aplanar aquí.
                continue

        select_options = select_options[:25]

        if not select_options:
            select_options.append(
                discord.SelectOption(
                    label="Sin opciones disponibles",
                    value="__invalid__",
                )
            )

        max_values = min(
            self.choose,
            len(select_options),
        )

        min_values = max_values if self.choose > 1 else 1

        super().__init__(
            placeholder="Elegí tu equipamiento...",
            min_values=min_values,
            max_values=max_values,
            options=select_options,
        )

    async def callback(self, interaction):
        if interaction.user.id != self.creation_view.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        if "__invalid__" in self.values:
            await interaction.response.send_message(
                "No hay opciones válidas disponibles.",
                ephemeral=True,
            )
            return

        self.creation_view._replace_player_selection(
            self.group_index,
            list(self.values),
        )

        self.creation_view.rebuild()

        await interaction.response.edit_message(
            embed=self.creation_view.build_embed(),
            view=self.creation_view,
        )


class CharacterCreationView(discord.ui.View):
    """
    UI interactiva para la creación manual de equipamiento.

    Cada grupo debe quedar completamente resuelto antes de avanzar.
    Las selecciones se mantienen en memoria y se persisten en
    campaign_state["character_creation"].
    """

    def __init__(
        self,
        user_id,
        equipment_options,
        proposal=None,
        timeout=900,
    ):
        super().__init__(timeout=timeout)

        self.user_id = int(user_id)
        self.equipment_options = equipment_options or []
        self.proposal = proposal or {}
        self.current_group = 0
        self.selections = []
        self.selected_alternatives = {}
        self.message = None

        self.rebuild()

    def _get_group(self, group_index):
        index = int(group_index) - 1

        if not (
            0 <= index < len(self.equipment_options)
        ):
            return None

        group = self.equipment_options[index]

        return group if isinstance(group, dict) else None

    def _get_selection(self, group_index):
        for selection in self.selections:
            if selection.get("group") == group_index:
                return selection

        return None

    def _get_selected_values(self, group_index):
        selection = self._get_selection(group_index)

        if not isinstance(selection, dict):
            return []

        values = selection.get("values", [])

        if not isinstance(values, list):
            return []

        return [
            value
            for value in values
            if isinstance(value, str)
        ]

    def _normalize_alternative_index(self, group_index):
        value = self.selected_alternatives.get(
            int(group_index)
        )

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _required_values_for_option(self, option):
        """
        Devuelve cuántas selecciones reales requiere una opción.

        item       -> 1
        category   -> choose
        multiple   -> suma de sus componentes
        """

        if not isinstance(option, dict):
            return 0

        option_type = option.get("type")

        if option_type == "item":
            return 1

        if option_type == "category":
            try:
                return max(
                    1,
                    int(option.get("choose", 1) or 1),
                )
            except (TypeError, ValueError):
                return 1

        if option_type == "multiple":
            items = option.get("items", [])

            if not isinstance(items, list):
                return 0

            total = 0

            for item in items:
                total += self._required_values_for_option(
                    item
                )

            return total

        return 0

    def _required_values_for_group(self, group_index):
        group = self._get_group(group_index)

        if not isinstance(group, dict):
            return 0

        group_type = group.get("type")

        if group_type == "alternatives":
            alternatives = group.get("options", [])

            if not isinstance(alternatives, list):
                return 0

            alternative_index = (
                self._normalize_alternative_index(
                    group_index
                )
            )

            if (
                alternative_index is None
                or not (
                    0 <= alternative_index
                    < len(alternatives)
                )
            ):
                return 0

            return self._required_values_for_option(
                alternatives[alternative_index]
            )

        if group_type == "category":
            try:
                return max(
                    1,
                    int(group.get("choose", 1) or 1),
                )
            except (TypeError, ValueError):
                return 1

        return 0

    def _is_group_complete(self, group_index):
        group = self._get_group(group_index)

        if not isinstance(group, dict):
            return False

        group_type = group.get("type")

        if group_type == "alternatives":
            alternative_index = (
                self._normalize_alternative_index(
                    group_index
                )
            )

            if alternative_index is None:
                return False

        required = self._required_values_for_group(
            group_index
        )

        if required <= 0:
            return False

        selected = self._get_selected_values(
            group_index
        )

        return len(selected) >= required

    def _all_groups_complete(self):
        if not self.equipment_options:
            return False

        for index in range(
            1,
            len(self.equipment_options) + 1,
        ):
            if not self._is_group_complete(index):
                return False

        return True

    def _persist_state(self):
        creation_state = campaign_state.get(
            "character_creation"
        )

        if not isinstance(creation_state, dict):
            creation_state = {}

        creation_state.update({
            "user_id": str(self.user_id),
            "proposal": deepcopy(self.proposal),
            "equipment_options": deepcopy(
                self.equipment_options
            ),
            "selected_choices": deepcopy(
                self.selections
            ),
            "selected_alternatives": deepcopy(
                self.selected_alternatives
            ),
            "current_group": self.current_group + 1,
            "stage": "manual",
        })

        campaign_state["character_creation"] = (
            creation_state
        )

        persist()

    def _replace_player_selection(self, group_index, values):
        """
        Reemplaza las elecciones realizadas por el jugador en
        un grupo, conservando los elementos fijos que la
        alternativa haya agregado automáticamente.
        """

        group_index = int(group_index)

        existing = self._get_selection(group_index)

        existing_values = []

        if isinstance(existing, dict):
            old_values = existing.get("values", [])

            if isinstance(old_values, list):
                existing_values = [
                    value
                    for value in old_values
                    if isinstance(value, str)
                ]

        # Obtener los elementos fijos de la alternativa actual.
        fixed_values = []

        group = self._get_group(group_index)

        if isinstance(group, dict):
            if group.get("type") == "alternatives":
                alternatives = group.get("options", [])

                alternative = (
                    self._normalize_alternative_index(
                        group_index
                    )
                )

                if (
                    isinstance(alternatives, list)
                    and alternative is not None
                    and 0 <= alternative < len(alternatives)
                ):
                    fixed_values = (
                        EquipmentAlternativeSelect
                        ._collect_fixed_values(
                            alternatives[alternative]
                        )
                    )

        # Solo conservar automáticamente los elementos fijos.
        preserved = [
            value
            for value in existing_values
            if value in fixed_values
        ]

        # Agregar las nuevas elecciones del jugador.
        for value in values:
            if (
                isinstance(value, str)
                and value not in preserved
            ):
                preserved.append(value)

        if existing is None:
            existing = {
                "group": group_index,
                "values": [],
            }
            self.selections.append(existing)

        existing["values"] = preserved

        alternative = (
            self._normalize_alternative_index(
                group_index
            )
        )

        if alternative is not None:
            existing["alternative"] = alternative

        if (
            group_index == self.current_group + 1
            and self._is_group_complete(group_index)
        ):
            if (
                self.current_group
                < len(self.equipment_options) - 1
            ):
                self.current_group += 1

        self._persist_state()

    def _find_item_name(self, idx):
        """
        Busca el nombre visible correspondiente a un ID interno
        dentro de las opciones de equipamiento.
        """

        def search(option):
            if not isinstance(option, dict):
                return None

            if option.get("idx") == idx:
                name = option.get("name")

                if name:
                    return str(name)

            option_type = option.get("type")

            if option_type in ("multiple", "alternatives"):
                items = option.get("items")

                if not isinstance(items, list):
                    items = option.get("options", [])

                if isinstance(items, list):
                    for item in items:
                        result = search(item)

                        if result:
                            return result

            if option_type in ("category", "choice"):
                items = option.get("options", [])

                if isinstance(items, list):
                    for item in items:
                        result = search(item)

                        if result:
                            return result

            return None

        for group in self.equipment_options:
            result = search(group)

            if result:
                return result

        return str(idx)

    def _display_selection_value(self, value):
        if not isinstance(value, str):
            return str(value)

        # Categorías internas -> nombres visibles.
        category_names = {
            "martial-weapons": "Armas marciales",
            "simple-weapons": "Armas simples",
            "martial-melee-weapons": (
                "Armas marciales cuerpo a cuerpo"
            ),
            "martial-ranged-weapons": (
                "Armas marciales a distancia"
            ),
            "simple-melee-weapons": (
                "Armas simples cuerpo a cuerpo"
            ),
            "simple-ranged-weapons": (
                "Armas simples a distancia"
            ),
        }

        if value in category_names:
            return category_names[value]

        # Traducciones de nombres de armas cuyo dataset
        # actualmente entrega el nombre en inglés.
        equipment_names = {
            "battleaxe": "Hacha de batalla",
            "flail": "Mangual",
            "glaive": "Glaive",
            "greataxe": "Hacha a dos manos",
            "greatsword": "Espadón",
            "halberd": "Alabarda",
            "lance": "Lanza de caballería",
            "longsword": "Espada larga",
            "maul": "Maza a dos manos",
            "morningstar": "Lucero del alba",
            "pike": "Pica",
            "rapier": "Estoque",
            "scimitar": "Cimitarra",
            "shortsword": "Espada corta",
            "trident": "Tridente",
            "war-pick": "Pico de guerra",
            "warhammer": "Martillo de guerra",
            "whip": "Látigo",
            "blowgun": "Cerbatana",
            "crossbow-hand": "Ballesta de mano",
            "crossbow-heavy": "Ballesta pesada",
            "longbow": "Arco largo",
            "net": "Red",
        }

        if value in equipment_names:
            return equipment_names[value]

        return self._find_item_name(value)

    def register_selection(self, group_index, values):
        """
        Registra componentes fijos de una alternativa.

        Las elecciones realizadas mediante los selectores del
        jugador utilizan _replace_player_selection(), para poder
        cambiar una elección sin acumular la anterior.
        """

        group_index = int(group_index)

        existing = self._get_selection(group_index)

        if existing is None:
            existing = {
                "group": group_index,
                "values": [],
            }
            self.selections.append(existing)

        current_values = existing.get("values", [])

        if not isinstance(current_values, list):
            current_values = []

        for value in values:
            if (
                isinstance(value, str)
                and value not in current_values
            ):
                current_values.append(value)

        existing["values"] = current_values

        alternative = (
            self._normalize_alternative_index(
                group_index
            )
        )

        if alternative is not None:
            existing["alternative"] = alternative

        if (
            group_index == self.current_group + 1
            and self._is_group_complete(group_index)
        ):
            if (
                self.current_group
                < len(self.equipment_options) - 1
            ):
                self.current_group += 1

        self._persist_state()


    def build_embed(self):
        total = len(self.equipment_options)

        if not total:
            return discord.Embed(
                title="Creación de personaje",
                description=(
                    "No hay opciones de equipamiento "
                    "disponibles."
                ),
            )

        if self.current_group < 0:
            self.current_group = 0

        if self.current_group >= total:
            self.current_group = total - 1

        group = self.equipment_options[
            self.current_group
        ]

        description = ""

        if isinstance(group, dict):
            description = str(
                group.get("description") or ""
            ).strip()

        # Una vez elegida una alternativa, la descripción original
        # deja de ser útil porque repetiría la misma información.
        current_group_number = self.current_group + 1

        selected_alternative = (
            self._normalize_alternative_index(
                current_group_number
            )
        )

        if (
            isinstance(group, dict)
            and group.get("type") == "alternatives"
            and selected_alternative is not None
        ):
            alternatives = group.get("options", [])

            if (
                isinstance(alternatives, list)
                and 0 <= selected_alternative < len(alternatives)
            ):
                alternative = alternatives[selected_alternative]

                alternative_text = (
                    EquipmentAlternativeSelect
                    ._describe_option(alternative)
                )

                description = (
                    f"**Elegiste:** {alternative_text}"
                )

        if not description:
            description = (
                "Elegí el equipamiento para este grupo."
            )

        embed = discord.Embed(
            title="Equipamiento inicial",
            description=description,
        )

        embed.add_field(
            name="Progreso",
            value=(
                f"Grupo **{current_group_number} "
                f"de {total}**"
            ),
            inline=False,
        )

        current_complete = self._is_group_complete(
            current_group_number
        )

        if current_complete:
            embed.add_field(
                name="Estado",
                value="✅ Grupo completado",
                inline=False,
            )
        else:
            required = self._required_values_for_group(
                current_group_number
            )

            selected = self._get_selected_values(
                current_group_number
            )

            remaining = max(
                0,
                required - len(selected),
            )

            if remaining > 0:
                embed.add_field(
                    name="Falta",
                    value=(
                        f"Elegir **{remaining}** "
                        f"elemento"
                        f"{'s' if remaining != 1 else ''}."
                    ),
                    inline=False,
                )

        # Mostrar los grupos anteriores como resumen compacto.
        previous_lines = []

        for selection in sorted(
            self.selections,
            key=lambda item: item.get("group", 0),
        ):
            group_number = selection.get("group")

            if not isinstance(group_number, int):
                continue

            if group_number >= current_group_number:
                continue

            values = selection.get("values", [])

            if not isinstance(values, list) or not values:
                continue

            names = [
                self._display_selection_value(value)
                for value in values
                if isinstance(value, str)
            ]

            if names:
                previous_lines.append(
                    f"Grupo {group_number}: "
                    + ", ".join(names)
                )

        if previous_lines:
            embed.add_field(
                name="Equipamiento elegido",
                value="\n".join(previous_lines)[:1024],
                inline=False,
            )

        return embed


    def rebuild(self):
        self.clear_items()

        if not self.equipment_options:
            return

        if self.current_group < 0:
            self.current_group = 0

        if self.current_group >= len(
            self.equipment_options
        ):
            self.current_group = (
                len(self.equipment_options) - 1
            )

        group = self.equipment_options[
            self.current_group
        ]

        if not isinstance(group, dict):
            return

        group_type = group.get("type")
        choose = group.get("choose", 1)

        if group_type == "alternatives":
            alternatives = group.get(
                "options",
                [],
            )

            if not isinstance(alternatives, list):
                alternatives = []

            selected_alternative = (
                self._normalize_alternative_index(
                    self.current_group + 1
                )
            )

            # Mostrar A/B solamente antes de elegir.
            # Después queda únicamente lo que el jugador
            # todavía debe seleccionar.
            if (
                selected_alternative is None
                and len(alternatives) > 1
            ):
                alternative_select = (
                    EquipmentAlternativeSelect(
                        self,
                        self.current_group + 1,
                        alternatives,
                    )
                )

                self.add_item(
                    alternative_select
                )

            if (
                selected_alternative is not None
                and 0 <= selected_alternative
                < len(alternatives)
            ):
                selected = alternatives[
                    selected_alternative
                ]

                if isinstance(selected, dict):
                    self._add_option_components(
                        self.current_group + 1,
                        selected,
                    )

            elif len(alternatives) == 1:
                self._add_option_components(
                    self.current_group + 1,
                    alternatives[0],
                )

        elif group_type == "category":
            options = group.get(
                "options",
                [],
            )

            if isinstance(options, list):
                select = EquipmentChoiceSelect(
                    self,
                    self.current_group + 1,
                    options,
                    choose=choose,
                )

                self.add_item(select)

        current_complete = (
            self._is_group_complete(
                self.current_group + 1
            )
        )

        previous_button = discord.ui.Button(
            label="Anterior",
            style=discord.ButtonStyle.secondary,
            disabled=(
                self.current_group == 0
            ),
            row=4,
        )

        next_button = discord.ui.Button(
            label="Siguiente",
            style=discord.ButtonStyle.primary,
            disabled=(
                self.current_group
                >= len(self.equipment_options) - 1
                or not current_complete
            ),
            row=4,
        )

        confirm_button = discord.ui.Button(
            label="Confirmar",
            style=discord.ButtonStyle.success,
            disabled=not self._all_groups_complete(),
            row=4,
        )

        cancel_button = discord.ui.Button(
            label="Cancelar",
            style=discord.ButtonStyle.danger,
            row=4,
        )

        previous_button.callback = (
            self.previous_callback
        )

        next_button.callback = (
            self.next_callback
        )

        confirm_button.callback = (
            self.confirm_callback
        )

        cancel_button.callback = (
            self.cancel_callback
        )

        self.add_item(previous_button)
        self.add_item(next_button)
        self.add_item(confirm_button)
        self.add_item(cancel_button)

    def _add_option_components(
        self,
        group_index,
        option,
    ):
        if not isinstance(option, dict):
            return

        option_type = option.get("type")

        if option_type == "item":
            idx = option.get("idx")
            name = option.get("name") or idx

            if isinstance(idx, str):
                self.add_item(
                    EquipmentChoiceSelect(
                        self,
                        group_index,
                        [{
                            "type": "item",
                            "idx": idx,
                            "name": name,
                        }],
                        choose=1,
                    )
                )

        elif option_type == "category":
            options = option.get(
                "options",
                [],
            )

            if isinstance(options, list):
                self.add_item(
                    EquipmentChoiceSelect(
                        self,
                        group_index,
                        [{
                            "type": "category",
                            "category": option.get(
                                "category"
                            ),
                            "choose": option.get(
                                "choose",
                                1,
                            ),
                            "options": options,
                        }],
                        choose=option.get(
                            "choose",
                            1,
                        ),
                    )
                )

        elif option_type == "multiple":
            items = option.get(
                "items",
                [],
            )

            if not isinstance(items, list):
                return

            # Si todos los componentes son objetos simples,
            # se presentan juntos en un único selector.
            simple_items = []

            for item in items:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "item"
                    and isinstance(
                        item.get("idx"),
                        str,
                    )
                ):
                    simple_items.append({
                        "type": "item",
                        "idx": item.get("idx"),
                        "name": (
                            item.get("name")
                            or item.get("idx")
                        ),
                    })

            if (
                simple_items
                and len(simple_items) == len(items)
            ):
                self.add_item(
                    EquipmentChoiceSelect(
                        self,
                        group_index,
                        simple_items,
                        choose=len(simple_items),
                    )
                )
                return

            # Si hay componentes mixtos, cada componente
            # conserva su propio selector.
            for item in items:
                self._add_option_components(
                    group_index,
                    item,
                )

    async def previous_callback(
        self,
        interaction,
    ):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        if self.current_group > 0:
            self.current_group -= 1

        creation_state = campaign_state.get(
            "character_creation"
        )

        if isinstance(creation_state, dict):
            creation_state["current_group"] = (
                self.current_group + 1
            )
            creation_state["stage"] = "manual"
            creation_state["channel_id"] = str(
                interaction.channel.id
            )
            creation_state["message_id"] = str(
                interaction.message.id
            )
            creation_state["guild_id"] = str(
                getattr(
                    interaction.guild,
                    "id",
                    "",
                )
            )

            persist()

        self.rebuild()

        await interaction.response.edit_message(
            embed=self.build_embed(),
            view=self,
        )

    async def next_callback(
        self,
        interaction,
    ):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        # No permitir saltar un grupo incompleto.
        if not self._is_group_complete(
            self.current_group + 1
        ):
            await interaction.response.send_message(
                "Primero completá todas las elecciones "
                "de este grupo.",
                ephemeral=True,
            )
            return

        if (
            self.current_group
            < len(self.equipment_options) - 1
        ):
            self.current_group += 1

        creation_state = campaign_state.get(
            "character_creation"
        )

        if isinstance(creation_state, dict):
            creation_state["current_group"] = (
                self.current_group + 1
            )
            creation_state["stage"] = "manual"
            creation_state["channel_id"] = str(
                interaction.channel.id
            )
            creation_state["message_id"] = str(
                interaction.message.id
            )
            creation_state["guild_id"] = str(
                getattr(
                    interaction.guild,
                    "id",
                    "",
                )
            )

            persist()

        self.rebuild()

        await interaction.response.edit_message(
            embed=self.build_embed(),
            view=self,
        )

    async def confirm_callback(
        self,
        interaction,
    ):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        # La confirmación depende de la completitud real
        # de cada grupo, no de la cantidad de registros.
        if not self._all_groups_complete():
            await interaction.response.send_message(
                "Todavía faltan elecciones por completar.",
                ephemeral=True,
            )
            return

        equipment_choices = []

        for selection in sorted(
            self.selections,
            key=lambda item: item.get(
                "group",
                0,
            ),
        ):
            values = selection.get(
                "values",
                [],
            )

            if not isinstance(values, list):
                continue

            for value in values:
                if (
                    isinstance(value, str)
                    and value not in equipment_choices
                ):
                    equipment_choices.append(value)

        if not equipment_choices:
            await interaction.response.send_message(
                "No se encontraron selecciones válidas.",
                ephemeral=True,
            )
            return

        proposal = deepcopy(
            self.proposal
        )

        proposal["equipment_mode"] = "manual"
        proposal["equipment_choices"] = (
            equipment_choices
        )

        character, error = (
            create_character_in_campaign(
                user_id=self.user_id,
                name=proposal.get("name"),
                race=proposal.get("race"),
                character_class=proposal.get(
                    "class"
                ),
                level=proposal.get("level"),
                abilities=proposal.get(
                    "abilities"
                ),
                skills=proposal.get("skills"),
                saving_throws=proposal.get(
                    "saving_throws"
                ),
                equipment_mode="manual",
                equipment_choices=equipment_choices,
            )
        )

        if character is None:
            await interaction.response.send_message(
                f"No pude crear la ficha: {error}",
                ephemeral=True,
            )
            return

        campaign_state[
            "character_creation"
        ] = None

        persist()

        self.stop()

        # Actualizar inmediatamente los paneles/fichas
        # de Discord después de crear el personaje.
        try:
            if interaction.guild is not None:
                await update_panels(
                    interaction.guild
                )
        except Exception as exc:
            print(
                "[ERROR update_panels tras "
                f"creación manual]: {exc}"
            )

        identity = character["identity"]
        abilities = character["abilities"]

        await interaction.response.edit_message(
            content=(
                f"**Ficha creada:** "
                f"{identity['name']}\n"
                f"**Raza:** "
                f"{identity['race']}\n"
                f"**Clase:** "
                f"{identity['class']}\n"
                f"**Nivel:** "
                f"{identity['level']}\n"
                f"**Características:** "
                f"FUE {abilities['strength']} · "
                f"DES {abilities['dexterity']} · "
                f"CON {abilities['constitution']} · "
                f"INT {abilities['intelligence']} · "
                f"SAB {abilities['wisdom']} · "
                f"CAR {abilities['charisma']}"
            ),
            embed=None,
            view=None,
        )

    async def cancel_callback(
        self,
        interaction,
    ):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "Esta selección pertenece a otro jugador.",
                ephemeral=True,
            )
            return

        self.stop()

        await interaction.response.edit_message(
            content="Creación de personaje cancelada.",
            view=None,
        )

    async def interaction_check(
        self,
        interaction,
    ):
        return interaction.user.id == self.user_id

    async def on_timeout(self):
        """
        Desactiva los componentes cuando expira la sesión.
        """
        for item in self.children:
            item.disabled = True

        if self.message:
            try:
                await self.message.edit(
                    view=self
                )
            except Exception:
                pass



def get_sheets_channel(guild):
    """Encuentra el canal destinado a fichas/botín."""
    if not guild:
        return None
    for ch in guild.text_channels:
        name = ch.name.lower()
        if "ficha" in name or "botin" in name or "botín" in name:
            return ch
    return None


def build_panel_contents():
    """Construye los paneles visuales directamente desde campaign_state."""
    chars = campaign_state.get("characters", {})

    fichas = []

    for name, c in chars.items():
        if not isinstance(c, dict):
            continue

        identity = c.get("identity", {})
        combat = c.get("combat", {})
        proficiency = c.get("proficiency", {})

        if not isinstance(identity, dict):
            identity = {}

        if not isinstance(combat, dict):
            combat = {}

        if not isinstance(proficiency, dict):
            proficiency = {}

        display_name = identity.get("name") or name
        race = identity.get("race") or "Raza no definida"
        character_class = identity.get("class") or "Clase no definida"
        level = identity.get("level")

        level_text = f"Nivel {level}" if level is not None else "Nivel no definido"

        # Compatibilidad con fichas antiguas.
        hp_current = combat.get("hp_current")
        hp_max = combat.get("hp_max")

        if hp_current is None or hp_max is None:
            old_hp = c.get("hp", "?")
            if isinstance(old_hp, str) and "/" in old_hp:
                hp_current, hp_max = old_hp.split("/", 1)
            else:
                hp_current = old_hp
                hp_max = "?"

        ac = combat.get("ac")
        if ac is None:
            ac = c.get("ac", "?")

        initiative = combat.get("initiative", 0)
        speed = combat.get("speed", 30)

        conditions = combat.get("conditions", [])
        if isinstance(conditions, list) and conditions:
            status = ", ".join(str(condition) for condition in conditions)
        else:
            status = c.get("status", "Sano")

        abilities = c.get("abilities", {})
        if not isinstance(abilities, dict):
            abilities = {}

        def ability_mod(score):
            if score is None:
                return "?"
            try:
                return f"{(int(score) - 10) // 2:+d}"
            except (TypeError, ValueError):
                return "?"

        skills = proficiency.get("skills", [])
        if not isinstance(skills, list):
            skills = []

        skills_text = ", ".join(
            str(skill) for skill in skills if str(skill).strip()
        ) or "Ninguna"

        fichas.append(
            f"⚔️ **{display_name.upper()}**\n"
            f"_{race} • {character_class} • {level_text}_\n"
            f"❤️ **{hp_current} / {hp_max} PG**   "
            f"🛡️ **CA {ac}**\n"
            f"🏃 Iniciativa **{initiative:+}** • Velocidad **{speed} pies**\n"
            f"✨ Estado: **{status}**\n"
            f"\n"
            f"📊 **Características**\n"
            f"💪 Fue {ability_mod(abilities.get('strength'))}   "
            f"🏹 Des {ability_mod(abilities.get('dexterity'))}   "
            f"❤️ Con {ability_mod(abilities.get('constitution'))}\n"
            f"🧠 Int {ability_mod(abilities.get('intelligence'))}   "
            f"👁️ Sab {ability_mod(abilities.get('wisdom'))}   "
            f"🗣️ Car {ability_mod(abilities.get('charisma'))}\n"
            f"\n"
            f"🎯 **Competencias:** {skills_text}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

    if not fichas:
        fichas = ["• Sin personajes registrados."]

    loot = campaign_state.get("loot", {})
    if not isinstance(loot, dict):
        loot = {}

    botin = [
        f"🪙 **Monedas**",
        f"• {loot.get('po', 0)} po | {loot.get('pp', 0)} pp | {loot.get('pc', 0)} pc",
        "",
        f"🎒 **Objetos y consumibles**",
        f"• {', '.join(loot.get('items', [])) if isinstance(loot.get('items', []), list) and loot.get('items') else 'Ninguno'}",
    ]

    facciones = []

    for name, faction in campaign_state.get("factions", {}).items():
        if not isinstance(faction, dict):
            continue

        facciones.append(
            f"🏛️ **{name}** — "
            f"{faction.get('attitude', 'Neutral')} "
            f"({faction.get('score', 0)}/5)"
        )

    if not facciones:
        facciones = ["• Ninguna registrada."]

    quests = campaign_state.get("quests", {})
    if not isinstance(quests, dict):
        quests = {}

    misiones = [
        "🎯 **MISIÓN PRINCIPAL**",
        f"• {quests.get('main', 'Sin objetivo principal.')}",
        "",
        "⚔️ **MISIONES ACTIVAS**",
    ]

    active_quests = quests.get("active", [])
    if isinstance(active_quests, list) and active_quests:
        for quest in active_quests:
            if isinstance(quest, dict):
                misiones.append(
                    f"• **{quest.get('title', '?')}** — "
                    f"{quest.get('objective', '?')}"
                )
            else:
                misiones.append(f"• {quest}")
    else:
        misiones.append("• Ninguna.")

    clues = quests.get("clues", [])
    if isinstance(clues, list) and clues:
        misiones.extend(["", "🔎 **PISTAS**"])
        for clue in clues:
            misiones.append(f"• {clue}")

    return {
        "fichas": (
            "📜 **FICHAS DE PERSONAJES**\n\n"
            + "\n\n".join(fichas)
        ),
        "botin": (
            "💰 **BOTÍN Y TESORO GRUPAL**\n\n"
            + "\n".join(botin)
        ),
        "facciones": (
            "🏛️ **FACCIONES Y REPUTACIÓN**\n\n"
            + "\n".join(facciones)
        ),
        "misiones": (
            "📜 **MISIONES Y OBJETIVOS**\n\n"
            + "\n".join(misiones)
        ),
    }


def panel_state():
    """Sección persistente donde guardamos canal e IDs de los paneles."""
    return campaign_state.setdefault(
        "discord_panels",
        {
            "channel_id": None,
            "message_ids": {
                "fichas": None,
                "botin": None,
                "facciones": None,
                "misiones": None,
            },
        },
    )


async def reset_discord_panels(guild):
    """Borra únicamente los cuatro paneles del bot al iniciar una campaña nueva."""
    global sheet_messages
    channel = get_sheets_channel(guild)
    if not channel:
        return
    headings = (
        "📊 **FICHAS DE PERSONAJES**",
        "💰 **BOTÍN Y TESORO GRUPAL**",
        "🏛️ **FACCIONES Y REPUTACIÓN**",
        "📜 **MISIONES Y OBJETIVOS**",
    )
    deleted = 0
    try:
        async for m in channel.history(limit=200):
            if m.author == bot.user and any(m.content.startswith(h) for h in headings):
                try:
                    await m.delete()
                    deleted += 1
                except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                    pass
        sheet_messages = {}
        print(f"[PANELES] Nueva campaña: {deleted} panel(es) anterior(es) eliminados.")
    except Exception as exc:
        print(f"[PANELES] No se pudieron limpiar los paneles anteriores: {exc}")


async def setup_sheets_channel(guild):
    """
    Recupera los cuatro paneles existentes o crea únicamente los que falten.

    Ya no depende del orden de los últimos mensajes del canal.
    """
    global sheet_messages

    channel = get_sheets_channel(guild)
    if not channel:
        print(f"[AVISO] No se encontró canal de fichas en {guild.name}.")
        return None

    print(f"[OK] Canal de fichas detectado: #{channel.name}")

    try:
        ps = panel_state()
        ps["channel_id"] = channel.id
        saved_ids = ps.setdefault("message_ids", {})
        found = {}

        # Primero usamos los IDs persistidos.
        for key, message_id in list(saved_ids.items()):
            if not message_id:
                continue
            try:
                found[key] = await channel.fetch_message(int(message_id))
            except (discord.NotFound, discord.HTTPException, ValueError, TypeError):
                saved_ids[key] = None

        # Si faltan paneles, buscamos por título entre los mensajes recientes.
        if len(found) < 4:
            recent = [m async for m in channel.history(limit=100)]
            headings = {
                "fichas": "📊 **FICHAS DE PERSONAJES**",
                "botin": "💰 **BOTÍN Y TESORO GRUPAL**",
                "facciones": "🏛️ **FACCIONES Y REPUTACIÓN**",
                "misiones": "📜 **MISIONES Y OBJETIVOS**",
            }
            for m in recent:
                if m.author != bot.user:
                    continue
                for key, heading in headings.items():
                    if key not in found and m.content.startswith(heading):
                        found[key] = m
                        saved_ids[key] = m.id
                        break

        # Creamos solo los que realmente no existan.
        contents = build_panel_contents()
        for key in ("fichas", "botin", "facciones", "misiones"):
            if key not in found:
                found[key] = await channel.send(contents[key][:2000])
                saved_ids[key] = found[key].id
                print(f"[PANELES] Creado panel: {key}")

        sheet_messages = found
        persist()
        print("[OK] 4 paneles conectados y sincronizados.")
        return channel

    except Exception as exc:
        print(f"[ERROR setup_sheets_channel]: {exc}")
        return None


async def update_panels(guild=None):
    """
    Sincroniza los cuatro mensajes con campaign_state.

    Se ejecuta después de cada acción y al iniciar el bot.
    Si un panel fue borrado, lo recrea automáticamente.
    """
    global sheet_messages

    if guild is None:
        if sheet_messages:
            guild = next(
                (getattr(m, "guild", None) for m in sheet_messages.values() if m),
                None,
            )
        if guild is None:
            return

    if len(sheet_messages) < 4:
        await setup_sheets_channel(guild)

    channel = get_sheets_channel(guild)
    if not channel:
        return

    contents = build_panel_contents()
    ps = panel_state()
    saved_ids = ps.setdefault("message_ids", {})
    changed = False

    for key, content in contents.items():
        msg = sheet_messages.get(key)

        # Intentar recuperar el mensaje por ID si la caché se perdió.
        if msg is None and saved_ids.get(key):
            try:
                msg = await channel.fetch_message(int(saved_ids[key]))
                sheet_messages[key] = msg
            except (discord.NotFound, discord.HTTPException, ValueError, TypeError):
                msg = None

        # Si fue borrado, recrearlo.
        if msg is None:
            try:
                msg = await channel.send(content[:2000])
                sheet_messages[key] = msg
                saved_ids[key] = msg.id
                changed = True
                print(f"[PANELES] Recreado panel: {key}")
            except Exception as exc:
                print(f"[PANELES] No se pudo crear {key}: {exc}")
            continue

        try:
            new_content = content[:2000]

            # Evita ediciones innecesarias y, cuando cambia el estado,
            # modifica exactamente el mensaje existente.
            if msg.content != new_content:
                await msg.edit(content=new_content)
                changed = True
                print(f"[PANELES] Actualizado: {key}")

        except discord.NotFound:
            try:
                new_msg = await channel.send(content[:2000])
                sheet_messages[key] = new_msg
                saved_ids[key] = new_msg.id
                changed = True
                print(f"[PANELES] Mensaje borrado; recreado: {key}")
            except Exception as exc:
                print(f"[PANELES] Error recreando {key}: {exc}")

        except Exception as exc:
            print(f"[PANELES] {key}: {exc}")

    if changed:
        persist()


# MULTIMEDIA
# ============================================================

async def send_generated_image(guild, image_prompt):
    """Genera una imagen con Pollinations, sin consumir tokens de Groq/Gemini."""
    if not guild:
        return False
    channel = None
    for ch in guild.text_channels:
        name = ch.name.lower()
        if any(k in name for k in ["imagen", "imágenes", "imagenes", "mapa", "arte", "foto"]):
            channel = ch
            break
    if not channel:
        print("[IMAGEN] No encontré un canal de imágenes/mapas.")
        return False

    clean_prompt = image_prompt.strip()
    enhanced_prompt = (
        f"{clean_prompt}, masterwork dark fantasy tabletop RPG illustration, "
        "D&D 5e-inspired fantasy concept art, cinematic composition, atmospheric lighting, "
        "detailed environment, coherent character design, realistic textures, no text, no watermark"
    )
    encoded_prompt = urllib.parse.quote(enhanced_prompt)
    seed = random.randint(1, 9999999)
    url = (
        f"https://gen.pollinations.ai/image/{encoded_prompt}"
        f"?model={urllib.parse.quote(POLLINATIONS_IMAGE_MODEL)}&width=1024&height=576&seed={seed}"
    )
    headers = {}
    if POLLINATIONS_API_KEY:
        headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"

    try:
        timeout = aiohttp.ClientTimeout(total=90)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    print(f"[IMAGEN] Pollinations HTTP {resp.status}: {body[:300]}")
                    return False
                img_bytes = await resp.read()
                await channel.send(
                    f"🖼️ **Escena:** *{clean_prompt[:1000]}*",
                    file=discord.File(io.BytesIO(img_bytes), filename="escena.png"),
                )
                print(f"[OK] Imagen generada con Pollinations ({POLLINATIONS_IMAGE_MODEL}) y enviada a #{channel.name}.")
                return True
    except Exception as exc:
        print(f"[AVISO Imagen]: {exc}")
        return False


def build_automatic_image_prompt(message_text):
    """Construye el prompt visual sin llamar a ningún LLM."""
    campaign = campaign_state.get("campaign", {})
    scene = campaign.get("current_scene") or "una escena de aventura fantástica"
    location = campaign.get("location") or "un lugar de fantasía"
    weather = campaign.get("weather") or "atmósfera dramática"
    time_of_day = campaign.get("time") or "hora indeterminada"
    action = message_text.strip()[:500]
    return (
        f"{scene}. Location: {location}. Weather/atmosphere: {weather}. Time: {time_of_day}. "
        f"The adventurers are currently doing: {action}. "
        "Show the current moment as a wide cinematic scene, with no written labels or UI."
    )


# ============================================================
# PROMPT DEL DM
# ============================================================

def build_system_prompt():
    # Este bloque es deliberadamente estático para favorecer prompt caching.
    return """Eres el Dungeon Master de élite de una campaña de D&D 5e para dos jugadores.
Tu trabajo es dirigir un mundo persistente, no controlar a los jugadores.
Tu prosa es inmersiva, cinematográfica y concisa: normalmente 1-4 párrafos.

REGLAS: 
1. Nunca decidas acciones, pensamientos, palabras, emociones o decisiones de los jugadores.
2. Respeta continuidad de HP, CA, inventario, NPC, facciones, misiones, lugares y combate.
3. Los NPC tienen objetivos, memoria, miedos, lealtades y conocimiento limitado.
4. No regales información que el personaje no podría conocer.
5. Las consecuencias deben ser justas y mecánicamente justificables.
6. No reinicies el mundo al cambiar de escena.
7. Saqueo, recursos y relojes deben ser coherentes.
8. Si una acción requiere una prueba de habilidad, identifica la habilidad y establece una CD razonable según la dificultad de la situación.

PRUEBAS DE HABILIDAD:
Cuando una acción requiera una prueba de habilidad, emite EXACTAMENTE:
<<<CHECK:Percepción|15>>>

Usa el nombre de la habilidad en español y una CD numérica.
NO incluyas modificadores en <<<CHECK>>>.
Python determina el personaje, la característica asociada, la competencia y el modificador.
Dice Golem determina exclusivamente el resultado del d20.
Python calcula el total y decide éxito o fallo.
Nunca inventes un resultado de dados ni un modificador.

TIRADAS DE SALVACIÓN:
Cuando una acción o efecto requiera una tirada de salvación, emite EXACTAMENTE:
<<<SAVE:Fuerza|15>>>

Usa el nombre de la característica en español y una CD numérica.
Características válidas: Fuerza, Destreza, Constitución, Inteligencia, Sabiduría y Carisma.
NO incluyas modificadores en <<<SAVE>>>.
Python determina el personaje, la característica, el modificador, la competencia y las condiciones.
Dice Golem determina exclusivamente el resultado del d20.
Python calcula el total y decide éxito o fallo.
Nunca inventes un resultado de dados ni un modificador.

ATAQUES:
Cuando un personaje jugador declare que quiere atacar a un objetivo durante un combate activo, emite exactamente un marcador:

<<<ATTACK:Arma|Objetivo>>>

Ejemplo:
<<<ATTACK:Espada larga|Goblin>>>

REGLAS DE ATTACK:
- Arma y objetivo deben corresponder a lo declarado por el jugador.
- No inventes el resultado del ataque.
- No inventes el d20.
- No inventes la CA del objetivo.
- No calcules ni inventes modificadores.
- No calcules daño.
- No determines crítico.
- No determines ventaja o desventaja.
- No determines resistencias, inmunidades o vulnerabilidades.
- Python resuelve toda la mecánica del ataque.
- Dice Golem proporciona todos los resultados aleatorios.
- Si no existe un combate activo o la acción no corresponde a un ataque, no emitas ATTACK.

DADOS LEGACY:
<<<ROLL:1d20+5|motivo|CD 16>>> sigue disponible para casos que todavía no estén migrados al sistema de pruebas de habilidad.

MULTIMEDIA: puedes emitir <<<IMAGEN: prompt detallado en inglés dark fantasy>>> si una imagen narrativa aporta valor.
No emitas <<<VOZ:...>>>.

ESTADO: si cambió algo importante, emite <<<CAMPAÑA_JSON>>> con solo las claves modificadas y <<<FIN_CAMPAÑA_JSON>>>.
No uses null para borrar información existente.

El contexto dinámico que acompaña a esta instrucción es la fuente de verdad para la escena actual.
"""

# ============================================================
# LLM ROUTER
# ============================================================

def _header_int(headers, name, default=None):
    try:
        value = headers.get(name)
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _header_float(headers, name, default=None):
    try:
        value = headers.get(name)
        if value is None:
            return default
        m = re.match(r"([0-9.]+)", str(value))
        return float(m.group(1)) if m else default
    except (TypeError, ValueError):
        return default


def estimate_tokens(messages):
    # Estimación conservadora; el servidor es la autoridad final.
    chars = 0
    for m in messages:
        chars += len(str(m.get("content", ""))) + 40
    return max(1, chars // 4)


def groq_can_try(model, estimated_tokens):
    info = groq_limits.get(model, {})
    if groq_daily_used(model) + estimated_tokens > GROQ_DAILY_BUDGET_PER_MODEL:
        print(f"[GROQ] Saltando {model}: presupuesto diario local agotado ({groq_daily_used(model)}/{GROQ_DAILY_BUDGET_PER_MODEL}).")
        return False
    remaining = info.get("remaining_tokens")
    remaining_requests = info.get("remaining_requests")
    if remaining is not None and remaining < estimated_tokens + GROQ_SAFE_TPM_RESERVE:
        return False
    if remaining_requests is not None and remaining_requests <= 1:
        return False
    return True


def update_groq_limits(model, headers):
    groq_limits[model] = {
        "remaining_tokens": _header_int(headers, "x-ratelimit-remaining-tokens"),
        "limit_tokens": _header_int(headers, "x-ratelimit-limit-tokens"),
        "reset_tokens": headers.get("x-ratelimit-reset-tokens"),
        "remaining_requests": _header_int(headers, "x-ratelimit-remaining-requests"),
        "limit_requests": _header_int(headers, "x-ratelimit-limit-requests"),
        "reset_requests": headers.get("x-ratelimit-reset-requests"),
        "updated_at": time.time(),
    }


async def groq_request(messages, model, reasoning=DEFAULT_REASONING, image_data=None, max_tokens=None):
    if not GROQ_API_KEY:
        return None, "sin_clave"

    output_budget = max_tokens or MAX_OUTPUT_TOKENS
    estimated = estimate_tokens(messages) + output_budget
    if not groq_can_try(model, estimated):
        print(f"[GROQ] Saltando {model}: presupuesto insuficiente según headers.")
        return None, "budget"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7 if "qwen/" in model else 0.6,
        "max_tokens": output_budget,
        "reasoning_effort": reasoning,
    }
    if model == "openai/gpt-oss-120b":
        payload["include_reasoning"] = False

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=35) as resp:
                update_groq_limits(model, resp.headers)
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content and content.strip():
                        usage = data.get("usage", {}) or {}
                        used_tokens = int(usage.get("total_tokens") or (int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)))
                        record_groq_usage(model, used_tokens)
                        print(f"[GROQ] {model}: in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')} día={groq_daily_used(model)}/{GROQ_DAILY_BUDGET_PER_MODEL} rem/min={groq_limits.get(model,{}).get('remaining_tokens','?')}")
                        return content.strip(), None
                    return None, "respuesta_vacia"
                err = await resp.text()
                print(f"[GROQ {model} -> {resp.status}] {err[:500]}")
                return None, str(resp.status)
    except Exception as exc:
        print(f"[GROQ {model}] excepción: {exc}")
        return None, "exception"


async def openai_compatible_request(provider, base_url, api_key, model, messages, timeout=35, temperature=0.7, max_tokens=None):
    if not api_key or not base_url:
        return None, "sin_clave"
    payload={"model":model,"messages":messages,"temperature":temperature,"max_tokens":max_tokens or min(MAX_OUTPUT_TOKENS,1800)}
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{base_url}/chat/completions",headers=headers,json=payload,timeout=timeout) as resp:
                if resp.status==200:
                    data=await resp.json(); text=data.get("choices",[{}])[0].get("message",{}).get("content","")
                    if text and text.strip():
                        record_ai_usage(provider,model,data.get("usage",{}) or {},ok=True)
                        return text.strip(),None
                err=await resp.text(); record_ai_usage(provider,model,ok=False,error=f"HTTP {resp.status}: {err[:200]}")
                print(f"[{provider} {model} -> {resp.status}] {err[:500]}"); return None,str(resp.status)
    except Exception as exc:
        record_ai_usage(provider,model,ok=False,error=exc); print(f"[{provider} {model}] excepción: {exc}"); return None,"exception"

async def cerebras_request(messages, reasoning=DEFAULT_REASONING):
    return await openai_compatible_request("cerebras","https://api.cerebras.ai/v1",CEREBRAS_API_KEY,CEREBRAS_MODEL,messages,timeout=35,temperature=0.7,max_tokens=MAX_OUTPUT_TOKENS)

async def nvidia_request(messages, reasoning=DEFAULT_REASONING):
    return await openai_compatible_request("nvidia",NVIDIA_BASE_URL,NVIDIA_API_KEY,NVIDIA_MODEL,messages,timeout=45,temperature=0.6,max_tokens=MAX_OUTPUT_TOKENS)

async def openrouter_request(messages, reasoning=DEFAULT_REASONING):
    return await openai_compatible_request("openrouter",OPENROUTER_BASE_URL,OPENROUTER_API_KEY,OPENROUTER_MODEL,messages,timeout=45,temperature=0.7,max_tokens=MAX_OUTPUT_TOKENS)


async def gemini_request(system_prompt, contents):
    if not GEMINI_API_KEYS:
        return None

    global gemini_key_index
    attempts = max(2, len(GEMINI_API_KEYS) * 2)

    async with aiohttp.ClientSession() as session:
        for _ in range(attempts):
            key = GEMINI_API_KEYS[gemini_key_index % len(GEMINI_API_KEYS)]
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={key}"
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": MAX_OUTPUT_TOKENS},
            }
            try:
                async with session.post(url, json=payload, timeout=35) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            text = "".join(p.get("text", "") for p in parts if p.get("text"))
                            if text.strip():
                                gem_usage = data.get("usageMetadata", {}) or {}
                                record_ai_usage("gemini", GEMINI_MODEL, {
                                    "prompt_tokens": gem_usage.get("promptTokenCount", 0),
                                    "completion_tokens": gem_usage.get("candidatesTokenCount", 0),
                                    "total_tokens": gem_usage.get("totalTokenCount", 0),
                                }, ok=True)
                                return text.strip()
                    err = await resp.text()
                    print(f"[GEMINI {GEMINI_MODEL} -> {resp.status}] {err[:500]}")
                    if resp.status in {429, 503}:
                        gemini_key_index = (gemini_key_index + 1) % len(GEMINI_API_KEYS)
                        await asyncio.sleep(0.7)
                    else:
                        # No insistimos indefinidamente con una clave/modelo que devuelve 4xx.
                        gemini_key_index = (gemini_key_index + 1) % len(GEMINI_API_KEYS)
                        await asyncio.sleep(0.5)
            except Exception as exc:
                print(f"[GEMINI] excepción: {exc}")
                gemini_key_index = (gemini_key_index + 1) % len(GEMINI_API_KEYS)
                await asyncio.sleep(0.7)
    return None


async def local_llm_request(messages):
    """Gemma local vía LiteRT-LM/OpenAI-compatible. No consume cuota de API.

    Si el servidor local no está levantado, falla rápido y el router continúa
    con los proveedores externos. Esto permite iniciar el bot sin Gemma.
    """
    if not LOCAL_LLM_URL:
        return None
    headers = {"Content-Type": "application/json"}
    if LOCAL_LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LOCAL_LLM_API_KEY}"
    payload = {
        "model": LOCAL_LLM_MODEL,
        "messages": messages,
        "temperature": 0.65,
        "max_tokens": min(MAX_OUTPUT_TOKENS, 1400),
    }
    try:
        timeout = aiohttp.ClientTimeout(total=LOCAL_LLM_TIMEOUT, connect=2, sock_read=max(3, LOCAL_LLM_TIMEOUT - 2))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{LOCAL_LLM_URL}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status != 200:
                    err = (await resp.text())[:300]
                    record_ai_usage("local", LOCAL_LLM_MODEL, ok=False, error=f"HTTP {resp.status}: {err}")
                    print(f"[LOCAL {LOCAL_LLM_MODEL} -> {resp.status}] {err}")
                    return None
                data = await resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not text or not text.strip():
                    record_ai_usage("local", LOCAL_LLM_MODEL, ok=False, error="respuesta_vacia")
                    return None
                record_ai_usage("local", LOCAL_LLM_MODEL, data.get("usage", {}) or {}, ok=True)
                usage = data.get("usage", {}) or {}
                print(
                    f"[LOCAL] {LOCAL_LLM_MODEL}: "
                    f"in={usage.get('prompt_tokens','?')} out={usage.get('completion_tokens','?')}"
                )
                return text.strip()
    except Exception as exc:
        # No es un error fatal: el router seguirá con OpenRouter/Gemini.
        record_ai_usage("local", LOCAL_LLM_MODEL, ok=False, error=exc)
        print(f"[LOCAL] Gemma no disponible: {exc}")
        return None


async def local_model_available():
    """Comprueba rápidamente si LiteRT-LM está escuchando, sin generar texto."""
    if not LOCAL_LLM_URL:
        return False
    try:
        timeout = aiohttp.ClientTimeout(total=2.5, connect=1.0, sock_read=1.5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{LOCAL_LLM_URL}/models") as resp:
                return resp.status == 200
    except Exception:
        return False


def classify_local_task(text):
    t=text.lower()
    if any(k in t for k in ("cuánto","cuanto","suma","resta","multiplica","divide","calcula","oro","monedas","daño","danio","vida","hp","experiencia","xp")): return "utility"
    if any(k in t for k in ("inventario","qué tengo","que tengo","mi ficha","mis objetos")): return "state_lookup"
    if any(k in t for k in ("ataco","ataque","combate","iniciativa","salvación","tirada","hechizo","conjuro","crítico","critico")): return "combat"
    return "narrative"

def local_math(text):
    import ast, operator
    m=re.search(r"(?<![A-Za-z])([0-9][0-9\s+\-*/().%]*[0-9])",text)
    if not m: return None
    expr=m.group(1).replace(" ","")
    if len(expr)>80 or not re.fullmatch(r"[0-9+*/().%-]+",expr): return None
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Mod:operator.mod}
    try:
        tree=ast.parse(expr,mode="eval")
        def ev(n):
            if isinstance(n,ast.Expression): return ev(n.body)
            if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)): return n.value
            if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
                v=ev(n.operand); return v if isinstance(n.op,ast.UAdd) else -v
            if isinstance(n,ast.BinOp) and type(n.op) in ops:
                a,b=ev(n.left),ev(n.right)
                if isinstance(n.op,(ast.Div,ast.Mod)) and b==0: raise ZeroDivisionError
                return ops[type(n.op)](a,b)
            raise ValueError
        r=ev(tree); r=int(r) if isinstance(r,float) and r.is_integer() else r
        return f"{expr} = {r}"
    except Exception: return None

async def local_task_model(messages):
    if not LOCAL_TASK_URL: return None
    result,_=await openai_compatible_request("local-task",LOCAL_TASK_URL,LOCAL_TASK_API_KEY,LOCAL_TASK_MODEL,messages,timeout=LOCAL_TASK_TIMEOUT,temperature=0.1,max_tokens=500)
    return result

async def ask_text_llm(system_prompt, context_message, recent_history, current_user_message):
    messages=[{"role":"system","content":system_prompt},{"role":"user","content":context_message}]+recent_history+[{"role":"user","content":current_user_message}]
    task=classify_local_task(current_user_message)
    if task=="utility":
        calc=local_math(current_user_message)
        if calc:
            record_ai_usage("local", "python-math", {"total_tokens": 0}, ok=True)
            return f"🧮 {calc}","local-math"
    reasoning="high" if task=="combat" else DEFAULT_REASONING
    for model in GROQ_MODELS:
        raw,error=await groq_request(messages,model,reasoning)
        if raw: return raw,f"groq:{model}"
        if error=="401": break
    raw,_=await cerebras_request(messages,reasoning)
    if raw: return raw,f"cerebras:{CEREBRAS_MODEL}"
    raw,_=await nvidia_request(messages,reasoning)
    if raw: return raw,f"nvidia:{NVIDIA_MODEL}"
    # Gemma 4 E2B local es un fallback real, sin cuota de API. Solo se usa
    # para texto: las imágenes siguen entrando por Gemini multimodal.
    if await local_model_available():
        local=await local_llm_request(messages)
        if local: return local,f"local:{LOCAL_LLM_MODEL}"
    raw,_=await openrouter_request(messages,reasoning)
    if raw: return raw,f"openrouter:{OPENROUTER_MODEL}"
    contents=[{"role":"user" if m.get("role")=="user" else "model","parts":[{"text":str(m.get("content",""))}]} for m in messages[1:]]
    fallback=await gemini_request(system_prompt,contents)
    if fallback: return fallback,f"gemini:{GEMINI_MODEL}"
    return None,None

# ============================================================
# CLASIFICADOR DETERMINISTA
# ============================================================

def deterministic_state_query(user_id, user_name, message_text):
    """
    Responde consultas simples directamente desde campaign_state.
    Devuelve None si la consulta no es suficientemente clara.
    """
    text = str(message_text or "").strip().lower()
    if not text:
        return None

    characters = campaign_state.get("characters", {})
    loot = campaign_state.get("loot", {})
    quests = campaign_state.get("quests", {})
    campaign = campaign_state.get("campaign", {})

    # Identificar automáticamente el personaje mediante el ID de Discord.
    character_name = None
    character = None

    characters = campaign_state.get("characters", {})
    bindings = campaign_state.get("player_bindings", {})

    if isinstance(characters, dict) and isinstance(bindings, dict):
        character_name = bindings.get(str(user_id))
        if character_name:
            character = characters.get(character_name)

    # Si el personaje asociado ya no existe, no inventar una asociación.
    if character_name and not isinstance(character, dict):
        character_name = None
        character = None

    # HP / PG
    if re.search(r"\b(cuanta|cuánta|cuantos|cuántos)\b.*\b(vida|hp|pg|puntos de golpe)\b", text):
        if character is None:
            return None
        return f"**{character_name}** tiene **{character.get('hp', '?')} PG**."

    if re.search(r"\b(hp|pg)\b", text) and re.search(r"\b(tengo|tiene|queda|quedan)\b", text):
        if character is None:
            return None
        return f"**{character_name}** tiene **{character.get('hp', '?')} PG**."

    # CA
    if re.search(r"\b(ca|clase de armadura)\b", text):
        if character is None:
            return None
        return f"**{character_name}** tiene **CA {character.get('ac', '?')}**."

    # Estado
    if re.search(r"\b(estado|condición|condicion)\b", text):
        if character is None:
            return None
        return f"El estado de **{character_name}** es **{character.get('status', '?')}**."

    # Inventario
    if re.search(r"\b(inventario|llevo|tengo encima|mis objetos)\b", text):
        if character is None:
            return None
        items = character.get("inventory", [])
        if not isinstance(items, list):
            items = []
        display_items = []

        for item in items:
            if isinstance(item, str):
                item_text = item.strip()
                if item_text:
                    display_items.append(item_text)

            elif isinstance(item, dict):
                name = item.get("name")
                quantity = item.get("quantity", 1)

                if not isinstance(name, str) or not name.strip():
                    name = item.get("idx")

                if not isinstance(name, str) or not name.strip():
                    continue

                if not isinstance(quantity, int) or quantity <= 0:
                    continue

                display_items.append(
                    f"{name.strip()} x{quantity}"
                )

        return (
            f"**Inventario de {character_name}:** "
            + (", ".join(display_items) if display_items else "vacío.")
        )

    # Botín grupal
    if re.search(r"\b(botín|botin|tesoro|monedas|dinero)\b", text):
        return (
            f"**Botín grupal:** "
            f"{loot.get('po', 0)} po, "
            f"{loot.get('pp', 0)} pp y "
            f"{loot.get('pc', 0)} pc."
        )

    # Misión principal
    if re.search(r"\b(misión principal|mision principal|objetivo principal)\b", text):
        return f"**Misión principal:** {quests.get('main', 'Sin objetivo principal.')}"

    # Misiones activas
    if re.search(r"\b(misiones activas|misiones actuales)\b", text):
        active = quests.get("active", [])
        if not isinstance(active, list) or not active:
            return "No hay misiones activas registradas."

        lines = []
        for quest in active:
            if isinstance(quest, dict):
                lines.append(
                    f"• {quest.get('title', '?')} — {quest.get('objective', '?')}"
                )
            else:
                lines.append(f"• {quest}")

        return "**Misiones activas:**\n" + "\n".join(lines)

    # Ubicación / escena
    if re.search(r"\b(dónde estamos|donde estamos|ubicación|ubicacion|lugar actual|escena actual)\b", text):
        location = campaign.get("location") or campaign.get("current_scene")
        if not location:
            return "No hay una ubicación actual registrada."
        return f"**Ubicación actual:** {location}"

    # Clima
    if re.search(r"\b(clima|tiempo hace|cómo está el tiempo|como esta el tiempo)\b", text):
        weather = campaign.get("weather")
        return f"**Clima:** {weather or 'No registrado.'}"

    # Hora
    if re.search(r"\b(hora|qué hora|que hora|momento del día|momento del dia)\b", text):
        current_time = campaign.get("time")
        return f"**Hora/momento:** {current_time or 'No registrado.'}"

    return None

# ============================================================
# FLUJO DE UNA ACCIÓN
# ============================================================
    global chat_history
async def ask_llm(user_id, user_name, message_text, attachments_data=None, channel=None):
    global chat_history
    has_images = bool(attachments_data)
    system_prompt = build_system_prompt()

    # --------------------------------------------------------
    # Creación/configuración de personaje.
    # El LLM propone; Python valida y registra.
    # --------------------------------------------------------
    if not has_images:
        creation_text = str(message_text or "").strip().lower()
        creation_request = (
            re.search(r"\b(crea|crear|creame|créame|inventá|inventa|inventame|inventar)\b", creation_text)
            and re.search(r"\b(ficha|personaje)\b", creation_text)
        )

        if creation_request:
            print(
                f"[PERSONAJE] Solicitud de creación de "
                f"{user_name}: {message_text}"
            )

            proposal = await generate_character_proposal(
                message_text
            )

            if proposal is None:
                return (
                    "No pude generar una propuesta de personaje válida. "
                    "Probemos de nuevo indicando al menos el concepto "
                    "que querés."
                )

            class_idx = normalize_class_idx(
                proposal.get("class")
            )

            if not class_idx:
                return (
                    "No pude identificar la clase del personaje "
                    "generado."
                )

            proficiencies = get_class_proficiencies(
                class_idx
            )

            equipment_options = (
                get_manual_starting_equipment_options(
                    class_idx,
                    proficiencies=proficiencies,
                )
            )

            campaign_state["character_creation"] = {
                "user_id": str(user_id),
                "guild_id": str(getattr(channel.guild, "id", "")),
                "channel_id": str(channel.id),
                "message_id": None,
                "stage": "mode",
                "proposal": deepcopy(proposal),
                "equipment_options": deepcopy(
                    equipment_options
                ),
                "selected_choices": [],
                "selected_alternatives": {},
                "current_group": 1,
            }

            persist()

            view = CharacterEquipmentModeView(
                user_id,
                proposal,
            )

            embed = discord.Embed(
                title="Creación de personaje",
                description=(
                    f"**{proposal.get('name')}**\n"
                    f"Raza: **{proposal.get('race')}**\n"
                    f"Clase: **{proposal.get('class')}**\n"
                    f"Nivel: **{proposal.get('level')}**\n\n"
                    "Elegí cómo querés obtener el "
                    "equipamiento inicial."
                ),
            )

            sent_message = await channel.send(
                embed=embed,
                view=view,
            )

            view.message = sent_message

            creation_state = campaign_state.get(
                "character_creation"
            )

            if isinstance(creation_state, dict):
                creation_state["guild_id"] = str(
                    getattr(channel.guild, "id", "")
                )
                creation_state["channel_id"] = str(channel.id)
                creation_state["message_id"] = str(
                    sent_message.id
                )
                creation_state["stage"] = "mode"
                persist()

            return None

    # Consultas simples de estado: responder sin gastar una llamada al LLM.
    if not has_images:
        deterministic_reply = deterministic_state_query(
    user_id,
    user_name,
    message_text,
)
        if deterministic_reply:
            return deterministic_reply
    texto_historial = f"[{user_name}]: {message_text}"
    if has_images:
        texto_historial += " [El jugador ha mostrado una imagen de referencia]"

    recent_history = compact_recent_history(chat_history)
    context_message = build_context_message(texto_historial)
    current_user_message = texto_historial

    # Imagen del jugador: Gemini recibe solo el contexto compacto + historial reciente.
    if has_images:
        print("[SISTEMA] Imagen detectada. Ruta Gemini multimodal.")
        gemini_contents = [
            {"role": "user", "parts": [{"text": context_message}]},
        ] + [
            {"role": ("user" if m.get("role") == "user" else "model"), "parts": [{"text": m.get("content", "")}] }
            for m in recent_history
        ]
        gemini_contents.append({"role": "user", "parts": [{"text": current_user_message}]})
        for img in attachments_data:
            gemini_contents[-1]["parts"].append({"inlineData": {"mimeType": img["mime_type"], "data": img["data"]}})
        raw_reply = await gemini_request(system_prompt, gemini_contents)
        model_used = f"gemini:{GEMINI_MODEL}" if raw_reply else None
    else:
        raw_reply, model_used = await ask_text_llm(system_prompt, context_message, recent_history, current_user_message)

    if not raw_reply:
        return "El Master intenta conectar con sus motores de narración, pero ninguno está disponible en este momento."

    # --------------------------------------------------------
    # Sistema de tiradas de salvación D&D.
    # --------------------------------------------------------
    save_request = parse_save_marker(raw_reply)

    if save_request and channel is not None:
        ability_name, dc = save_request

        try:
            save_result = await resolve_player_save(
                user_id,
                ability_name,
                dc,
                channel,
            )

            if save_result is None:
                raw_reply = (
                    "⚠️ **No puedo resolver esta tirada de salvación todavía.** "
                    "La ficha del personaje asociado a este jugador "
                    "no tiene configurados todos los datos mecánicos necesarios."
                )
            else:
                campaign_state["last_roll"] = {
                    "expression": "1d20",
                    "reason": (
                        f"Salvación {save_result['ability']} "
                        f"CD {save_result['dc']}"
                    ),
                    "target": f"CD {save_result['dc']}",
                    "result": {
                        "expression": "1d20",
                        "total": save_result["d20"],
                        "source": "Dice Golem",
                        "raw": "",
                    },
                    "save": save_result,
                    "at": time.time(),
                }

                if save_result["automatic_failure"]:
                    d20_text = "fallo automático"
                    total_text = "fallo automático"
                else:
                    d20_text = str(save_result["d20"])
                    total_text = str(save_result["total"])

                followup_context = (
                    context_message
                    + "\nRESULTADO DE TIRADA DE SALVACIÓN: "
                    + f"{save_result['character']} | "
                    + f"característica={save_result['ability']} | "
                    + f"d20={d20_text} | "
                    + f"modificador={save_result['modifier']} | "
                    + f"total={total_text} | "
                    + f"CD={save_result['dc']} | "
                    + (
                        "ÉXITO"
                        if save_result["success"]
                        else "FALLO"
                    )
                )

                correction_messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": followup_context,
                    },
                    {
                        "role": "assistant",
                        "content": raw_reply,
                    },
                    {
                        "role": "user",
                        "content": (
                            "Narra ahora la consecuencia de la tirada de "
                            "salvación usando exclusivamente el resultado "
                            "mecánico proporcionado. No vuelvas a pedir la "
                            "tirada ni inventes otro resultado."
                        ),
                    },
                ]

                corrected = None

                for model in GROQ_MODELS:
                    corrected, err = await groq_request(
                        correction_messages,
                        model,
                        "high",
                        max_tokens=900,
                    )
                    if corrected:
                        model_used = f"groq:{model}"
                        break

                if not corrected:
                    corrected, _ = await cerebras_request(
                        correction_messages,
                        "high",
                    )
                    if corrected:
                        model_used = f"cerebras:{CEREBRAS_MODEL}"

                if not corrected:
                    corrected, _ = await nvidia_request(
                        correction_messages,
                        "high",
                    )
                    if corrected:
                        model_used = f"nvidia:{NVIDIA_MODEL}"

                if not corrected and await local_model_available():
                    corrected = await local_llm_request(
                        correction_messages
                    )
                    if corrected:
                        model_used = f"local:{LOCAL_LLM_MODEL}"

                if not corrected:
                    corrected = await gemini_request(
                        system_prompt,
                        [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": followup_context}
                                ],
                            },
                            {
                                "role": "model",
                                "parts": [
                                    {"text": raw_reply}
                                ],
                            },
                            {
                                "role": "user",
                                "parts": [
                                    {
                                        "text": (
                                            "Narra la consecuencia con el "
                                            "resultado real de la tirada "
                                            "de salvación. No pidas otra "
                                            "tirada."
                                        )
                                    }
                                ],
                            },
                        ],
                    )
                    if corrected:
                        model_used = f"gemini:{GEMINI_MODEL}"

                if corrected:
                    raw_reply = corrected

        except Exception as exc:
            print(f"[SALVACIÓN] Error: {exc}")
            raw_reply = (
                "🎲 **Salvación pendiente:** no se pudo completar la "
                "tirada con Dice Golem. El Master no resolverá esta "
                "acción hasta obtener una tirada real."
            )

    # --------------------------------------------------------
    # Sistema de ataques D&D.
    # --------------------------------------------------------
    attack_request = parse_attack_marker(raw_reply)

    if attack_request and channel is not None:
        weapon_name, target_name = attack_request

        character_name = campaign_state.get("player_bindings", {}).get(str(user_id))

        if not character_name:
            raw_reply = (
                "⚠️ No hay un personaje asociado a este jugador. "
                "El ataque no se realizó."
            )
        else:
            weapon_idx = resolve_weapon_for_attack(
                character_name,
                weapon_name,
            )

            target_combatant_id = resolve_attack_target(target_name)

            if not weapon_idx:
                raw_reply = (
                    f"⚠️ El arma «{weapon_name}» no está disponible "
                    f"en el inventario de {character_name}. "
                    "El ataque no se realizó."
                )
            elif not target_combatant_id:
                raw_reply = (
                    f"⚠️ No pude identificar un único objetivo "
                    f"«{target_name}» entre los monstruos en combate. "
                    "El ataque no se realizó."
                )
            else:
                try:
                    attack_result = await resolve_character_attack_with_dice_golem(
                        character_name,
                        weapon_idx,
                        target_combatant_id,
                        channel,
                    )

                    if attack_result is None:
                        raw_reply = (
                            "⚠️ El ataque no pudo resolverse mecánicamente. "
                            "No se inventó ningún resultado de dados."
                        )
                    else:
                        campaign_state["last_roll"] = attack_result
                        raw_reply = build_attack_narration(
                            attack_result
                        )

                except Exception as exc:
                    print(f"[ATTACK] Error: {exc}")
                    raw_reply = (
                        "⚠️ El ataque no pudo resolverse mecánicamente. "
                        "No se inventó ningún resultado de dados."
                    )

    # --------------------------------------------------------
    # Sistema de pruebas de habilidad D&D.
    # --------------------------------------------------------
    check_request = parse_check_marker(raw_reply)

    if check_request and channel is not None:
        skill_name, dc = check_request

        try:
            check_result = await resolve_player_check(
                user_id,
                skill_name,
                dc,
                channel,
            )

            if check_result is None:
                raw_reply = (
                    "⚠️ **No puedo resolver esta prueba todavía.** "
                    "La ficha del personaje asociado a este jugador "
                    "no tiene configurados todos los datos mecánicos necesarios."
                )
            else:
                campaign_state["last_roll"] = {
                    "expression": "1d20",
                    "reason": f"{check_result['skill']} CD {check_result['dc']}",
                    "target": f"CD {check_result['dc']}",
                    "result": {
                        "expression": "1d20",
                        "total": check_result["d20"],
                        "source": "Dice Golem",
                        "raw": "",
                    },
                    "check": check_result,
                    "at": time.time(),
                }

                followup_context = (
                    context_message
                    + "\nRESULTADO DE PRUEBA DE HABILIDAD: "
                    + f"{check_result['character']} | "
                    + f"{check_result['skill']} | "
                    + f"d20={check_result['d20']} | "
                    + f"modificador={check_result['modifier']} | "
                    + f"total={check_result['total']} | "
                    + f"CD={check_result['dc']} | "
                    + (
                        "ÉXITO"
                        if check_result["success"]
                        else "FALLO"
                    )
                )

                correction_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": followup_context},
                    {"role": "assistant", "content": raw_reply},
                    {
                        "role": "user",
                        "content": (
                            "Narra ahora la consecuencia de la prueba usando "
                            "exclusivamente el resultado mecánico proporcionado. "
                            "No vuelvas a pedir la tirada ni inventes otro resultado."
                        ),
                    },
                ]

                corrected = None

                for model in GROQ_MODELS:
                    corrected, err = await groq_request(
                        correction_messages,
                        model,
                        "high",
                        max_tokens=900,
                    )
                    if corrected:
                        model_used = f"groq:{model}"
                        break

                if not corrected:
                    corrected, _ = await cerebras_request(
                        correction_messages,
                        "high",
                    )
                    if corrected:
                        model_used = f"cerebras:{CEREBRAS_MODEL}"

                if not corrected:
                    corrected, _ = await nvidia_request(
                        correction_messages,
                        "high",
                    )
                    if corrected:
                        model_used = f"nvidia:{NVIDIA_MODEL}"

                if not corrected and await local_model_available():
                    corrected = await local_llm_request(
                        correction_messages
                    )
                    if corrected:
                        model_used = f"local:{LOCAL_LLM_MODEL}"

                if not corrected:
                    corrected = await gemini_request(
                        system_prompt,
                        [
                            {
                                "role": "user",
                                "parts": [{"text": followup_context}],
                            },
                            {
                                "role": "model",
                                "parts": [{"text": raw_reply}],
                            },
                            {
                                "role": "user",
                                "parts": [{
                                    "text": (
                                        "Narra la consecuencia con el resultado "
                                        "real de la prueba. No pidas otra tirada."
                                    )
                                }],
                            },
                        ],
                    )
                    if corrected:
                        model_used = f"gemini:{GEMINI_MODEL}"

                if corrected:
                    raw_reply = corrected

        except Exception as exc:
            print(f"[CHECK] Error: {exc}")
            raw_reply = (
                "🎲 **Prueba pendiente:** no se pudo completar la prueba "
                "con Dice Golem. El Master no resolverá esta acción hasta "
                "obtener una tirada real."
            )

    # --------------------------------------------------------
    # Sistema de tiradas en dos fases.
    # --------------------------------------------------------
    roll_match = re.search(
        r"<<<ROLL:\s*([^|>]+)\s*\|\s*([^|>]+)(?:\|([^>]+))?>>>",
        raw_reply, re.IGNORECASE,
    )

    if roll_match and channel is not None:
        expression = roll_match.group(1).strip()
        reason = roll_match.group(2).strip()
        target = (roll_match.group(3) or "").strip()
        try:
            expression = validate_dice_expression(expression)
            result = await roll_with_dice_golem(channel, expression, reason)
            if result:
                campaign_state["last_roll"] = {"expression": expression, "reason": reason, "target": target, "result": result, "at": time.time()}
                followup_context = (
                    context_message
                    + "\nRESULTADO DE DADOS: "
                    + f"{result['expression']} => {result['total']}"
                )
                correction_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": followup_context},
                    {"role": "assistant", "content": raw_reply},
                    {"role": "user", "content": "Narra ahora la consecuencia usando exclusivamente el resultado real. No vuelvas a pedir la tirada."},
                ]
                corrected = None
                # Una sola corrección usando el router multinivel.
                corrected = None
                for model in GROQ_MODELS:
                    corrected, err = await groq_request(correction_messages, model, "high", max_tokens=900)
                    if corrected:
                        model_used = f"groq:{model}"; break
                if not corrected:
                    corrected, _ = await cerebras_request(correction_messages, "high")
                    if corrected: model_used = f"cerebras:{CEREBRAS_MODEL}"
                if not corrected:
                    corrected, _ = await nvidia_request(correction_messages, "high")
                    if corrected: model_used = f"nvidia:{NVIDIA_MODEL}"
                if not corrected and await local_model_available():
                    corrected = await local_llm_request(correction_messages)
                    if corrected: model_used = f"local:{LOCAL_LLM_MODEL}"
                if not corrected:
                    corrected = await gemini_request(system_prompt, [
                        {"role": "user", "parts": [{"text": followup_context}]},
                        {"role": "model", "parts": [{"text": raw_reply}]},
                        {"role": "user", "parts": [{"text": "Narra la consecuencia con el resultado real. No pidas otra tirada."}]},
                    ])
                    if corrected:
                        model_used = f"gemini:{GEMINI_MODEL}"
                if corrected:
                    raw_reply = corrected
        except Exception as exc:
            print(f"[DADOS] Error: {exc}")
            raw_reply = "🎲 **Tirada pendiente:** Dice Golem no devolvió un resultado válido. El Master no resolverá esta acción hasta obtener la tirada real."

    update_state_from_model(raw_reply)
    chat_history.append({"role": "user", "content": texto_historial})
    chat_history.append({"role": "assistant", "content": raw_reply})
    persist()
    print(f"[LLM] Motor final: {model_used}")
    return raw_reply

# ============================================================
# FLUJO DE UNA ACCIÓN
# ============================================================
# ============================================================
# PROCESAMIENTO DE RESPUESTA
# ============================================================

def strip_control_blocks(reply):
    reply = re.sub(r"<<<CAMPAÑA_JSON>>>.*?<<<FIN_CAMPAÑA_JSON>>>", "", reply, flags=re.DOTALL)
    reply = re.sub(r"<<<ROLL:.*?>>>", "", reply, flags=re.DOTALL)
    reply = re.sub(r"<<<ATTACK:.*?>>>", "", reply, flags=re.DOTALL)
    reply = re.sub(r"<<<CHECK:.*?>>>", "", reply, flags=re.DOTALL)
    reply = re.sub(r"<<<SAVE:.*?>>>", "", reply, flags=re.DOTALL)
    reply = re.sub(r"<<<VOZ:.*?>>>", "", reply, flags=re.DOTALL)
    return reply.strip()


async def process_reply(message, reply):
    # El LLM puede pedir una imagen explícita. Eso no consume tokens adicionales:
    # la generación se hace en Pollinations.
    match_img = re.search(r"<<<IMAGEN:(.*?)>>>", reply, re.DOTALL)
    image_sent = False
    if match_img:
        try:
            image_sent = await send_generated_image(message.guild, match_img.group(1).strip())
        except Exception as exc:
            print(f"[ERROR Imagen]: {exc}")
        reply = re.sub(r"<<<IMAGEN:.*?>>>", "", reply, flags=re.DOTALL)

    # Además, generamos una imagen automáticamente cada N acciones del grupo,
    # independientemente de que el LLM haya emitido el marcador.
    if not image_sent:
        try:
            campaign_state["image_counter"] = int(campaign_state.get("image_counter", 0)) + 1
        except (TypeError, ValueError):
            campaign_state["image_counter"] = 1

        if campaign_state["image_counter"] >= IMAGE_EVERY_ACTIONS:
            auto_prompt = build_automatic_image_prompt(message.clean_content)
            image_sent = await send_generated_image(message.guild, auto_prompt)
            if image_sent:
                campaign_state["image_counter"] = 0

    sanitize_campaign_state()
    await update_panels(message.guild)
    return strip_control_blocks(reply)


# ============================================================
# DISCORD
# ============================================================

# TEST_COMBAT_INITIATIVE se ejecuta desde el on_ready existente.



# EVENTOS DISCORD


async def recover_character_creation():
    """
    Recupera una creación de personaje pendiente después de un reinicio.

    Reconstruye la View según la etapa persistida y vuelve a conectarla
    al mensaje original. Si el mensaje fue eliminado, crea uno nuevo.
    """
    creation = campaign_state.get("character_creation")

    if not isinstance(creation, dict):
        return

    user_id = creation.get("user_id")
    stage = creation.get("stage", "mode")
    proposal = creation.get("proposal", {})

    if not isinstance(user_id, str) or not user_id.isdigit():
        return

    if not isinstance(proposal, dict):
        print("[CREACION] Estado pendiente sin proposal válida.")
        return

    guild_id = creation.get("guild_id")
    channel_id = creation.get("channel_id")
    message_id = creation.get("message_id")

    if not channel_id:
        print("[CREACION] No hay channel_id para recuperar la creación.")
        return

    try:
        channel = bot.get_channel(int(channel_id))

        if channel is None:
            channel = await bot.fetch_channel(int(channel_id))

    except (
        discord.NotFound,
        discord.Forbidden,
        discord.HTTPException,
        ValueError,
        TypeError,
    ) as exc:
        print(f"[CREACION] No se pudo recuperar el canal: {exc}")
        return

    message = None

    if message_id:
        try:
            message = await channel.fetch_message(
                int(message_id)
            )
        except (
            discord.NotFound,
            discord.HTTPException,
            ValueError,
            TypeError,
        ):
            message = None

    try:
        if stage == "mode":
            view = CharacterEquipmentModeView(
                int(user_id),
                proposal,
            )

            embed = discord.Embed(
                title="Creación de personaje",
                description=(
                    f"**{proposal.get('name')}**\n"
                    f"Raza: **{proposal.get('race')}**\n"
                    f"Clase: **{proposal.get('class')}**\n"
                    f"Nivel: **{proposal.get('level')}**\n\n"
                    "Elegí cómo querés obtener el "
                    "equipamiento inicial."
                ),
            )

            if message is not None:
                await message.edit(
                    content=None,
                    embed=embed,
                    view=view,
                )
                view.message = message

            else:
                message = await channel.send(
                    embed=embed,
                    view=view,
                )
                view.message = message

        elif stage == "preset":
            presets = creation.get("presets", [])

            if not isinstance(presets, list) or not presets:
                class_idx = normalize_class_idx(
                    proposal.get("class")
                )

                if not class_idx:
                    print(
                        "[CREACION] No se pudo recuperar la clase "
                        "para reconstruir presets."
                    )
                    return

                proficiencies = get_class_proficiencies(
                    class_idx
                )

                presets = get_starting_equipment_presets(
                    class_idx,
                    proficiencies=proficiencies,
                    preset_count=3,
                )

                creation["presets"] = deepcopy(presets)

            view = CharacterEquipmentPresetView(
                int(user_id),
                proposal,
                presets,
            )

            if message is not None:
                await message.edit(
                    content="Elegí un preset de equipamiento:",
                    embed=None,
                    view=view,
                )
                view.message = message

            else:
                message = await channel.send(
                    content="Elegí un preset de equipamiento:",
                    view=view,
                )
                view.message = message

        elif stage == "manual":
            equipment_options = creation.get(
                "equipment_options",
                [],
            )

            if not isinstance(equipment_options, list):
                equipment_options = []

            view = CharacterCreationView(
                int(user_id),
                equipment_options,
                proposal=proposal,
            )

            try:
                persisted_group = int(
                    creation.get("current_group", 1)
                )
            except (TypeError, ValueError):
                persisted_group = 1

            view.current_group = max(
                0,
                min(
                    persisted_group - 1,
                    max(0, len(equipment_options) - 1),
                ),
            )

            selections = creation.get(
                "selected_choices",
                [],
            )

            if isinstance(selections, list):
                view.selections = deepcopy(selections)
            else:
                view.selections = []

            alternatives = creation.get(
                "selected_alternatives",
                {},
            )

            if isinstance(alternatives, dict):
                normalized_alternatives = {}

                for key, value in alternatives.items():
                    try:
                        normalized_alternatives[int(key)] = int(value)
                    except (TypeError, ValueError):
                        continue

                view.selected_alternatives = (
                    normalized_alternatives
                )
            else:
                view.selected_alternatives = {}

            view.rebuild()

            if message is not None:
                await message.edit(
                    content="Elegí el equipamiento inicial:",
                    embed=view.build_embed(),
                    view=view,
                )
                view.message = message

            else:
                message = await channel.send(
                    content="Elegí el equipamiento inicial:",
                    embed=view.build_embed(),
                    view=view,
                )
                view.message = message

        else:
            print(
                f"[CREACION] Etapa desconocida: {stage}"
            )
            return

        creation["guild_id"] = str(
            guild_id
            or getattr(getattr(channel, "guild", None), "id", "")
        )
        creation["channel_id"] = str(channel.id)
        creation["message_id"] = str(message.id)

        if stage not in ("mode", "preset", "manual"):
            creation["stage"] = "mode"

        persist()

        print(
            f"[CREACION] Creación recuperada: "
            f"user={user_id} stage={stage} "
            f"message={message.id}"
        )

    except Exception as exc:
        print(
            f"[ERROR recover_character_creation]: {exc}"
        )


@bot.event
async def on_ready():
    print(f"[DISCORD] Conectado como {bot.user} (ID: {bot.user.id})")

    try:
        for guild in bot.guilds:
            print(f"[DISCORD] Servidor: {guild.name} ({guild.id})")
            await setup_sheets_channel(guild)
            await update_panels(guild)

        await recover_character_creation()

        print("[DISCORD] Inicialización completada.")
    except Exception as exc:
        print(f"[ERROR on_ready]: {exc}")



def normalize_combat_command_text(value):
    """Normaliza texto para reconocer órdenes mecánicas de combate."""
    if not isinstance(value, str):
        return ""

    value = value.strip().casefold()

    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
    }

    for source, target in replacements.items():
        value = value.replace(source, target)

    return re.sub(r"\s+", " ", value)


def resolve_monster_idx_from_command(text):
    """
    Convierte un nombre visible de monstruo a un idx real de Magical20.

    No inventa monstruos: cada candidato se valida mediante
    get_monster_combat_data().
    """
    normalized = normalize_combat_command_text(text)

    if not normalized:
        return None

    aliases = {
        "goblin": "goblin",
        "goblins": "goblin",
        "trasgo": "goblin",
        "trasgos": "goblin",
    }

    candidates = []

    for alias, monster_idx in aliases.items():
        if re.search(rf"\b{re.escape(alias)}\b", normalized):
            candidates.append(monster_idx)

    # También permitir que el texto sea directamente un idx.
    if normalized in aliases.values():
        candidates.append(normalized)

    seen = set()

    for monster_idx in candidates:
        if monster_idx in seen:
            continue

        seen.add(monster_idx)

        monster_data = get_monster_combat_data(monster_idx)

        if isinstance(monster_data, dict):
            return monster_idx

    return None


def find_combat_monster_by_idx(monster_idx):
    """Devuelve la instancia existente de un monstruo, si hay exactamente una."""
    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return None

    combatants = combat.get("combatants")

    if not isinstance(combatants, list):
        return None

    matches = [
        combatant
        for combatant in combatants
        if (
            isinstance(combatant, dict)
            and combatant.get("type") == "monster"
            and combatant.get("monster_idx") == monster_idx
        )
    ]

    if len(matches) == 1:
        return matches[0]

    return None


async def handle_direct_combat_command(user_id, text, channel):
    """
    Ejecuta órdenes explícitas de combate sin pasar por el LLM.

    Devuelve:
        str  -> la orden fue reconocida y procesada.
        None -> no era una orden directa de combate.
    """
    normalized = normalize_combat_command_text(text)

    if not normalized:
        return None

    start_match = re.search(
        r"\b(?:iniciar|inicia|iniciemos|empezar|empieza|comenzar|comienza)"
        r"\s+(?:un\s+)?combate\b",
        normalized,
    )

    add_match = re.search(
        r"\b(?:agrega|agregar|anade|anadir|añade|añadir|mete|meter)"
        r"\s+(?:un\s+)?(.+?)"
        r"\s+al\s+combate\b",
        normalized,
    )

    if not start_match and not add_match:
        return None

    monster_idx = resolve_monster_idx_from_command(normalized)

    if monster_idx is None:
        return (
            "⚠️ No pude identificar un monstruo válido en la orden de combate."
        )

    combat = campaign_state.get("combat")

    if not isinstance(combat, dict):
        return "⚠️ El estado de combate no está disponible."

    # --------------------------------------------------------
    # AGREGAR MONSTRUO
    # --------------------------------------------------------
    if add_match and not start_match:
        if combat.get("active") is True:
            return (
                "⚠️ El combate ya está activo. No se puede agregar "
                "un monstruo sin recalcular la iniciativa."
            )

        combatant = add_monster_to_combat(monster_idx)

        if not isinstance(combatant, dict):
            return (
                f"⚠️ No pude agregar el monstruo «{monster_idx}» "
                "al combate."
            )

        monster_data = get_monster_combat_data(monster_idx)
        monster_name = (
            monster_data.get("name", monster_idx)
            if isinstance(monster_data, dict)
            else monster_idx
        )

        persist()

        return (
            f"⚔️ **{monster_name} agregado al combate.**\n"
            "El combate todavía no comenzó. "
            "Podés iniciar el combate cuando estén preparados."
        )

    # --------------------------------------------------------
    # INICIAR COMBATE
    # --------------------------------------------------------
    character_name = campaign_state.get(
        "player_bindings", {}
    ).get(str(user_id))

    if not character_name:
        return (
            "⚠️ No hay un personaje asociado a este jugador. "
            "No puedo iniciar el combate."
        )

    if combat.get("active") is True:
        return "⚠️ Ya hay un combate activo."

    character = get_character_by_name(character_name)

    if character is None:
        return (
            f"⚠️ No encontré la ficha de {character_name}."
        )

    # Agregar el personaje si todavía no forma parte del combate.
    character_combatant_id = f"character:{character_name}"

    character_present = any(
        isinstance(combatant, dict)
        and combatant.get("id") == character_combatant_id
        for combatant in combat.get("combatants", [])
    )

    if not character_present:
        if add_character_to_combat(character_name) is None:
            return (
                f"⚠️ No pude agregar a {character_name} "
                "al combate."
            )

    # Agregar el monstruo si todavía no existe.
    existing_monster = find_combat_monster_by_idx(monster_idx)

    if existing_monster is None:
        if add_monster_to_combat(monster_idx) is None:
            return (
                f"⚠️ No pude agregar el monstruo «{monster_idx}» "
                "al combate."
            )

    # La iniciativa se resuelve exclusivamente mediante Dice Golem.
    result = await start_combat(channel)

    if not isinstance(result, dict):
        return (
            "⚠️ No se pudo iniciar el combate porque "
            "la iniciativa no pudo resolverse con Dice Golem. "
            "No se inventó ningún resultado."
        )

    persist()

    monster_data = get_monster_combat_data(monster_idx)

    monster_name = (
        monster_data.get("name", monster_idx)
        if isinstance(monster_data, dict)
        else monster_idx
    )

    turn = result.get("turn")
    round_number = result.get("round")

    if turn == character_combatant_id:
        turn_text = f"Es el turno de **{character_name}**."
    else:
        turn_text = "Es el turno del monstruo."

    return (
        f"⚔️ **Combate iniciado contra {monster_name}.**\n"
        f"Ronda: **{round_number}**.\n"
        f"{turn_text}"
    )


def build_attack_narration(attack_result):
    """
    Genera una narración final usando exclusivamente datos mecánicos reales.

    No tira dados, no modifica estado y no permite que el LLM invente
    el resultado del ataque.
    """
    if not isinstance(attack_result, dict):
        return "⚠️ El ataque fue resuelto, pero no hay resultado narrable."

    character = attack_result.get("character", "El personaje")
    weapon = attack_result.get("weapon", "arma")
    monster_idx = attack_result.get("monster", "monstruo")

    attack = attack_result.get("attack")
    damage = attack_result.get("damage")
    applied = attack_result.get("applied")

    if not isinstance(attack, dict):
        return "⚠️ El ataque no contiene un resultado mecánico válido."

    outcome = str(attack.get("outcome", "")).casefold()

    if outcome == "miss":
        return (
            f"⚔️ **{character}** ataca al **{monster_idx}** "
            f"con **{weapon}**, pero el ataque falla."
        )

    if not isinstance(damage, dict):
        return (
            f"⚔️ **{character}** impacta al **{monster_idx}** "
            f"con **{weapon}**."
        )

    damage_value = damage.get("damage", 0)

    try:
        damage_value = int(damage_value)
    except (TypeError, ValueError):
        damage_value = 0

    critical = outcome == "critical"

    if critical:
        opening = (
            f"💥 **Golpe crítico. {character}** impacta al "
            f"**{monster_idx}** con **{weapon}**."
        )
    else:
        opening = (
            f"⚔️ **{character}** impacta al **{monster_idx}** "
            f"con **{weapon}**."
        )

    if isinstance(applied, dict) and applied.get("dead") is True:
        return (
            f"{opening} Causa **{damage_value}** de daño. "
            f"El **{monster_idx}** cae derrotado."
        )

    return (
        f"{opening} Causa **{damage_value}** de daño."
    )


@bot.event
async def on_message(message):
    # Ignorar mensajes propios.
    if message.author == bot.user:
        return

    # Los mensajes de otros bots se ignoran salvo Dice Golem,
    # cuyos resultados son consumidos por el sistema de dados.
    if message.author.bot and str(message.author.id) != DICE_GOLEM_ID:
        return

    # Los mensajes de Dice Golem se manejan mediante el sistema
    # específico de espera de tiradas; no pasan por el LLM.
    if DICE_GOLEM_ID and str(message.author.id) == DICE_GOLEM_ID:
        return

    if not message.guild:
        return

    texto = (message.clean_content or "").strip()
    if not texto and not message.attachments:
        return

    attachments_data = []

    for attachment in message.attachments:
        content_type = (attachment.content_type or "").lower()
        if content_type.startswith("image/"):
            try:
                data = await attachment.read()
                attachments_data.append({
                    "filename": attachment.filename,
                    "content_type": content_type,
                    "data": data,
                })
            except Exception as exc:
                print(f"[ERROR Imagen adjunta]: {exc}")

    try:
        # Las órdenes explícitas de combate tienen prioridad sobre el LLM.
        # Así el Master no puede convertir una acción mecánica en narrativa
        # sin modificar realmente el estado de combate.
        reply = await handle_direct_combat_command(
            str(message.author.id),
            texto,
            message.channel,
        )

        if reply is None:
            reply = await ask_llm(
                user_id=str(message.author.id),
                user_name=message.author.display_name,
                message_text=texto,
                attachments_data=attachments_data or None,
                channel=message.channel,
            )

        if reply:
            reply = await process_reply(message, reply)
            if reply:
                await message.channel.send(reply[:2000])

    except Exception as exc:
        print(f"[ERROR on_message]: {exc}")
        try:
            await message.channel.send(
                "⚠️ Ocurrió un error procesando la acción."
            )
        except Exception:
            pass


# ============================================================
# ARRANQUE DEL BOT
# ============================================================

if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN no está definido.")

bot.run(DISCORD_TOKEN)
