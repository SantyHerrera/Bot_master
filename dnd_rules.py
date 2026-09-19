import json
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "dnd_rules.db"


def _connect():
    return sqlite3.connect(DB_PATH)


def _row_to_dict(cursor, row):
    if row is None:
        return None

    columns = [column[0] for column in cursor.description]
    data = dict(zip(columns, row))

    if "raw_json" in data and data["raw_json"]:
        try:
            data["raw"] = json.loads(data["raw_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            data["raw"] = None

    return data


def _get_one(query, params=()):
    conn = _connect()
    try:
        cursor = conn.execute(query, params)
        row = cursor.fetchone()
        return _row_to_dict(cursor, row)
    finally:
        conn.close()


def get_ability(idx):
    return _get_one(
        """
        SELECT *
        FROM abilities
        WHERE idx = ?
        """,
        (idx,),
    )


def get_skill(idx):
    return _get_one(
        """
        SELECT *
        FROM skills
        WHERE idx = ?
        """,
        (idx,),
    )


def get_class(idx):
    return _get_one(
        """
        SELECT *
        FROM classes
        WHERE idx = ?
        """,
        (idx,),
    )



def get_class_hit_die(class_idx):
    """
    Devuelve el dado de golpe de una clase según Magical20.
    Ejemplo: 8 para Bardo, 10 para Guerrero, 12 para Bárbaro.
    """
    if not isinstance(class_idx, str):
        return None

    class_idx = class_idx.strip().lower()

    character_class = get_class(class_idx)

    if not character_class:
        return None

    raw = character_class.get("raw")

    if not isinstance(raw, dict):
        return None

    hit_die = raw.get("hit_die")

    try:
        return int(hit_die)
    except (TypeError, ValueError):
        return None


def get_fixed_hp_gain(class_idx, constitution_score):
    """
    Calcula los HP ganados al subir un nivel usando el valor fijo
    del dado de golpe, según las reglas de D&D 5e.

    HP ganado = valor fijo del dado + modificador de Constitución.
    El mínimo de HP ganados por nivel es 1.
    """
    hit_die = get_class_hit_die(class_idx)

    if hit_die is None:
        return None

    constitution_modifier = get_ability_modifier(constitution_score)

    if constitution_modifier is None:
        return None

    fixed_value = (hit_die // 2) + 1
    hp_gain = fixed_value + constitution_modifier

    return max(1, hp_gain)


def get_rolled_hp_gain(class_idx, constitution_score, roll_result):
    """
    Calcula los HP ganados al subir de nivel usando el resultado
    de una tirada del dado de golpe.

    Dice Golem proporciona únicamente el resultado del dado.
    Python valida el resultado, suma Constitución y aplica el
    mínimo de 1 HP.
    """
    hit_die = get_class_hit_die(class_idx)

    if hit_die is None:
        return None

    constitution_modifier = get_ability_modifier(constitution_score)

    if constitution_modifier is None:
        return None

    try:
        roll_result = int(roll_result)
    except (TypeError, ValueError):
        return None

    if roll_result < 1 or roll_result > hit_die:
        return None

    hp_gain = roll_result + constitution_modifier

    return max(1, hp_gain)


def get_max_hp_at_level(class_idx, level, constitution_score):
    """
    Calcula los HP máximos de un personaje hasta un nivel concreto,
    usando el método fijo de aumento de HP.

    Nivel 1:
        dado de golpe máximo + modificador de Constitución.

    Cada nivel posterior:
        valor fijo del dado + modificador de Constitución.
    """
    if not isinstance(class_idx, str):
        return None

    try:
        level = int(level)
    except (TypeError, ValueError):
        return None

    if level < 1 or level > 20:
        return None

    level_one_hp = get_level_one_hp(
        class_idx,
        constitution_score,
    )

    if level_one_hp is None:
        return None

    hp = level_one_hp

    for _ in range(2, level + 1):
        hp_gain = get_fixed_hp_gain(
            class_idx,
            constitution_score,
        )

        if hp_gain is None:
            return None

        hp += hp_gain

    return hp







def get_racial_proficiencies(race_idx, subrace_idx=None):
    """
    Devuelve las competencias otorgadas por una raza y, opcionalmente,
    por su subraza, usando exclusivamente los datos de Magical20.
    """
    if not isinstance(race_idx, str):
        return []

    race_idx = race_idx.strip().lower()

    if not race_idx:
        return []

    race = get_race(race_idx)

    if not race:
        return []

    result = []

    def collect_starting_proficiencies(data):
        if not data:
            return

        raw = data.get("raw")

        if not isinstance(raw, dict):
            return

        starting = raw.get("starting_proficiencies", [])

        if not isinstance(starting, list):
            return

        for item in starting:
            if not isinstance(item, dict):
                continue

            idx = item.get("index")

            if isinstance(idx, str):
                result.append(idx)

    collect_starting_proficiencies(race)

    if subrace_idx is not None:
        if not isinstance(subrace_idx, str):
            return []

        subrace_idx = subrace_idx.strip().lower()

        if not subrace_idx:
            return []

        subrace = get_subrace(subrace_idx)

        if not subrace:
            return []

        raw = subrace.get("raw")

        if not isinstance(raw, dict):
            return []

        race_reference = raw.get("race")

        if not isinstance(race_reference, dict):
            return []

        if race_reference.get("index") != race_idx:
            return []

        collect_starting_proficiencies(subrace)

    return list(dict.fromkeys(result))

def is_weapon_proficient(
    class_idx,
    weapon_idx,
    race_idx=None,
    subrace_idx=None,
):
    """
    Determina si un personaje es competente con un arma.

    Se consideran:
    - competencias de clase;
    - competencias de raza/subraza;
    - competencia por categoría de arma;
    - competencia específica del arma.
    """
    if not isinstance(class_idx, str):
        return False

    if not isinstance(weapon_idx, str):
        return False

    class_idx = class_idx.strip().lower()
    weapon_idx = weapon_idx.strip().lower()

    if not class_idx or not weapon_idx:
        return False

    weapon = get_equipment(weapon_idx)

    if not weapon:
        return False

    raw = weapon.get("raw")

    if not isinstance(raw, dict):
        return False

    weapon_category = raw.get("weapon_category")

    if weapon_category == "Simple":
        category_proficiency = "simple-weapons"
    elif weapon_category == "Martial":
        category_proficiency = "martial-weapons"
    else:
        category_proficiency = None

    class_proficiencies = set(get_class_proficiencies(class_idx))

    racial_proficiencies = set(
        get_racial_proficiencies(
            race_idx,
            subrace_idx,
        )
    )

    all_proficiencies = class_proficiencies | racial_proficiencies

    if category_proficiency in all_proficiencies:
        return True

    for proficiency_idx in all_proficiencies:
        proficiency = get_proficiency(proficiency_idx)

        if not proficiency:
            continue

        proficiency_raw = proficiency.get("raw")

        if not isinstance(proficiency_raw, dict):
            continue

        reference = proficiency_raw.get("reference")

        if not isinstance(reference, dict):
            continue

        reference_idx = reference.get("index")

        if reference_idx == weapon_idx:
            return True

    return False








def resolve_attack_damage(
    weapon_idx,
    ability_scores,
    attack_result,
    damage_roll_result,
    damage_resistances=None,
    damage_vulnerabilities=None,
    damage_immunities=None,
):
    """
    Resuelve el daño de un ataque después de conocer el resultado
    de la tirada de ataque.

    El daño base se calcula con la tirada de daño y el modificador
    de característica correspondiente al arma.

    Después se aplican, en este orden:
    - inmunidad
    - resistencia
    - vulnerabilidad
    """

    if not isinstance(attack_result, dict):
        return None

    outcome = attack_result.get("outcome")

    if outcome not in ("hit", "critical", "miss"):
        return None

    if outcome == "miss":
        return {
            "weapon": weapon_idx,
            "damage_roll": 0,
            "ability_modifier": 0,
            "damage": 0,
            "damage_type": None,
            "critical": False,
            "damage_modifier": "none",
        }

    damage_data = get_weapon_damage_data(
        weapon_idx,
        ability_scores,
    )

    if not damage_data:
        return None

    try:
        damage_roll_result = int(damage_roll_result)
    except (TypeError, ValueError):
        return None

    if damage_roll_result < 0:
        return None

    ability_modifier = damage_data.get("ability_modifier")

    if ability_modifier is None:
        return None

    base_damage = damage_roll_result + ability_modifier

    modified_damage = apply_damage_modifiers(
        base_damage,
        damage_data["damage_type"],
        damage_resistances=damage_resistances,
        damage_vulnerabilities=damage_vulnerabilities,
        damage_immunities=damage_immunities,
    )

    if modified_damage is None:
        return None

    return {
        "weapon": damage_data["weapon"],
        "damage_roll": damage_roll_result,
        "ability_modifier": ability_modifier,
        "base_damage": base_damage,
        "damage": modified_damage["damage"],
        "damage_type": modified_damage["damage_type"],
        "critical": outcome == "critical",
        "damage_modifier": modified_damage["modifier"],
    }


def get_monster_attack_data(monster_idx, action_name):
    """
    Devuelve los datos mecánicos de un ataque de monstruo.

    Usa exclusivamente las acciones almacenadas en Magical20.
    Las acciones que no poseen attack_bonus no se consideran
    ataques normales.
    """
    if not isinstance(monster_idx, str):
        return None

    if not isinstance(action_name, str):
        return None

    monster_idx = monster_idx.strip().lower()
    action_name = action_name.strip().lower()

    if not monster_idx or not action_name:
        return None

    monster = get_monster(monster_idx)

    if not monster:
        return None

    raw = monster.get("raw")

    if not isinstance(raw, dict):
        return None

    actions = raw.get("actions", [])

    if not isinstance(actions, list):
        return None

    for action in actions:
        if not isinstance(action, dict):
            continue

        name = action.get("name")

        if not isinstance(name, str):
            continue

        if name.strip().lower() != action_name:
            continue

        if "attack_bonus" not in action:
            return None

        try:
            attack_bonus = int(action["attack_bonus"])
        except (TypeError, ValueError):
            return None

        damage_entries = action.get("damage", [])

        if not isinstance(damage_entries, list):
            damage_entries = []

        damages = []

        for damage in damage_entries:
            if not isinstance(damage, dict):
                continue

            damage_type = damage.get("damage_type")

            if not isinstance(damage_type, dict):
                continue

            damage_type_idx = damage_type.get("index")
            damage_dice = damage.get("damage_dice")

            if not isinstance(damage_type_idx, str):
                continue

            if not isinstance(damage_dice, str):
                continue

            damages.append({
                "damage_type": damage_type_idx,
                "damage_dice": damage_dice,
            })

        return {
            "monster": monster_idx,
            "action": name,
            "attack_bonus": attack_bonus,
            "damages": damages,
        }

    return None

def resolve_attack_roll(
    d20_result,
    attack_modifier,
    target_ac,
):
    """
    Resuelve una tirada de ataque de D&D 5e.

    Reglas:
    - 1 natural: fallo automático.
    - 20 natural: impacto crítico automático.
    - cualquier otro resultado: impacta si total >= CA.

    No calcula daño.
    """
    try:
        d20_result = int(d20_result)
        attack_modifier = int(attack_modifier)
        target_ac = int(target_ac)
    except (TypeError, ValueError):
        return None

    if d20_result < 1 or d20_result > 20:
        return None

    if target_ac < 0:
        return None

    total = d20_result + attack_modifier

    if d20_result == 1:
        outcome = "miss"
        critical = False
        hit = False

    elif d20_result == 20:
        outcome = "critical"
        critical = True
        hit = True

    elif total >= target_ac:
        outcome = "hit"
        critical = False
        hit = True

    else:
        outcome = "miss"
        critical = False
        hit = False

    return {
        "d20": d20_result,
        "attack_modifier": attack_modifier,
        "total": total,
        "target_ac": target_ac,
        "hit": hit,
        "critical": critical,
        "outcome": outcome,
    }



def resolve_initiative_roll(
    d20_result,
    initiative_modifier,
):
    """
    Resuelve una tirada de iniciativa.

    El d20 es proporcionado por Dice Golem.
    Python suma el modificador de iniciativa.

    No ordena combatientes ni gestiona turnos.
    """
    try:
        d20_result = int(d20_result)
        initiative_modifier = int(initiative_modifier)
    except (TypeError, ValueError):
        return None

    if d20_result < 1 or d20_result > 20:
        return None

    total = d20_result + initiative_modifier

    return {
        "d20": d20_result,
        "initiative_modifier": initiative_modifier,
        "total": total,
    }


def apply_damage_modifiers(
    damage,
    damage_type,
    damage_resistances=None,
    damage_vulnerabilities=None,
    damage_immunities=None,
):
    """
    Aplica inmunidad, resistencia y vulnerabilidad al daño.

    - Inmunidad: daño = 0
    - Resistencia: daño reducido a la mitad
    - Vulnerabilidad: daño duplicado
    """

    try:
        damage = int(damage)
    except (TypeError, ValueError):
        return None

    if damage < 0:
        return None

    if not isinstance(damage_type, str):
        return None

    damage_type = damage_type.strip().lower()

    resistances = {
        str(value).strip().lower()
        for value in (damage_resistances or [])
        if isinstance(value, str)
    }

    vulnerabilities = {
        str(value).strip().lower()
        for value in (damage_vulnerabilities or [])
        if isinstance(value, str)
    }

    immunities = {
        str(value).strip().lower()
        for value in (damage_immunities or [])
        if isinstance(value, str)
    }

    if damage_type in immunities:
        return {
            "damage": 0,
            "damage_type": damage_type,
            "modifier": "immunity",
        }

    if damage_type in resistances:
        damage = damage // 2

    if damage_type in vulnerabilities:
        damage = damage * 2

    modifier = "none"

    if damage_type in resistances and damage_type in vulnerabilities:
        modifier = "resistance_and_vulnerability"
    elif damage_type in resistances:
        modifier = "resistance"
    elif damage_type in vulnerabilities:
        modifier = "vulnerability"

    return {
        "damage": damage,
        "damage_type": damage_type,
        "modifier": modifier,
    }


def resolve_monster_attack_damage(
    attack_result,
    damage_entries,
    damage_roll_results,
    damage_resistances=None,
    damage_vulnerabilities=None,
    damage_immunities=None,
):
    """
    Resuelve el daño de un ataque de monstruo.

    El monstruo puede tener uno o varios tipos de daño.
    Cada resultado de daño es proporcionado por Dice Golem.

    No realiza tiradas de dados.
    """

    if not isinstance(attack_result, dict):
        return None

    outcome = attack_result.get("outcome")

    if outcome not in ("hit", "critical", "miss"):
        return None

    if outcome == "miss":
        return {
            "damage": 0,
            "damages": [],
            "critical": False,
        }

    if not isinstance(damage_entries, list):
        return None

    if not isinstance(damage_roll_results, list):
        return None

    if len(damage_entries) != len(damage_roll_results):
        return None

    resolved_damages = []
    total_damage = 0

    for entry, roll_result in zip(
        damage_entries,
        damage_roll_results,
    ):
        if not isinstance(entry, dict):
            return None

        damage_type = entry.get("damage_type")
        damage_dice = entry.get("damage_dice")

        if not isinstance(damage_type, str):
            return None

        if not isinstance(damage_dice, str):
            return None

        try:
            damage_roll = int(roll_result)
        except (TypeError, ValueError):
            return None

        if damage_roll < 0:
            return None

        modified_damage = apply_damage_modifiers(
            damage_roll,
            damage_type,
            damage_resistances=damage_resistances,
            damage_vulnerabilities=damage_vulnerabilities,
            damage_immunities=damage_immunities,
        )

        if modified_damage is None:
            return None

        final_damage = modified_damage["damage"]
        total_damage += final_damage

        resolved_damages.append({
            "damage_dice": damage_dice,
            "damage_roll": damage_roll,
            "damage_type": modified_damage["damage_type"],
            "damage": final_damage,
            "damage_modifier": modified_damage["modifier"],
        })

    return {
        "damage": total_damage,
        "damages": resolved_damages,
        "critical": outcome == "critical",
    }

def resolve_weapon_attack(
    weapon_idx,
    ability_scores,
    class_idx,
    level,
    race_idx=None,
    subrace_idx=None,
):
    """
    Resuelve la información mecánica necesaria para realizar
    un ataque con un arma.

    No tira dados ni determina impacto.
    """
    if not isinstance(weapon_idx, str):
        return None

    if not isinstance(class_idx, str):
        return None

    if not isinstance(ability_scores, dict):
        return None

    weapon_idx = weapon_idx.strip().lower()
    class_idx = class_idx.strip().lower()

    if not weapon_idx or not class_idx:
        return None

    weapon_data = get_weapon_data(weapon_idx)

    if not weapon_data:
        return None

    damage_data = get_weapon_damage_data(
        weapon_idx,
        ability_scores,
    )

    if not damage_data:
        return None

    proficient = is_weapon_proficient(
        class_idx,
        weapon_idx,
        race_idx,
        subrace_idx,
    )

    attack_modifier = get_weapon_attack_modifier(
        weapon_idx,
        ability_scores,
        class_idx,
        level,
        race_idx,
        subrace_idx,
    )

    if attack_modifier is None:
        return None

    return {
        "weapon": weapon_idx,
        "proficient": proficient,
        "attack_ability": damage_data["ability"],
        "attack_modifier": attack_modifier,
        "damage_dice": damage_data["damage_dice"],
        "damage_type": damage_data["damage_type"],
        "damage_ability_modifier": damage_data["ability_modifier"],
    }


def resolve_weapon_damage(
    weapon_idx,
    ability_scores,
    roll_result,
):
    """
    Resuelve el daño normal de un arma a partir del resultado
    de su tirada de daño y del modificador de característica.

    No aplica todavía:
    - críticos;
    - resistencias;
    - vulnerabilidades;
    - bonificadores mágicos;
    - otros modificadores situacionales.
    """
    damage_data = get_weapon_damage_data(
        weapon_idx,
        ability_scores,
    )

    if not damage_data:
        return None

    try:
        roll_result = int(roll_result)
    except (TypeError, ValueError):
        return None

    if roll_result < 1:
        return None

    total_damage = roll_result + damage_data["ability_modifier"]

    return {
        "weapon": damage_data["weapon"],
        "roll_result": roll_result,
        "ability_modifier": damage_data["ability_modifier"],
        "damage": max(1, total_damage),
        "damage_type": damage_data["damage_type"],
    }


def get_weapon_damage_data(
    weapon_idx,
    ability_scores,
):
    """
    Devuelve los datos mecánicos básicos de daño de un arma.

    Incluye:
    - dado de daño;
    - tipo de daño;
    - característica utilizada para añadir el modificador al daño.

    No calcula todavía críticos, resistencias, vulnerabilidades,
    bonificadores mágicos ni otros modificadores situacionales.
    """
    if not isinstance(weapon_idx, str):
        return None

    if not isinstance(ability_scores, dict):
        return None

    weapon_idx = weapon_idx.strip().lower()

    if not weapon_idx:
        return None

    weapon = get_weapon_data(weapon_idx)

    if not weapon:
        return None

    damage_dice = weapon.get("damage_dice")

    if not damage_dice:
        return None

    damage_type = weapon.get("damage_type")

    if not damage_type:
        return None

    attack_ability = get_weapon_attack_ability(
        weapon_idx,
        ability_scores,
    )

    if attack_ability not in ("str", "dex"):
        return None

    ability_key = {
        "str": "strength",
        "dex": "dexterity",
    }.get(attack_ability)

    if ability_key is None:
        return None

    ability_score = ability_scores.get(ability_key)

    if ability_score is None:
        ability_score = ability_scores.get(attack_ability)

    ability_modifier = get_ability_modifier(ability_score)

    if ability_modifier is None:
        return None

    return {
        "weapon": weapon_idx,
        "damage_dice": damage_dice,
        "damage_type": damage_type,
        "ability": attack_ability,
        "ability_modifier": ability_modifier,
    }


def get_weapon_attack_modifier(
    weapon_idx,
    ability_scores,
    class_idx,
    level,
    race_idx=None,
    subrace_idx=None,
):
    """
    Calcula el modificador total de ataque con un arma.

    Incluye:
    - modificador de la característica de ataque;
    - bonificador de competencia si el personaje es competente.

    No incluye todavía bonificadores mágicos u otros modificadores
    situacionales.
    """
    if not isinstance(class_idx, str):
        return None

    if not isinstance(weapon_idx, str):
        return None

    if not isinstance(ability_scores, dict):
        return None

    attack_ability = get_weapon_attack_ability(
        weapon_idx,
        ability_scores,
    )

    if attack_ability not in ("str", "dex"):
        return None

    ability_key = {
        "str": "strength",
        "dex": "dexterity",
    }.get(attack_ability)

    if ability_key is None:
        return None

    ability_score = ability_scores.get(ability_key)

    if ability_score is None:
        ability_score = ability_scores.get(attack_ability)

    ability_modifier = get_ability_modifier(ability_score)

    if ability_modifier is None:
        return None

    proficient = is_weapon_proficient(
        class_idx,
        weapon_idx,
        race_idx,
        subrace_idx,
    )

    if not proficient:
        return ability_modifier

    proficiency_bonus = get_proficiency_bonus(
        class_idx,
        level,
    )

    if proficiency_bonus is None:
        return None

    return ability_modifier + proficiency_bonus

def get_weapon_attack_ability(
    weapon_idx,
    ability_scores,
):
    """
    Determina la característica utilizada para atacar con un arma.

    Armas cuerpo a cuerpo:
        Fuerza.

    Armas a distancia:
        Destreza.

    Armas con Finesse:
        Fuerza o Destreza; se utiliza el modificador más alto.

    La decisión se basa en los datos del arma almacenados en Magical20.
    """
    if not isinstance(ability_scores, dict):
        return None

    weapon_data = get_weapon_data(weapon_idx)

    if not weapon_data:
        return None

    weapon_range = weapon_data.get("weapon_range")
    properties = weapon_data.get("properties", [])

    strength_score = ability_scores.get("strength")
    if strength_score is None:
        strength_score = ability_scores.get("str")

    dexterity_score = ability_scores.get("dexterity")
    if dexterity_score is None:
        dexterity_score = ability_scores.get("dex")

    strength_modifier = get_ability_modifier(strength_score)
    dexterity_modifier = get_ability_modifier(dexterity_score)

    if strength_modifier is None or dexterity_modifier is None:
        return None

    if "finesse" in properties:
        if dexterity_modifier > strength_modifier:
            return "dex"
        return "str"

    if weapon_range == "Ranged":
        return "dex"

    if weapon_range == "Melee":
        return "str"

    return None

def get_weapon_data(weapon_idx):
    """
    Devuelve los datos mecánicos de un arma usando exclusivamente
    el registro correspondiente de Magical20.
    """
    if not isinstance(weapon_idx, str):
        return None

    weapon_idx = weapon_idx.strip().lower()

    if not weapon_idx:
        return None

    weapon = get_equipment(weapon_idx)

    if not weapon:
        return None

    raw = weapon.get("raw")

    if not isinstance(raw, dict):
        return None

    weapon_category = raw.get("weapon_category")
    weapon_range = raw.get("weapon_range")
    damage = raw.get("damage")
    properties = raw.get("properties", [])

    if not isinstance(weapon_category, str):
        return None

    if not isinstance(weapon_range, str):
        return None

    if not isinstance(properties, list):
        properties = []

    property_indices = []

    for item in properties:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")

        if isinstance(idx, str):
            property_indices.append(idx)

    damage_dice = None
    damage_type = None

    if isinstance(damage, dict):
        damage_dice = damage.get("damage_dice")

        damage_type_data = damage.get("damage_type")

        if isinstance(damage_type_data, dict):
            damage_type = damage_type_data.get("index")

    weapon_range_data = raw.get("range")

    normal_range = None
    long_range = None

    if isinstance(weapon_range_data, dict):
        normal_range = weapon_range_data.get("normal")
        long_range = weapon_range_data.get("long")

    return {
        "index": weapon_idx,
        "name": raw.get("name"),
        "weapon_category": weapon_category,
        "weapon_range": weapon_range,
        "damage_dice": damage_dice,
        "damage_type": damage_type,
        "normal_range": normal_range,
        "long_range": long_range,
        "properties": property_indices,
    }

def get_weapon_damage_dice(
    weapon_idx,
    critical=False,
):
    """
    Devuelve la expresión de dados de daño de un arma.

    Ataque normal:
        usa exactamente los dados definidos por Magical20.

    Golpe crítico:
        duplica la cantidad de dados, manteniendo el mismo tipo de dado.

    Ejemplos:
        1d8  -> 1d8
        1d8  -> 2d8 en crítico
        2d6  -> 4d6 en crítico
    """
    if not isinstance(weapon_idx, str):
        return None

    weapon_idx = weapon_idx.strip().lower()

    if not weapon_idx:
        return None

    weapon = get_weapon_data(weapon_idx)

    if not weapon:
        return None

    damage_dice = weapon.get("damage_dice")

    if not isinstance(damage_dice, str):
        return None

    damage_dice = damage_dice.strip().lower()

    if not damage_dice:
        return None

    if not critical:
        return damage_dice

    parts = damage_dice.split("d", 1)

    if len(parts) != 2:
        return None

    try:
        number_of_dice = int(parts[0])
    except (TypeError, ValueError):
        return None

    die_size = parts[1]

    if number_of_dice < 1 or not die_size.isdigit():
        return None

    return f"{number_of_dice * 2}d{die_size}"


def get_armor_class(
    dexterity_score,
    armor_idx=None,
    shield=False,
):
    """
    Calcula la Clase de Armadura (AC) usando exclusivamente
    los datos de armaduras almacenados en Magical20.

    Sin armadura:
        AC = 10 + modificador de Destreza.

    Armadura ligera:
        AC = base + DEX.

    Armadura media:
        AC = base + DEX, limitado por max_bonus.

    Armadura pesada:
        AC = base.

    Escudo:
        suma el valor base del escudo.

    Devuelve None si los datos son inválidos.
    """
    dexterity_modifier = get_ability_modifier(dexterity_score)

    if dexterity_modifier is None:
        return None

    armor = None

    if armor_idx is not None:
        if not isinstance(armor_idx, str):
            return None

        armor_idx = armor_idx.strip().lower()

        if not armor_idx:
            return None

        armor = get_equipment(armor_idx)

        if not armor:
            return None

        raw = armor.get("raw")

        if not isinstance(raw, dict):
            return None

        armor_class = raw.get("armor_class")

        if not isinstance(armor_class, dict):
            return None

        armor_category = raw.get("armor_category")

        if armor_category == "Shield":
            return None

        base = armor_class.get("base")

        try:
            base = int(base)
        except (TypeError, ValueError):
            return None

        ac = base

        if armor_class.get("dex_bonus") is True:
            dex_bonus = dexterity_modifier

            max_bonus = armor_class.get("max_bonus")

            if max_bonus is not None:
                try:
                    max_bonus = int(max_bonus)
                except (TypeError, ValueError):
                    return None

                dex_bonus = min(dex_bonus, max_bonus)

            ac += dex_bonus

    else:
        ac = 10 + dexterity_modifier

    if shield:
        shield_data = get_equipment("shield")

        if not shield_data:
            return None

        shield_raw = shield_data.get("raw")

        if not isinstance(shield_raw, dict):
            return None

        if shield_raw.get("armor_category") != "Shield":
            return None

        shield_class = shield_raw.get("armor_class")

        if not isinstance(shield_class, dict):
            return None

        shield_bonus = shield_class.get("base")

        try:
            shield_bonus = int(shield_bonus)
        except (TypeError, ValueError):
            return None

        ac += shield_bonus

    return ac

def get_initiative_modifier(dexterity_score):
    """
    Devuelve el modificador de iniciativa de un personaje.

    En D&D 5e, la iniciativa utiliza el modificador de Destreza.
    """
    return get_ability_modifier(dexterity_score)

def get_level_one_hp(class_idx, constitution_score):
    """
    Calcula los puntos de golpe máximos de nivel 1.

    HP = dado de golpe máximo de la clase + modificador de Constitución.
    """
    hit_die = get_class_hit_die(class_idx)

    if hit_die is None:
        return None

    constitution_modifier = get_ability_modifier(constitution_score)

    if constitution_modifier is None:
        return None

    return hit_die + constitution_modifier

def get_race(idx):
    return _get_one(
        """
        SELECT *
        FROM races
        WHERE idx = ?
        """,
        (idx,),
    )


def get_subrace(idx):
    return _get_one(
        """
        SELECT *
        FROM subraces
        WHERE idx = ?
        """,
        (idx,),
    )


def get_level(class_idx, level):
    return _get_one(
        """
        SELECT *
        FROM levels
        WHERE idx = ?
        """,
        (f"{class_idx}-{int(level)}",),
    )


def get_feature(idx):
    return _get_one("SELECT * FROM features WHERE idx = ?", (idx,))


def get_trait(idx):
    return _get_one("SELECT * FROM traits WHERE idx = ?", (idx,))


    return _get_one(
        """
        SELECT *
        FROM features
        WHERE idx = ?
        """,
        (idx,),
    )


def get_proficiency(idx):
    return _get_one(
        """
        SELECT *
        FROM proficiencies
        WHERE idx = ?
        """,
        (idx,),
    )



def get_monster(idx):
    return _get_one(
        "SELECT * FROM monsters WHERE idx = ?",
        (idx,),
    )


def get_monster_combat_data(monster_idx):
    """
    Devuelve los datos mecánicos de combate de un monstruo
    usando exclusivamente el registro de Magical20.

    No modifica el monstruo ni campaign_state.
    """
    if not isinstance(monster_idx, str):
        return None

    monster_idx = monster_idx.strip().lower()

    if not monster_idx:
        return None

    monster = get_monster(monster_idx)

    if not monster:
        return None

    raw = monster.get("raw")

    if not isinstance(raw, dict):
        return None

    name = raw.get("name")
    hit_points = raw.get("hit_points")
    dexterity = raw.get("dexterity")

    if not isinstance(name, str):
        return None

    try:
        hit_points = int(hit_points)
    except (TypeError, ValueError):
        return None

    if hit_points < 0:
        return None

    try:
        dexterity = int(dexterity)
    except (TypeError, ValueError):
        return None

    initiative_modifier = get_initiative_modifier(dexterity)

    if initiative_modifier is None:
        return None

    armor_class = raw.get("armor_class")

    if not isinstance(armor_class, list):
        return None

    ac_values = []

    for entry in armor_class:
        if not isinstance(entry, dict):
            continue

        value = entry.get("value")

        try:
            value = int(value)
        except (TypeError, ValueError):
            continue

        if value >= 0:
            ac_values.append(value)

    if not ac_values:
        return None

    resistances = raw.get("damage_resistances", [])
    vulnerabilities = raw.get("damage_vulnerabilities", [])
    immunities = raw.get("damage_immunities", [])

    if not isinstance(resistances, list):
        resistances = []

    if not isinstance(vulnerabilities, list):
        vulnerabilities = []

    if not isinstance(immunities, list):
        immunities = []

    actions = raw.get("actions", [])

    if not isinstance(actions, list):
        actions = []

    return {
        "index": monster_idx,
        "name": name,
        "hp": hit_points,
        "ac": ac_values[0],
        "dexterity": dexterity,
        "initiative_modifier": initiative_modifier,
        "damage_resistances": resistances,
        "damage_vulnerabilities": vulnerabilities,
        "damage_immunities": immunities,
        "actions": actions,
    }


def get_condition(idx):
    return _get_one(
        """
        SELECT *
        FROM conditions
        WHERE idx = ?
        """,
        (idx,),
    )


def get_condition_effects(condition_idx):
    """
    Devuelve los efectos mecánicos normalizados de una condición.

    La condición válida se verifica exclusivamente contra Magical20.
    Los efectos se representan de forma determinista para que otras
    funciones mecánicas puedan consultarlos sin depender del LLM.
    """
    if not isinstance(condition_idx, str):
        return None

    condition_idx = condition_idx.strip().lower()

    if not condition_idx:
        return None

    condition = get_condition(condition_idx)

    if not isinstance(condition, dict):
        return None

    effects = {
        "attack_disadvantage": False,
        "attacks_against_advantage": False,
        "attacks_against_disadvantage": False,
        "ability_check_disadvantage": False,
        "dexterity_save_disadvantage": False,
        "strength_save_auto_fail": False,
        "dexterity_save_auto_fail": False,
        "speed_zero": False,
        "speed_halved": False,
        "cannot_move": False,
        "cannot_take_actions": False,
        "cannot_take_reactions": False,
        "critical_hit_within_5ft": False,
        "resistance_all_damage": False,
        "poison_immunity": False,
        "disease_immunity": False,
    }

    condition_effects = {
        "blinded": {
            "attack_disadvantage": True,
            "attacks_against_advantage": True,
        },
        "charmed": {},
        "deafened": {},
        "exhaustion": {},
        "frightened": {
            "ability_check_disadvantage": True,
            "attack_disadvantage": True,
        },
        "grappled": {
            "speed_zero": True,
        },
        "incapacitated": {
            "cannot_take_actions": True,
            "cannot_take_reactions": True,
        },
        "invisible": {
            "attack_disadvantage": False,
            "attacks_against_disadvantage": True,
        },
        "paralyzed": {
            "cannot_move": True,
            "speed_zero": True,
            "cannot_take_actions": True,
            "cannot_take_reactions": True,
            "strength_save_auto_fail": True,
            "dexterity_save_auto_fail": True,
            "attacks_against_advantage": True,
            "critical_hit_within_5ft": True,
        },
        "petrified": {
            "cannot_move": True,
            "speed_zero": True,
            "cannot_take_actions": True,
            "cannot_take_reactions": True,
            "strength_save_auto_fail": True,
            "dexterity_save_auto_fail": True,
            "attacks_against_advantage": True,
            "resistance_all_damage": True,
            "poison_immunity": True,
            "disease_immunity": True,
        },
        "poisoned": {
            "attack_disadvantage": True,
            "ability_check_disadvantage": True,
        },
        "prone": {
            "attack_disadvantage": True,
            "attacks_against_advantage": True,
            "attacks_against_disadvantage": True,
        },
        "restrained": {
            "speed_zero": True,
            "attack_disadvantage": True,
            "attacks_against_advantage": True,
            "dexterity_save_disadvantage": True,
        },
        "stunned": {
            "cannot_move": True,
            "speed_zero": True,
            "cannot_take_actions": True,
            "cannot_take_reactions": True,
            "strength_save_auto_fail": True,
            "dexterity_save_auto_fail": True,
            "attacks_against_advantage": True,
        },
        "unconscious": {
            "cannot_move": True,
            "speed_zero": True,
            "cannot_take_actions": True,
            "cannot_take_reactions": True,
            "strength_save_auto_fail": True,
            "dexterity_save_auto_fail": True,
            "attacks_against_advantage": True,
            "critical_hit_within_5ft": True,
        },
    }

    selected = condition_effects.get(condition_idx)

    if selected is None:
        return None

    effects.update(selected)

    return {
        "idx": condition_idx,
        "name": condition.get("name"),
        "effects": effects,
    }


def can_take_action(conditions):
    """
    Determina si un combatiente puede realizar acciones
    según sus condiciones actuales.

    Las reglas mecánicas se obtienen exclusivamente mediante
    get_condition_effects().
    """
    if not isinstance(conditions, list):
        return False

    for condition in conditions:
        condition_idx = None

        if isinstance(condition, str):
            condition_idx = condition.strip()

        elif isinstance(condition, dict):
            condition_idx = condition.get("idx")

        if not isinstance(condition_idx, str):
            continue

        condition_idx = condition_idx.strip().lower()

        if not condition_idx:
            continue

        effects_data = get_condition_effects(condition_idx)

        if not isinstance(effects_data, dict):
            return False

        effects = effects_data.get("effects")

        if not isinstance(effects, dict):
            return False

        if effects.get("cannot_take_actions") is True:
            return False

    return True



def resolve_attack_advantage_disadvantage(
    attacker_conditions,
    target_conditions,
    distance_ft=None,
):
    """
    Determina ventaja o desventaja en un ataque según las condiciones
    del atacante y del objetivo.

    Devuelve:
    - advantage: True/False
    - disadvantage: True/False
    - mode: advantage, disadvantage o normal

    Si existen simultáneamente ventaja y desventaja, se cancelan.
    Las reglas que requieren contexto adicional no se aplican aquí.
    """
    if not isinstance(attacker_conditions, list):
        attacker_conditions = []

    if not isinstance(target_conditions, list):
        target_conditions = []

    attacker_keys = set()
    target_keys = set()

    for condition in attacker_conditions:
        if isinstance(condition, str):
            key = condition.strip().lower()
            if key:
                attacker_keys.add(key)
        elif isinstance(condition, dict):
            idx = condition.get("idx")
            if isinstance(idx, str) and idx.strip():
                attacker_keys.add(idx.strip().lower())

    for condition in target_conditions:
        if isinstance(condition, str):
            key = condition.strip().lower()
            if key:
                target_keys.add(key)
        elif isinstance(condition, dict):
            idx = condition.get("idx")
            if isinstance(idx, str) and idx.strip():
                target_keys.add(idx.strip().lower())

    advantage = False
    disadvantage = False

    # Condiciones del atacante.
    if "blinded" in attacker_keys:
        disadvantage = True

    if "frightened" in attacker_keys:
        disadvantage = True

    if "poisoned" in attacker_keys:
        disadvantage = True

    if "restrained" in attacker_keys:
        disadvantage = True

    if "prone" in attacker_keys:
        disadvantage = True

    if "invisible" in attacker_keys:
        advantage = True

    # Condiciones del objetivo.
    if "blinded" in target_keys:
        advantage = True

    if "invisible" in target_keys:
        disadvantage = True

    if "paralyzed" in target_keys:
        advantage = True

    if "petrified" in target_keys:
        advantage = True

    if "restrained" in target_keys:
        advantage = True

    if "stunned" in target_keys:
        advantage = True

    if "unconscious" in target_keys:
        advantage = True

    # Prone depende de la distancia del atacante.
    if "prone" in target_keys:
        if isinstance(distance_ft, (int, float)) and not isinstance(
            distance_ft, bool
        ):
            if distance_ft <= 5:
                advantage = True
            else:
                disadvantage = True

    # Ventaja y desventaja simultáneas se cancelan.
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

def get_equipment(idx):
    return _get_one(
        """
        SELECT *
        FROM equipment
        WHERE idx = ?
        """,
        (idx,),
    )


def get_equipment_by_category(category_idx):
    """
    Devuelve los objetos de equipo pertenecientes a una categoría.

    Usa exclusivamente los datos almacenados en Magical20.
    La categoría se obtiene desde el raw_json de cada objeto.
    """
    if not isinstance(category_idx, str):
        return []

    category_idx = category_idx.strip().lower()
    if not category_idx:
        return []

    conn = _connect()
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM equipment
            """
        )
        rows = cursor.fetchall()

        results = []

        for row in rows:
            data = _row_to_dict(cursor, row)
            raw = data.get("raw")

            if not isinstance(raw, dict):
                continue

            equipment_category = raw.get("equipment_category")

            if (
                isinstance(equipment_category, dict)
                and equipment_category.get("index") == category_idx
            ):
                results.append(data)

        return results
    finally:
        conn.close()


def _equipment_prerequisites_met(option, proficiencies=None):
    """
    Comprueba los prerrequisitos de una opción de equipamiento.

    Actualmente soporta prerrequisitos de competencia, que es el
    formato utilizado por los presets de equipamiento de Magical20.
    """
    if not isinstance(option, dict):
        return False

    if proficiencies is None:
        proficiencies = []

    if not isinstance(proficiencies, (list, tuple, set)):
        return False

    normalized = {
        str(value).strip().lower()
        for value in proficiencies
        if str(value).strip()
    }

    prerequisites = option.get("prerequisites", [])
    if not prerequisites:
        return True

    if not isinstance(prerequisites, list):
        return False

    for prerequisite in prerequisites:
        if not isinstance(prerequisite, dict):
            return False

        if prerequisite.get("type") != "proficiency":
            return False

        proficiency = prerequisite.get("proficiency", {})
        if not isinstance(proficiency, dict):
            return False

        proficiency_idx = proficiency.get("index")
        if not isinstance(proficiency_idx, str):
            return False

        if proficiency_idx.strip().lower() not in normalized:
            return False

    return True


def resolve_starting_equipment_option(
    option,
    proficiencies=None,
    rng=None,
):
    """
    Resuelve recursivamente una opción de equipamiento de Magical20.

    Devuelve una lista de objetos:
        {
            "idx": str,
            "name": str,
            "quantity": int,
        }

    El azar se utiliza únicamente para seleccionar entre opciones
    válidas del preset.
    """
    import random

    if rng is None:
        rng = random

    if not isinstance(option, dict):
        return []

    option_type = option.get("option_type")

    if option_type == "counted_reference":
        reference = option.get("of", {})
        if not isinstance(reference, dict):
            return []

        idx = reference.get("index")
        name = reference.get("name")

        if not isinstance(idx, str) or not idx.strip():
            return []

        if not _equipment_prerequisites_met(option, proficiencies):
            return []

        item = get_equipment(idx)
        if item is None:
            return []

        try:
            quantity = int(option.get("count", 1))
        except (TypeError, ValueError):
            return []

        if quantity <= 0:
            return []

        return [{
            "idx": idx.strip(),
            "name": str(name or item.get("name") or idx).strip(),
            "quantity": quantity,
        }]

    if option_type == "multiple":
        items = option.get("items", [])
        if not isinstance(items, list):
            return []

        result = []

        for item in items:
            result.extend(
                resolve_starting_equipment_option(
                    item,
                    proficiencies=proficiencies,
                    rng=rng,
                )
            )

        return result

    if option_type == "choice":
        choice = option.get("choice")
        if not isinstance(choice, dict):
            return []

        return _resolve_starting_equipment_choice(
            choice,
            proficiencies=proficiencies,
            rng=rng,
        )

    return []



def resolve_manual_starting_equipment(
    class_idx,
    equipment_choices,
    proficiencies=None,
):
    """
    Valida y resuelve una selección manual de equipamiento.

    La estructura raw de starting_equipment_options de Magical20
    es la única fuente de verdad.

    equipment_choices contiene únicamente los objetos que el jugador
    seleccionó explícitamente. Los objetos derivados por una opción
    (por ejemplo, 20 virotes al elegir una ballesta) se generan
    automáticamente.

    Se utiliza backtracking para resolver correctamente las
    ambigüedades producidas cuando un objeto concreto también
    pertenece a una categoría de equipamiento.
    """
    if not isinstance(class_idx, str):
        return []

    class_data = get_class(class_idx.strip().lower())
    if not isinstance(class_data, dict):
        return []

    raw = class_data.get("raw")
    if not isinstance(raw, dict):
        return []

    if not isinstance(equipment_choices, list):
        return []

    choices = []

    for value in equipment_choices:
        if not isinstance(value, str):
            return []

        value = value.strip()

        if not value or value in choices:
            return []

        choices.append(value)

    if not choices:
        return []

    selected_ids = set(choices)

    options_groups = raw.get("starting_equipment_options", [])

    if not isinstance(options_groups, list):
        return []

    def equipment_entry(idx, quantity=1, reference=None):
        item = get_equipment(idx)

        if item is None:
            return None

        try:
            quantity = int(quantity)
        except (TypeError, ValueError):
            return None

        if quantity <= 0:
            return None

        return {
            "idx": idx,
            "name": str(
                (reference or {}).get("name")
                or item.get("name")
                or idx
            ).strip(),
            "quantity": quantity,
        }

    def resolve_counted_reference(option, available_ids):
        if not isinstance(option, dict):
            return []

        reference = option.get("of", {})

        if not isinstance(reference, dict):
            return []

        idx = reference.get("index")

        if not isinstance(idx, str):
            return []

        idx = idx.strip()

        if idx not in available_ids:
            return []

        if not _equipment_prerequisites_met(
            option,
            proficiencies,
        ):
            return []

        try:
            quantity = int(option.get("count", 1))
        except (TypeError, ValueError):
            return []

        entry = equipment_entry(
            idx,
            quantity,
            reference,
        )

        if entry is None:
            return []

        return [([entry], {idx})]

    def resolve_choice_option(option, available_ids):
        if not isinstance(option, dict):
            return []

        choice = option.get("choice")

        if not isinstance(choice, dict):
            return []

        try:
            choose = int(choice.get("choose", 1))
        except (TypeError, ValueError):
            return []

        if choose <= 0:
            return []

        source = choice.get("from", {})

        if not isinstance(source, dict):
            return []

        option_set_type = source.get("option_set_type")

        if option_set_type == "equipment_category":
            category = source.get(
                "equipment_category",
                {},
            )

            if not isinstance(category, dict):
                return []

            category_idx = category.get("index")

            if not isinstance(category_idx, str):
                return []

            available = get_equipment_by_selection_category(
                category_idx
            )

            if not isinstance(available, list):
                return []

            available_by_idx = {
                item.get("idx"): item
                for item in available
                if isinstance(item, dict)
                and isinstance(item.get("idx"), str)
            }

            matching = [
                idx
                for idx in available_ids
                if idx in available_by_idx
            ]

            if len(matching) != choose:
                return []

            result = []

            for idx in matching:
                entry = equipment_entry(idx)

                if entry is None:
                    return []

                result.append(entry)

            return [(result, set(matching))]

        return []

    def resolve_option(option, available_ids):
        if not isinstance(option, dict):
            return []

        option_type = option.get("option_type")

        if option_type == "counted_reference":
            return resolve_counted_reference(
                option,
                available_ids,
            )

        if option_type == "choice":
            return resolve_choice_option(
                option,
                available_ids,
            )

        if option_type == "multiple":
            items = option.get("items", [])

            if not isinstance(items, list):
                return []

            partial = [([], set())]

            for item in items:
                next_partial = []

                for current_result, current_consumed in partial:
                    remaining_ids = (
                        set(available_ids)
                        - current_consumed
                    )

                    # Los counted_reference con cantidad > 1 que no
                    # fueron seleccionados explícitamente son objetos
                    # derivados de la elección principal.
                    #
                    # Ejemplos:
                    # crossbow-light -> crossbow-bolt x20
                    # shortbow -> arrow x20
                    if item.get("option_type") == "counted_reference":
                        reference = item.get("of", {})

                        if not isinstance(reference, dict):
                            continue

                        derived_idx = reference.get("index")

                        try:
                            quantity = int(item.get("count", 1))
                        except (TypeError, ValueError):
                            continue

                        if (
                            isinstance(derived_idx, str)
                            and derived_idx not in remaining_ids
                            and quantity > 1
                            and _equipment_prerequisites_met(
                                item,
                                proficiencies,
                            )
                        ):
                            entry = equipment_entry(
                                derived_idx,
                                quantity,
                                reference,
                            )

                            if entry is not None:
                                next_partial.append(
                                    (
                                        current_result + [entry],
                                        set(current_consumed),
                                    )
                                )

                            continue

                    item_results = resolve_option(
                        item,
                        remaining_ids,
                    )

                    for item_result, item_consumed in item_results:
                        next_partial.append(
                            (
                                current_result + item_result,
                                current_consumed | item_consumed,
                            )
                        )

                partial = next_partial

                if not partial:
                    return []

            return partial

        return []

    def resolve_group(group, available_ids):
        """
        Devuelve todas las interpretaciones válidas de un grupo.

        No se exige que haya una única coincidencia: el backtracking
        exterior decidirá cuál combinación satisface todos los grupos.
        """
        if not isinstance(group, dict):
            return []

        try:
            choose = int(group.get("choose", 1))
        except (TypeError, ValueError):
            return []

        if choose <= 0:
            return []

        source = group.get("from", {})

        if not isinstance(source, dict):
            return []

        option_set_type = source.get("option_set_type")

        if option_set_type == "options_array":
            candidates = source.get("options", [])

            if not isinstance(candidates, list):
                return []

            results = []

            for candidate in candidates:
                results.extend(
                    resolve_option(
                        candidate,
                        available_ids,
                    )
                )

            return results

        if option_set_type == "equipment_category":
            category = source.get(
                "equipment_category",
                {},
            )

            if not isinstance(category, dict):
                return []

            category_idx = category.get("index")

            if not isinstance(category_idx, str):
                return []

            available = get_equipment_by_selection_category(
                category_idx
            )

            if not isinstance(available, list):
                return []

            available_by_idx = {
                item.get("idx"): item
                for item in available
                if isinstance(item, dict)
                and isinstance(item.get("idx"), str)
            }

            matching = [
                idx
                for idx in available_ids
                if idx in available_by_idx
            ]

            if len(matching) != choose:
                return []

            result = []

            for idx in matching:
                entry = equipment_entry(idx)

                if entry is None:
                    return []

                result.append(entry)

            return [(result, set(matching))]

        return []

    def backtrack(group_index, remaining_ids):
        if group_index >= len(options_groups):
            if not remaining_ids:
                return []

            return None

        group = options_groups[group_index]

        matches = resolve_group(
            group,
            remaining_ids,
        )

        for result, consumed in matches:
            if not consumed:
                continue

            next_remaining = (
                set(remaining_ids)
                - consumed
            )

            tail = backtrack(
                group_index + 1,
                next_remaining,
            )

            if tail is not None:
                return result + tail

        return None

    resolved = backtrack(
        0,
        selected_ids,
    )

    if resolved is None:
        return []

    return resolved

def _resolve_starting_equipment_choice(
    choice,
    proficiencies=None,
    rng=None,
):
    """
    Resuelve una estructura de elección de equipamiento.
    """
    import random

    if rng is None:
        rng = random

    if not isinstance(choice, dict):
        return []

    try:
        choose = int(choice.get("choose", 1))
    except (TypeError, ValueError):
        return []

    if choose <= 0:
        return []

    source = choice.get("from", {})
    if not isinstance(source, dict):
        return []

    option_set_type = source.get("option_set_type")

    if option_set_type == "options_array":
        candidates = source.get("options", [])
        if not isinstance(candidates, list):
            return []

        valid = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and _equipment_prerequisites_met(candidate, proficiencies)
        ]

        if not valid:
            return []

        if choose == 1:
            selected = rng.choice(valid)
            return resolve_starting_equipment_option(
                selected,
                proficiencies=proficiencies,
                rng=rng,
            )

        if choose > len(valid):
            return []

        selected = rng.sample(valid, choose)

        result = []
        for item in selected:
            result.extend(
                resolve_starting_equipment_option(
                    item,
                    proficiencies=proficiencies,
                    rng=rng,
                )
            )

        return result

    if option_set_type == "equipment_category":
        category = source.get("equipment_category", {})
        if not isinstance(category, dict):
            return []

        category_idx = category.get("index")
        if not isinstance(category_idx, str):
            return []

        candidates = get_equipment_by_selection_category(category_idx)

        if not candidates:
            return []

        if choose > len(candidates):
            return []

        selected = (
            [rng.choice(candidates)]
            if choose == 1
            else rng.sample(candidates, choose)
        )

        return [
            {
                "idx": item["idx"],
                "name": str(item.get("name") or item["idx"]).strip(),
                "quantity": 1,
            }
            for item in selected
        ]

    return []


def resolve_class_starting_equipment(
    class_idx,
    proficiencies=None,
    rng=None,
):
    """
    Resuelve todo el equipamiento inicial de una clase.

    Cada grupo de starting_equipment_options selecciona
    aleatoriamente una alternativa válida.
    """
    import random

    if rng is None:
        rng = random

    if not isinstance(class_idx, str):
        return []

    class_data = get_class(class_idx.strip().lower())
    if not isinstance(class_data, dict):
        return []

    raw = class_data.get("raw")
    if not isinstance(raw, dict):
        return []

    result = []

    starting_equipment = raw.get("starting_equipment", [])
    if isinstance(starting_equipment, list):
        for entry in starting_equipment:
            if not isinstance(entry, dict):
                continue

            reference = entry.get("equipment", {})
            if not isinstance(reference, dict):
                continue

            idx = reference.get("index")
            if not isinstance(idx, str):
                continue

            item = get_equipment(idx)
            if item is None:
                continue

            try:
                quantity = int(entry.get("quantity", 1))
            except (TypeError, ValueError):
                continue

            if quantity <= 0:
                continue

            result.append({
                "idx": idx,
                "name": str(item.get("name") or reference.get("name") or idx).strip(),
                "quantity": quantity,
            })

    options = raw.get("starting_equipment_options", [])
    if not isinstance(options, list):
        return result

    for option_group in options:
        if not isinstance(option_group, dict):
            continue

        resolved = _resolve_starting_equipment_choice(
            option_group,
            proficiencies=proficiencies,
            rng=rng,
        )

        result.extend(resolved)

    return result


def get_starting_equipment_presets(
    class_idx,
    proficiencies=None,
    preset_count=3,
):
    """
    Genera configuraciones predeterminadas de equipamiento inicial.

    Cada preset utiliza una semilla determinista basada en la clase
    y el número de preset. De esta forma:

        misma clase + mismo preset = mismo equipamiento

    Las elecciones siguen pasando por el resolver de Magical20,
    por lo que no se introducen reglas de equipamiento nuevas.

    Devuelve una lista de presets. Cada preset contiene la lista
    completa de objetos iniciales resueltos.
    """
    import hashlib
    import random

    if not isinstance(class_idx, str):
        return []

    class_idx = class_idx.strip().lower()
    if not class_idx:
        return []

    try:
        preset_count = int(preset_count)
    except (TypeError, ValueError):
        return []

    if preset_count <= 0:
        return []

    presets = []
    signatures = set()

    for preset_number in range(1, preset_count + 1):
        seed_source = f"{class_idx}:preset:{preset_number}"
        seed_bytes = seed_source.encode("utf-8")
        seed = int.from_bytes(
            hashlib.sha256(seed_bytes).digest()[:8],
            "big",
        )

        rng = random.Random(seed)

        equipment = resolve_class_starting_equipment(
            class_idx,
            proficiencies=proficiencies,
            rng=rng,
        )

        if not equipment:
            continue

        # Agrupar objetos iguales dentro del preset.
        # Ejemplo:
        # javelin x1 + javelin x1 -> javelin x2
        grouped = {}

        for item in equipment:
            if not isinstance(item, dict):
                continue

            idx = item.get("idx")
            name = item.get("name")
            quantity = item.get("quantity", 1)

            if not isinstance(idx, str):
                continue

            try:
                quantity = int(quantity)
            except (TypeError, ValueError):
                continue

            if quantity <= 0:
                continue

            if idx not in grouped:
                grouped[idx] = {
                    "idx": idx,
                    "name": str(name or idx).strip(),
                    "quantity": quantity,
                }
            else:
                grouped[idx]["quantity"] += quantity

        equipment = list(grouped.values())

        signature = tuple(
            (
                item["idx"],
                item["quantity"],
            )
            for item in equipment
        )

        if not signature or signature in signatures:
            continue

        signatures.add(signature)

        presets.append({
            "preset": preset_number,
            "equipment": equipment,
        })

    return presets

def get_manual_starting_equipment_options(
    class_idx,
    proficiencies=None,
):
    """
    Devuelve las opciones de selección manual de equipamiento inicial.

    La fuente de verdad es exclusivamente:
        raw["starting_equipment_options"]

    Conserva la estructura de alternativas de Magical20 para que
    el flujo interactivo pueda presentar correctamente cada elección.
    """

    if not isinstance(class_idx, str):
        return []

    class_data = get_class(class_idx.strip().lower())

    if not isinstance(class_data, dict):
        return []

    raw = class_data.get("raw")

    if not isinstance(raw, dict):
        return []

    options_groups = raw.get("starting_equipment_options", [])

    if not isinstance(options_groups, list):
        return []

    def item_option(idx, reference=None):
        if not isinstance(idx, str):
            return None

        item = get_equipment(idx)

        if item is None:
            return None

        return {
            "type": "item",
            "idx": idx,
            "name": str(
                (reference or {}).get("name")
                or item.get("name")
                or idx
            ).strip(),
        }

    def category_option(category_idx, choose):
        if not isinstance(category_idx, str):
            return None

        try:
            choose = int(choose)
        except (TypeError, ValueError):
            return None

        if choose <= 0:
            return None

        available = get_equipment_by_selection_category(
            category_idx
        )

        if not isinstance(available, list):
            return None

        options = []

        for item in available:
            if not isinstance(item, dict):
                continue

            idx = item.get("idx")

            if not isinstance(idx, str):
                continue

            options.append({
                "idx": idx,
                "name": str(
                    item.get("name") or idx
                ).strip(),
            })

        return {
            "type": "category",
            "category": category_idx,
            "choose": choose,
            "options": options,
        }

    def parse_option(option):
        if not isinstance(option, dict):
            return []

        option_type = option.get("option_type")

        if option_type == "counted_reference":
            reference = option.get("of", {})

            if not isinstance(reference, dict):
                return []

            idx = reference.get("index")

            entry = item_option(idx, reference)

            if entry is None:
                return []

            return [entry]

        if option_type == "choice":
            choice = option.get("choice", {})

            if not isinstance(choice, dict):
                return []

            try:
                choose = int(choice.get("choose", 1))
            except (TypeError, ValueError):
                return []

            source = choice.get("from", {})

            if not isinstance(source, dict):
                return []

            option_set_type = source.get("option_set_type")

            if option_set_type == "equipment_category":
                category = source.get(
                    "equipment_category",
                    {},
                )

                if not isinstance(category, dict):
                    return []

                category_idx = category.get("index")

                result = category_option(
                    category_idx,
                    choose,
                )

                return [result] if result else []

            if option_set_type == "options_array":
                candidates = source.get("options", [])

                if not isinstance(candidates, list):
                    return []

                alternatives = []

                for candidate in candidates:
                    parsed = parse_option(candidate)

                    if parsed:
                        alternatives.append({
                            "type": "alternative",
                            "choose": choose,
                            "options": parsed,
                        })

                return alternatives

            return []

        if option_type == "multiple":
            items = option.get("items", [])

            if not isinstance(items, list):
                return []

            parsed_items = []

            for item in items:
                parsed = parse_option(item)

                if not parsed:
                    return []

                if len(parsed) == 1:
                    parsed_items.append(parsed[0])
                else:
                    parsed_items.append({
                        "type": "alternatives",
                        "options": parsed,
                    })

            return [{
                "type": "multiple",
                "items": parsed_items,
            }]

        return []

    result = []

    for group_index, group in enumerate(
        options_groups,
        start=1,
    ):
        if not isinstance(group, dict):
            continue

        try:
            choose = int(group.get("choose", 1))
        except (TypeError, ValueError):
            continue

        if choose <= 0:
            continue

        source = group.get("from", {})

        if not isinstance(source, dict):
            continue

        option_set_type = source.get("option_set_type")

        if option_set_type == "options_array":
            candidates = source.get("options", [])

            if not isinstance(candidates, list):
                continue

            alternatives = []

            for candidate in candidates:
                parsed = parse_option(candidate)

                if parsed:
                    if len(parsed) == 1:
                        alternatives.append(parsed[0])
                    else:
                        alternatives.extend(parsed)

            if alternatives:
                result.append({
                    "group": group_index,
                    "choose": choose,
                    "type": "alternatives",
                    "options": alternatives,
                    "description": str(
                        group.get("desc") or ""
                    ).strip(),
                })

        elif option_set_type == "equipment_category":
            category = source.get(
                "equipment_category",
                {},
            )

            if not isinstance(category, dict):
                continue

            category_idx = category.get("index")

            category = category_option(
                category_idx,
                choose,
            )

            if category:
                result.append({
                    "group": group_index,
                    "choose": choose,
                    "type": "category",
                    "options": [category],
                    "description": str(
                        group.get("desc") or ""
                    ).strip(),
                })

    return result



def get_equipment_by_selection_category(category_idx):
    """
    Devuelve objetos válidos para una categoría de selección de equipo.

    Usa exclusivamente los datos almacenados en Magical20.
    Soporta categorías generales, armas por tipo/rango y las
    categorías especiales de equipo definidas por el SRD.
    """
    if not isinstance(category_idx, str):
        return []

    category_idx = category_idx.strip().lower()
    if not category_idx:
        return []

    conn = _connect()
    try:
        cursor = conn.execute(
            """
            SELECT *
            FROM equipment
            """
        )
        rows = cursor.fetchall()

        results = []

        for row in rows:
            data = _row_to_dict(cursor, row)
            raw = data.get("raw")

            if not isinstance(raw, dict):
                continue

            equipment_category = raw.get("equipment_category", {})
            equipment_category_idx = (
                equipment_category.get("index")
                if isinstance(equipment_category, dict)
                else None
            )

            matches = equipment_category_idx == category_idx

            if category_idx == "martial-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Martial"
                )

            elif category_idx == "martial-melee-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Martial"
                    and raw.get("weapon_range") == "Melee"
                )

            elif category_idx == "martial-ranged-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Martial"
                    and raw.get("weapon_range") == "Ranged"
                )

            elif category_idx == "simple-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Simple"
                )

            elif category_idx == "simple-melee-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Simple"
                    and raw.get("weapon_range") == "Melee"
                )

            elif category_idx == "simple-ranged-weapons":
                matches = (
                    equipment_category_idx == "weapon"
                    and raw.get("weapon_category") == "Simple"
                    and raw.get("weapon_range") == "Ranged"
                )

            elif category_idx == "musical-instruments":
                matches = (
                    equipment_category_idx == "tools"
                    and raw.get("tool_category") == "Musical Instrument"
                )

            elif category_idx in ("holy-symbols", "arcane-foci", "druidic-foci"):
                gear_category = raw.get("gear_category", {})
                gear_category_idx = (
                    gear_category.get("index")
                    if isinstance(gear_category, dict)
                    else None
                )
                matches = (
                    equipment_category_idx == "adventuring-gear"
                    and gear_category_idx == category_idx
                )

            if matches:
                results.append(data)

        return results
    finally:
        conn.close()

def get_spell(idx):
    return _get_one(
        """
        SELECT *
        FROM spells
        WHERE idx = ?
        """,
        (idx,),
    )


def get_class_level_data(class_idx, level):
    """
    Devuelve la progresión mecánica de una clase para un nivel concreto,
    usando exclusivamente los datos almacenados en Magical20.
    """
    if not isinstance(class_idx, str):
        return None

    class_idx = class_idx.strip().lower()

    if not class_idx or not get_class(class_idx):
        return None

    try:
        level = int(level)
    except (TypeError, ValueError):
        return None

    if level < 1 or level > 20:
        return None

    level_data = get_level(class_idx, level)
    if not level_data:
        return None

    raw = level_data.get("raw")
    if not isinstance(raw, dict):
        return None

    return {
        "class": class_idx,
        "level": level,
        "ability_score_bonuses": raw.get("ability_score_bonuses", 0),
        "prof_bonus": raw.get("prof_bonus"),
        "features": raw.get("features", []),
        "spellcasting": raw.get("spellcasting"),
        "class_specific": raw.get("class_specific", {}),
    }


def get_class_level_features(class_idx, level):
    """
    Devuelve las características obtenidas por una clase en un nivel,
    resolviendo sus datos desde la tabla de features de Magical20.
    """
    level_data = get_class_level_data(class_idx, level)

    if not level_data:
        return None

    result = []

    for feature in level_data.get("features", []):
        if not isinstance(feature, dict):
            continue

        idx = feature.get("index")

        if not isinstance(idx, str) or not idx:
            continue

        feature_data = get_feature(idx)

        if not feature_data:
            continue

        raw = feature_data.get("raw")
        if not isinstance(raw, dict):
            raw = {}

        desc = raw.get("desc", [])

        if not isinstance(desc, list):
            desc = [desc] if desc else []

        result.append({
            "index": idx,
            "name": feature_data.get("name") or feature.get("name") or idx,
            "level": raw.get("level", level_data["level"]),
            "description": desc,
        })

    return result


def get_racial_traits(race_idx, subrace_idx=None):
    """
    Devuelve los rasgos de una raza y, opcionalmente, de su subraza,
    resolviendo sus datos desde la base de reglas de Magical20.
    """
    if not isinstance(race_idx, str):
        return None

    race_idx = race_idx.strip().lower()

    race = get_race(race_idx)

    if not race:
        return None

    result = []

    raw_race = race.get("raw")

    if isinstance(raw_race, dict):
        for trait in raw_race.get("traits", []):
            if not isinstance(trait, dict):
                continue

            idx = trait.get("index")

            if not isinstance(idx, str) or not idx:
                continue

            trait_data = get_trait(idx)

            if not trait_data:
                continue

            raw = trait_data.get("raw")
            if not isinstance(raw, dict):
                raw = {}

            desc = raw.get("desc", [])

            if not isinstance(desc, list):
                desc = [desc] if desc else []

            result.append({
                "index": idx,
                "name": trait_data.get("name") or trait.get("name") or idx,
                "source": "race",
                "description": desc,
                "proficiencies": raw.get("proficiencies", []),
                "trait_specific": raw.get("trait_specific"),
                "language_options": raw.get("language_options"),
            })

    if subrace_idx is not None:
        if not isinstance(subrace_idx, str):
            return None

        subrace_idx = subrace_idx.strip().lower()

        subrace = get_subrace(subrace_idx)

        if not subrace:
            return None

        subrace_raw = subrace.get("raw")

        if not isinstance(subrace_raw, dict):
            return None

        parent_race = subrace_raw.get("race", {}).get("index")

        if parent_race != race_idx:
            return None

        for trait in subrace_raw.get("racial_traits", []):
            if not isinstance(trait, dict):
                continue

            idx = trait.get("index")

            if not isinstance(idx, str) or not idx:
                continue

            trait_data = get_trait(idx)

            if not trait_data:
                continue

            raw = trait_data.get("raw")
            if not isinstance(raw, dict):
                raw = {}

            desc = raw.get("desc", [])

            if not isinstance(desc, list):
                desc = [desc] if desc else []

            result.append({
                "index": idx,
                "name": trait_data.get("name") or trait.get("name") or idx,
                "source": "subrace",
                "description": desc,
                "proficiencies": raw.get("proficiencies", []),
                "trait_specific": raw.get("trait_specific"),
                "language_options": raw.get("language_options"),
            })

    return result


def get_racial_traits(race_idx, subrace_idx=None):
    """
    Devuelve los rasgos de una raza y, opcionalmente, de su subraza,
    resolviendo sus datos desde la base de reglas de Magical20.
    """
    if not isinstance(race_idx, str):
        return None

    race_idx = race_idx.strip().lower()

    race = get_race(race_idx)

    if not race:
        return None

    result = []

    raw_race = race.get("raw")

    if isinstance(raw_race, dict):
        for trait in raw_race.get("traits", []):
            if not isinstance(trait, dict):
                continue

            idx = trait.get("index")

            if not isinstance(idx, str) or not idx:
                continue

            trait_data = get_trait(idx)

            if not trait_data:
                continue

            raw = trait_data.get("raw")
            if not isinstance(raw, dict):
                raw = {}

            desc = raw.get("desc", [])

            if not isinstance(desc, list):
                desc = [desc] if desc else []

            result.append({
                "index": idx,
                "name": trait_data.get("name") or trait.get("name") or idx,
                "source": "race",
                "description": desc,
                "proficiencies": raw.get("proficiencies", []),
                "trait_specific": raw.get("trait_specific"),
                "language_options": raw.get("language_options"),
            })

    if subrace_idx is not None:
        if not isinstance(subrace_idx, str):
            return None

        subrace_idx = subrace_idx.strip().lower()

        subrace = get_subrace(subrace_idx)

        if not subrace:
            return None

        subrace_raw = subrace.get("raw")

        if not isinstance(subrace_raw, dict):
            return None

        parent_race = subrace_raw.get("race", {}).get("index")

        if parent_race != race_idx:
            return None

        for trait in subrace_raw.get("racial_traits", []):
            if not isinstance(trait, dict):
                continue

            idx = trait.get("index")

            if not isinstance(idx, str) or not idx:
                continue

            trait_data = get_trait(idx)

            if not trait_data:
                continue

            raw = trait_data.get("raw")
            if not isinstance(raw, dict):
                raw = {}

            desc = raw.get("desc", [])

            if not isinstance(desc, list):
                desc = [desc] if desc else []

            result.append({
                "index": idx,
                "name": trait_data.get("name") or trait.get("name") or idx,
                "source": "subrace",
                "description": desc,
                "proficiencies": raw.get("proficiencies", []),
                "trait_specific": raw.get("trait_specific"),
                "language_options": raw.get("language_options"),
            })

    return result


def get_racial_ability_bonuses(race_idx, subrace_idx=None):
    """
    Devuelve los bonificadores de características de raza y subraza
    combinados, usando exclusivamente Magical20.
    """
    if not isinstance(race_idx, str):
        return None

    race_idx = race_idx.strip().lower()

    race = get_race(race_idx)

    if not race:
        return None

    result = {}

    def add_bonuses(raw):
        if not isinstance(raw, dict):
            return

        bonuses = raw.get("ability_bonuses", [])

        if not isinstance(bonuses, list):
            return

        for bonus in bonuses:
            if not isinstance(bonus, dict):
                continue

            ability = bonus.get("ability_score", {})
            if not isinstance(ability, dict):
                continue

            ability_idx = ability.get("index")
            value = bonus.get("bonus")

            if not isinstance(ability_idx, str):
                continue

            try:
                value = int(value)
            except (TypeError, ValueError):
                continue

            result[ability_idx] = result.get(ability_idx, 0) + value

    add_bonuses(race.get("raw"))

    if subrace_idx is not None:
        if not isinstance(subrace_idx, str):
            return None

        subrace_idx = subrace_idx.strip().lower()

        subrace = get_subrace(subrace_idx)

        if not subrace:
            return None

        subrace_raw = subrace.get("raw")

        if not isinstance(subrace_raw, dict):
            return None

        parent_race = subrace_raw.get("race", {}).get("index")

        if parent_race != race_idx:
            return None

        add_bonuses(subrace_raw)

    return result


def get_racial_skill_proficiencies(race_idx, subrace_idx=None):
    """
    Devuelve las competencias de habilidad otorgadas por la raza y subraza,
    usando exclusivamente los datos de Magical20.
    """
    if not isinstance(race_idx, str):
        return None

    race_idx = race_idx.strip().lower()

    race = get_race(race_idx)

    if not race:
        return None
    result = set()

    def collect_proficiencies(raw):
        if not isinstance(raw, dict):
            return

        for proficiency in raw.get("starting_proficiencies", []):
            if not isinstance(proficiency, dict):
                continue

            idx = proficiency.get("index")

            if isinstance(idx, str) and idx.startswith("skill-"):
                result.add(idx[6:])

    collect_proficiencies(race.get("raw"))

    if subrace_idx is not None:
        if not isinstance(subrace_idx, str):
            return None

        subrace_idx = subrace_idx.strip().lower()

        subrace = get_subrace(subrace_idx)

        if not subrace:
            return None

        subrace_raw = subrace.get("raw")

        if not isinstance(subrace_raw, dict):
            return None

        parent_race = subrace_raw.get("race", {}).get("index")

        if parent_race != race_idx:
            return None

        collect_proficiencies(subrace_raw)

    return sorted(result)


def get_total_skill_proficiencies(
    race_idx,
    subrace_idx=None,
    class_idx=None,
    selected_skills=None,
):
    """
    Devuelve las competencias de habilidad totales del personaje.

    Combina:
    - competencias otorgadas por raza/subraza;
    - habilidades elegidas mediante las opciones de competencia de la clase.

    Las elecciones de clase se validan contra la estructura de
    proficiency_choices de Magical20.
    """
    result = set()

    racial = get_racial_skill_proficiencies(
        race_idx,
        subrace_idx,
    )

    if racial is None:
        return None

    result.update(racial)

    if class_idx is None:
        if selected_skills is not None:
            valid_skills = get_skill_idx_set()

            if not isinstance(selected_skills, (list, tuple, set)):
                return None

            normalized = {
                str(skill).strip().lower()
                for skill in selected_skills
                if isinstance(skill, str)
            }

            if not normalized.issubset(valid_skills):
                return None

            result.update(normalized)

        return sorted(result)

    if not isinstance(class_idx, str):
        return None

    class_idx = class_idx.strip().lower()

    if not get_class(class_idx):
        return None

    if selected_skills is None:
        return sorted(result)

    if not isinstance(selected_skills, (list, tuple, set)):
        return None

    normalized_selected = [
        str(skill).strip().lower()
        for skill in selected_skills
        if isinstance(skill, str)
    ]

    if len(normalized_selected) != len(set(normalized_selected)):
        return None

    choices = get_class_proficiency_choices(class_idx)

    skill_choice = None

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        if choice.get("type") != "proficiencies":
            continue

        options = choice.get("from", {}).get("options", [])

        if not isinstance(options, list):
            continue

        skill_options = set()

        for option in options:
            if not isinstance(option, dict):
                continue

            item = option.get("item")

            if not isinstance(item, dict):
                continue

            idx = item.get("index")

            if isinstance(idx, str) and idx.startswith("skill-"):
                skill_options.add(idx[6:])

        if skill_options:
            skill_choice = {
                "choose": choice.get("choose"),
                "options": skill_options,
            }
            break

    if skill_choice is None:
        if normalized_selected:
            return None

        return sorted(result)

    choose = skill_choice.get("choose")

    try:
        choose = int(choose)
    except (TypeError, ValueError):
        return None

    if len(normalized_selected) != choose:
        return None

    allowed = skill_choice["options"]

    if not set(normalized_selected).issubset(allowed):
        return None

    result.update(normalized_selected)

    return sorted(result)


def get_skill_idx_set():
    """
    Devuelve el conjunto de índices de habilidades válidas
    almacenadas en la base de reglas de Magical20.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT idx FROM skills"
        ).fetchall()

        return {
            row[0]
            for row in rows
            if isinstance(row[0], str)
        }
    finally:
        conn.close()


def get_final_ability_scores(base_scores, race_idx, subrace_idx=None):
    """
    Combina las puntuaciones base con los bonificadores raciales/subraciales.
    Usa exclusivamente los datos de Magical20.
    """
    if not isinstance(base_scores, dict):
        return None

    required = {"str", "dex", "con", "int", "wis", "cha"}

    if not required.issubset(base_scores):
        return None

    final_scores = {}

    for ability in required:
        try:
            score = int(base_scores[ability])
        except (TypeError, ValueError):
            return None

        if score < 1 or score > 30:
            return None

        final_scores[ability] = score

    racial_bonuses = get_racial_ability_bonuses(
        race_idx,
        subrace_idx,
    )

    if racial_bonuses is None:
        return None

    for ability, bonus in racial_bonuses.items():
        if ability in final_scores:
            final_scores[ability] += bonus

    return final_scores


def get_final_ability_modifiers(base_scores, race_idx, subrace_idx=None):
    """
    Calcula los modificadores finales de las seis características,
    después de aplicar los bonificadores de raza y subraza.
    """
    final_scores = get_final_ability_scores(
        base_scores,
        race_idx,
        subrace_idx,
    )

    if final_scores is None:
        return None

    return {
        ability: get_ability_modifier(score)
        for ability, score in final_scores.items()
    }


def get_proficiency_bonus(class_idx, level):
    """
    Obtiene el bonificador de competencia desde la progresión
    oficial almacenada en Magical20.
    """
    level_data = get_level(class_idx, level)

    if not level_data:
        return None

    raw = level_data.get("raw")

    if not isinstance(raw, dict):
        return None

    bonus = raw.get("prof_bonus")

    try:
        return int(bonus)
    except (TypeError, ValueError):
        return None


def get_class_saving_throws(class_idx):
    """
    Devuelve los índices de las características en las que
    la clase tiene competencia en tiradas de salvación.
    """
    character_class = get_class(class_idx)

    if not character_class:
        return []

    raw = character_class.get("raw")

    if not isinstance(raw, dict):
        return []

    saving_throws = raw.get("saving_throws", [])

    if not isinstance(saving_throws, list):
        return []

    result = []

    for item in saving_throws:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")

        if idx:
            result.append(idx)

    return result


def get_class_proficiencies(class_idx):
    """
    Devuelve las competencias fijas de una clase.

    Magical20 utiliza principalmente el campo `proficiencies`,
    pero algunas clases, como Paladín, usan `competencias`.
    Ambos pertenecen al registro de la misma fuente.
    """
    character_class = get_class(class_idx)

    if not character_class:
        return []

    raw = character_class.get("raw")

    if not isinstance(raw, dict):
        return []

    proficiencies = raw.get("proficiencies")

    if not isinstance(proficiencies, list):
        proficiencies = raw.get("competencias", [])

    if not isinstance(proficiencies, list):
        return []

    result = []

    for item in proficiencies:
        if not isinstance(item, dict):
            continue

        idx = item.get("index")

        if isinstance(idx, str):
            result.append(idx)

    return result

def get_class_proficiency_choices(class_idx):
    """
    Devuelve las elecciones de competencia que ofrece la clase
    durante la creación del personaje.
    """
    character_class = get_class(class_idx)

    if not character_class:
        return []

    raw = character_class.get("raw")

    if not isinstance(raw, dict):
        return []

    choices = raw.get("proficiency_choices", [])

    if not isinstance(choices, list):
        return []

    return choices


def get_translated_name(entity_type, entity_idx, fallback=None):
    """
    Obtiene el nombre traducido al español.
    Si no existe traducción, usa el nombre de la entidad
    almacenado en Magical20 o el fallback indicado.
    """
    translation = get_translation(
        entity_type,
        entity_idx,
        language="es",
        field="name",
    )

    if translation and translation.get("text"):
        return translation["text"]

    table_map = {
        "abilities": "abilities",
        "skills": "skills",
        "classes": "classes",
        "races": "races",
        "subraces": "subraces",
        "conditions": "conditions",
        "equipment": "equipment",
        "spells": "spells",
        "features": "features",
        "proficiencies": "proficiencies",
    }

    table = table_map.get(entity_type)

    if table:
        row = _get_one(
            f"""
            SELECT name
            FROM {table}
            WHERE idx = ?
            """,
            (entity_idx,),
        )

        if row and row.get("name"):
            return row["name"]

    return fallback if fallback is not None else entity_idx


def get_ability_name(idx):
    """
    Nombre completo en español de una característica.
    """
    ability = get_ability(idx)

    if not ability:
        return idx

    return ability.get("full_name") or ability.get("name") or idx


def get_skill_name(idx):
    return get_translated_name("skills", idx)


def get_class_name(idx):
    return get_translated_name("classes", idx)


def get_race_name(idx):
    return get_translated_name("races", idx)


def get_condition_name(idx):
    return get_translated_name("conditions", idx)


def get_translation(entity_type, entity_idx, language="es", field="name"):
    return _get_one(
        """
        SELECT text
        FROM translations
        WHERE entity_type = ?
          AND entity_idx = ?
          AND language = ?
          AND field = ?
        """,
        (entity_type, entity_idx, language, field),
    )


if __name__ == "__main__":
    print("D&D RULES DATABASE TEST")
    print("=" * 60)

    tests = [
        ("ability", get_ability("str")),
        ("skill", get_skill("perception")),
        ("class", get_class("bard")),
        ("level", get_level("bard", 5)),
        ("race", get_race("elf")),
        ("feature", get_feature("rage")),
        ("proficiency", get_proficiency("light-armor")),
        ("condition", get_condition("poisoned")),
        ("equipment", get_equipment("dagger")),
        ("spell", get_spell("acid-arrow")),
        ("translation", get_translation("skills", "perception")),
    ]

    for name, result in tests:
        print(f"\n{name}:")
        print("OK" if result is not None else "NO ENCONTRADO")

    print("\nBase:", DB_PATH)


def get_ability_modifier(score):
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None

    if score < 1 or score > 30:
        return None

    return (score - 10) // 2


def get_skill_ability_idx(skill_idx):
    skill = get_skill(skill_idx)
    if not skill:
        return None

    raw = skill.get("raw")
    if not isinstance(raw, dict):
        return None

    ability = raw.get("ability_score")
    if not isinstance(ability, dict):
        return None

    return ability.get("index")


def get_class_skill_proficiency_options(class_idx):
    choices = get_class_proficiency_choices(class_idx)
    result = []

    for choice in choices:
        if not isinstance(choice, dict):
            continue

        if choice.get("type") != "proficiencies":
            continue

        options = choice.get("from", {}).get("options", [])
        if not isinstance(options, list):
            continue

        for option in options:
            if not isinstance(option, dict):
                continue

            item = option.get("item")
            if not isinstance(item, dict):
                continue

            idx = item.get("index")
            if isinstance(idx, str) and idx.startswith("skill-"):
                result.append(idx[6:])

    return list(dict.fromkeys(result))


def get_skill_modifier(
    skill_idx,
    ability_scores,
    class_idx,
    level,
    proficient_skills=None,
):
    if not isinstance(ability_scores, dict):
        return None

    ability_idx = get_skill_ability_idx(skill_idx)
    if not ability_idx:
        return None

    score = ability_scores.get(ability_idx)
    modifier = get_ability_modifier(score)

    if modifier is None:
        return None

    if proficient_skills is None:
        proficient_skills = []

    normalized_skills = {
        str(skill).strip().lower()
        for skill in proficient_skills
    }

    if skill_idx in normalized_skills:
        proficiency = get_proficiency_bonus(class_idx, level)

        if proficiency is None:
            return None

        modifier += proficiency

    return modifier


def get_saving_throw_modifier(
    ability_idx,
    ability_scores,
    class_idx,
    level,
):
    if not isinstance(ability_scores, dict):
        return None

    if ability_idx not in ("str", "dex", "con", "int", "wis", "cha"):
        return None

    score = ability_scores.get(ability_idx)
    modifier = get_ability_modifier(score)

    if modifier is None:
        return None

    saving_throws = get_class_saving_throws(class_idx)

    if ability_idx in saving_throws:
        proficiency = get_proficiency_bonus(class_idx, level)

        if proficiency is None:
            return None

        modifier += proficiency

    return modifier


def get_class_proficiency_options(class_idx):
    """
    Devuelve las elecciones de competencia disponibles para una clase,
    conservando elecciones simples y anidadas.

    Usa exclusivamente la estructura de proficiency_choices
    almacenada en Magical20.
    """
    if not isinstance(class_idx, str):
        return None

    class_idx = class_idx.strip().lower()

    if not class_idx or not get_class(class_idx):
        return None

    choices = get_class_proficiency_choices(class_idx)

    if not isinstance(choices, list):
        return None

    def parse_choice(choice):
        if not isinstance(choice, dict):
            return None

        result = {
            "type": choice.get("type"),
            "choose": choice.get("choose"),
            "desc": choice.get("desc"),
            "options": [],
            "nested_choices": [],
        }

        source = choice.get("from", {})

        if not isinstance(source, dict):
            return result

        options = source.get("options", [])

        if not isinstance(options, list):
            return result

        for option in options:
            if not isinstance(option, dict):
                continue

            option_type = option.get("option_type")

            if option_type == "reference":
                item = option.get("item")

                if not isinstance(item, dict):
                    continue

                idx = item.get("index")

                if not isinstance(idx, str):
                    continue

                result["options"].append({
                    "index": idx,
                    "name": item.get("name"),
                })

            elif option_type == "choice":
                nested = parse_choice(option.get("choice"))

                if nested is not None:
                    result["nested_choices"].append(nested)

        return result

    result = []

    for choice in choices:
        parsed = parse_choice(choice)

        if parsed is not None:
            result.append(parsed)

    return result


def validate_class_proficiency_selection(
    class_idx,
    choice_index,
    selected,
):
    """
    Valida una selección de competencias de una clase.

    Comprueba:
    - que la clase exista;
    - que la elección exista;
    - que la cantidad seleccionada coincida con `choose`;
    - que no haya duplicados;
    - que cada competencia exista en la base;
    - que cada competencia pertenezca a las opciones permitidas.

    Usa exclusivamente la estructura de Magical20.
    """
    if not isinstance(class_idx, str):
        return False

    class_idx = class_idx.strip().lower()

    if not get_class(class_idx):
        return False

    try:
        choice_index = int(choice_index)
    except (TypeError, ValueError):
        return False

    if choice_index < 1:
        return False

    choices = get_class_proficiency_choices(class_idx)

    if not isinstance(choices, list):
        return False

    if choice_index > len(choices):
        return False

    if not isinstance(selected, (list, tuple, set)):
        return False

    selected = [
        str(item).strip().lower()
        for item in selected
        if isinstance(item, str)
    ]

    if len(selected) != len(set(selected)):
        return False

    choice = choices[choice_index - 1]

    try:
        choose = int(choice.get("choose"))
    except (TypeError, ValueError):
        return False

    if len(selected) != choose:
        return False

    allowed = set()

    def collect_references(current_choice):
        if not isinstance(current_choice, dict):
            return

        source = current_choice.get("from", {})

        if not isinstance(source, dict):
            return

        options = source.get("options", [])

        if not isinstance(options, list):
            return

        for option in options:
            if not isinstance(option, dict):
                continue

            option_type = option.get("option_type")

            if option_type == "reference":
                item = option.get("item")

                if not isinstance(item, dict):
                    continue

                idx = item.get("index")

                if isinstance(idx, str):
                    allowed.add(idx)

            elif option_type == "choice":
                nested = option.get("choice")
                collect_references(nested)

    collect_references(choice)

    if not set(selected).issubset(allowed):
        return False

    for idx in selected:
        proficiency = get_proficiency(idx)

        if not proficiency:
            return False

    return True
