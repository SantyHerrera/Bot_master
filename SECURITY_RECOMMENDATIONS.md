# Security Audit Report - D&D Discord Bot

**Audit Date**: 2024
**Files Reviewed**: bot_master.py (9,796 lines), dnd_rules.py (4,460 lines)
**Overall Security Rating**: ✅ **GOOD** - No critical vulnerabilities found

---

## Executive Summary

This Discord bot demonstrates **strong security practices** with proper credential management, SQL injection protection, and secure expression evaluation. The code follows security best practices and is production-ready with minor recommended improvements for operational security.

### Key Findings:
- ✅ All API keys and tokens properly externalized to environment variables
- ✅ SQL queries use parameterized statements (protected against SQL injection - CWE-89)
- ✅ No use of dangerous functions (eval/exec on user input, os.system, subprocess)
- ✅ Safe mathematical expression evaluation using AST parsing
- ⚠️ Consider adding rate limiting for Discord commands
- ⚠️ Verbose logging may expose internal details in production

---

## Detailed Security Analysis

### 1. Credential Management ✅ SECURE

**Status**: COMPLIANT
**CWE Reference**: CWE-798 (Use of Hard-coded Credentials)

All sensitive credentials are properly loaded from environment variables:

```python
# bot_master.py lines 117-183
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEYS = [x.strip() for x in [
    os.getenv("GEMINI_API_KEY_1", ""),
    os.getenv("GEMINI_API_KEY_2", ""),
    os.getenv("GEMINI_API_KEY_3", ""),
] if x.strip()]
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "").strip()
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "").strip()
```

**Validation**: The bot validates the Discord token before starting (line 9792-9793):

```python
if not DISCORD_TOKEN:
    raise RuntimeError("DISCORD_TOKEN no está definido.")
```

**Recommendation**: ✅ No changes needed. This is the correct approach.

---

### 2. SQL Injection Protection ✅ SECURE

**Status**: COMPLIANT
**CWE Reference**: CWE-89 (SQL Injection)

#### dnd_rules.py - All queries use parameterized statements:

```python
# Example from line 40-47
def get_ability(idx):
    return _get_one(
        """
        SELECT *
        FROM abilities
        WHERE idx = ?
        """,
        (idx,),  # ✅ Parameterized
    )
```

#### bot_master.py - Campaign history queries are safe:

```python
# Example from line 623-631
cur = conn.execute(
    """
    INSERT OR IGNORE INTO campaign_history(
        event_key, campaign_id, timestamp, 
        event_type, actor_id, actor_name, content
    )
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """,
    (key, campaign_id, msg["timestamp"], msg.get("event_type", "message"),
     msg.get("actor_id", ""), msg.get("actor_name", ""), content),
)  # ✅ All values parameterized
```

#### Database Migration Code (Lines 538-558):

While technically secure (uses hardcoded dictionary values), the code could be clearer:

**Current Code** (bot_master.py:538-558):
```python
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
```

**Recommended Enhancement** (for code clarity):
```python
# Define valid tables as constants for clarity
VALID_TABLES = {"campaign_history", "state_snapshots", "important_memory"}

migrations = {
    "campaign_history": "campaign_id",
    "state_snapshots": "campaign_id",
    "important_memory": "campaign_id",
}

for table, column in migrations.items():
    # SECURITY: table and column names are from hardcoded dict, not user input
    assert table in VALID_TABLES, f"Invalid table: {table}"
    assert re.match(r'^[a-z_]+$', column), f"Invalid column: {column}"
    
    columns = [
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table})")
    ]
    # ... rest of code
```

**Risk Level**: ⚠️ LOW (Already secure, but recommendation improves code maintainability)

---

### 3. Safe Expression Evaluation ✅ SECURE

**Status**: COMPLIANT
**CWE Reference**: CWE-94 (Improper Control of Generation of Code)

The `local_math()` function (line 8279) properly uses AST parsing instead of `eval()`:

```python
def local_math(text):
    import ast, operator
    m = re.search(r"(?<![A-Za-z])([0-9][0-9\s+\-*/().%]*[0-9])", text)
    if not m: return None
    expr = m.group(1).replace(" ", "")
    
    # ✅ Input validation
    if len(expr) > 80 or not re.fullmatch(r"[0-9+*/().%-]+", expr):
        return None
    
    # ✅ Whitelist of allowed operations
    ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod
    }
    
    try:
        tree = ast.parse(expr, mode="eval")  # ✅ AST parsing (safe)
        # ... custom evaluator that only allows whitelisted operations
```

**Analysis**: This is a **secure implementation**. The code:
1. ✅ Validates input length (max 80 characters)
2. ✅ Uses regex to restrict characters to numbers and math operators only
3. ✅ Uses AST parsing instead of eval()
4. ✅ Implements custom evaluator with whitelisted operations
5. ✅ Handles division by zero

**Recommendation**: ✅ No changes needed. This is the correct approach for safe expression evaluation.

---

### 4. Information Disclosure (Logging) ⚠️ REVIEW RECOMMENDED

**Status**: REVIEW RECOMMENDED
**CWE Reference**: CWE-209 (Information Exposure Through Error Messages)
**Risk Level**: MEDIUM (for production environments)

**Issue**: Detailed API usage and error information is printed to console, which may expose:
- Token usage patterns
- API rate limits
- Model configurations
- Internal system structure

**Examples**:
```python
# Line 8123
print(f"[GROQ] {model}: in={usage.get('prompt_tokens','?')} "
      f"out={usage.get('completion_tokens','?')} "
      f"día={groq_daily_used(model)}/{GROQ_DAILY_BUDGET_PER_MODEL}")
```

**Recommendation**: Implement structured logging with appropriate levels:

```python
import logging

# At module level
logging.basicConfig(
    level=logging.INFO,  # Set to WARNING in production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace print statements:
logger.debug(f"[GROQ] Token usage: {usage}")  # Only shown in debug mode
logger.info("API request completed")  # Generic info for production
logger.error("API request failed", exc_info=True)  # Errors with context
```

---

### 5. Rate Limiting ⚠️ RECOMMENDED

**Status**: NOT IMPLEMENTED
**CWE Reference**: CWE-770 (Allocation of Resources Without Limits)
**Risk Level**: LOW-MEDIUM

**Issue**: No explicit rate limiting visible for Discord commands, which could lead to:
- Bot abuse
- Resource exhaustion
- Unintended API costs

**Recommendation**: Add command cooldowns using discord.py decorators:

```python
from discord.ext import commands

@bot.command()
@commands.cooldown(1, 5, commands.BucketType.user)  # 1 use per 5 seconds per user
async def roll(ctx, dice: str):
    """Roll dice with rate limiting"""
    # Command implementation
    pass
```

---

## Summary of Recommendations

### Immediate Actions (Optional - No Critical Issues)
None required. The code is secure for production use as-is.

### Recommended Improvements

| Priority | Area | Recommendation | Risk if Not Implemented |
|----------|------|----------------|------------------------|
| Medium | Logging | Implement structured logging with levels | Information disclosure in logs |
| Medium | Rate Limiting | Add command cooldowns | Potential abuse/resource exhaustion |
| Low | Code Clarity | Add comments to database migration code | Developer confusion (no security risk) |
| Low | Input Validation | Add explicit validation (defense in depth) | None (already protected by parameterization) |

---

## Security Checklist

- [x] No hardcoded credentials (CWE-798)
- [x] SQL injection protection via parameterized queries (CWE-89)
- [x] No dangerous code execution (eval/exec on user input) (CWE-94)
- [x] No command injection vulnerabilities (CWE-78)
- [x] Proper database connection handling
- [x] Safe expression evaluation using AST
- [ ] Rate limiting on user commands (CWE-770) - **Recommended**
- [ ] Structured logging with appropriate levels (CWE-209) - **Recommended**

---

## Compliance Notes

### OWASP Top 10 Coverage:
- **A03:2021 - Injection**: ✅ Protected (parameterized queries)
- **A07:2021 - Identification and Authentication Failures**: ✅ Proper credential management
- **A09:2021 - Security Logging and Monitoring Failures**: ⚠️ Could be improved with structured logging

### Security Standards:
- CWE-89 (SQL Injection): ✅ COMPLIANT
- CWE-94 (Code Injection): ✅ COMPLIANT  
- CWE-798 (Hardcoded Credentials): ✅ COMPLIANT
- CWE-209 (Information Exposure): ⚠️ REVIEW RECOMMENDED
- CWE-770 (Resource Exhaustion): ⚠️ IMPROVEMENT RECOMMENDED

---

## Conclusion

**Overall Security Posture**: ✅ **GOOD**

This Discord bot demonstrates solid security practices and is suitable for production use. The code properly handles sensitive data, prevents SQL injection, and avoids common code injection vulnerabilities. 

The recommended improvements focus on operational security (logging) and user experience (rate limiting) rather than addressing critical security flaws.

**Recommended Next Steps**:
1. Implement structured logging for production environments
2. Add rate limiting to prevent command abuse
3. Review and sanitize error messages displayed to users
4. Consider adding input validation as defense-in-depth

---

**Auditor Notes**: This codebase shows evidence of security-conscious development practices. The developers have properly externalized credentials, used parameterized database queries throughout, and implemented safe expression evaluation. The recommendations provided are enhancements for operational security and code maintainability rather than fixes for security vulnerabilities.
