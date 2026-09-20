# Security Quick Reference Guide

Quick security guide for developers working on this Discord D&D Bot.

---

## ✅ What's Already Secure (Keep Doing This!)

### 1. Environment Variables for Secrets
**Always load credentials from environment variables, never hardcode them:**

```python
# ✅ GOOD - Already doing this
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
API_KEY = os.getenv("API_KEY", "").strip()

# ❌ BAD - Never do this
DISCORD_TOKEN = "MTEx..."  # Hardcoded
API_KEY = "sk-..."  # Hardcoded
```

### 2. Parameterized SQL Queries
**Always use parameterized queries with `?` placeholders:**

```python
# ✅ GOOD - Already doing this
conn.execute(
    "SELECT * FROM users WHERE user_id = ?",
    (user_id,)  # Tuple of parameters
)

# ❌ BAD - SQL injection risk
conn.execute(f"SELECT * FROM users WHERE user_id = {user_id}")
conn.execute("SELECT * FROM users WHERE user_id = " + user_id)
```

### 3. Safe Expression Evaluation
**Use AST parsing for mathematical expressions, not eval():**

```python
# ✅ GOOD - Already doing this in local_math()
import ast
tree = ast.parse(expr, mode="eval")
# ... validate and evaluate safely

# ❌ BAD - Code injection risk
result = eval(user_input)
result = exec(user_input)
```

---

## 🔧 Recommended Improvements

### 1. Add Rate Limiting to Commands

**Why**: Prevents abuse and resource exhaustion

```python
from discord.ext import commands

@bot.command()
@commands.cooldown(1, 5, commands.BucketType.user)  # 1 use per 5 seconds
async def my_command(ctx):
    """Your command with rate limiting"""
    # Implementation
    pass

# Handle cooldown errors
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Comando en espera. Intenta en {error.retry_after:.1f}s")
```

### 2. Use Structured Logging

**Why**: Better control over what gets logged in production

```python
import logging

# Setup at module level
logging.basicConfig(
    level=logging.INFO,  # Change to WARNING in production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Replace print() statements:
logger.debug("Detailed info for development")  # Only in debug mode
logger.info("General information")
logger.warning("Something unusual happened")
logger.error("Error occurred", exc_info=True)  # Includes stack trace

# Production environment variable:
# export LOG_LEVEL=WARNING
```

### 3. Input Validation Example

**Why**: Defense in depth (additional layer of protection)

```python
def validate_campaign_id(campaign_id):
    """Validate campaign ID format"""
    if not isinstance(campaign_id, str):
        raise ValueError("Campaign ID must be a string")
    
    if not re.match(r'^[a-f0-9]{16}$', campaign_id):
        raise ValueError("Invalid campaign ID format")
    
    return campaign_id

# Use before database operations
campaign_id = validate_campaign_id(user_input)
```

---

## 🚨 Never Do These

### 1. Never Use eval() or exec() on User Input

```python
# ❌ EXTREMELY DANGEROUS
user_code = message.content
eval(user_code)   # Can execute arbitrary code!
exec(user_code)   # Can execute arbitrary code!

# ✅ Use AST parsing instead (see local_math() function)
```

### 2. Never Concatenate User Input into SQL

```python
# ❌ SQL INJECTION VULNERABILITY
query = f"SELECT * FROM users WHERE name = '{user_name}'"
query = "SELECT * FROM users WHERE name = '" + user_name + "'"

# ✅ Use parameterized queries
query = "SELECT * FROM users WHERE name = ?"
conn.execute(query, (user_name,))
```

### 3. Never Hardcode Secrets

```python
# ❌ SECURITY BREACH
API_KEY = "sk-1234567890abcdef"
PASSWORD = "mypassword123"

# ✅ Use environment variables
API_KEY = os.getenv("API_KEY")
PASSWORD = os.getenv("PASSWORD")
```

### 4. Never Use os.system() or subprocess with User Input

```python
# ❌ COMMAND INJECTION
os.system(f"echo {user_input}")
subprocess.call(["command", user_input])  # Still risky without proper validation

# ✅ Avoid shell commands, use Python libraries instead
# If absolutely necessary, validate input strictly
```

---

## 🔍 Code Review Checklist

Before committing code, check:

- [ ] No hardcoded secrets (API keys, passwords, tokens)
- [ ] All SQL queries use parameterized statements (`?` placeholders)
- [ ] No `eval()` or `exec()` on user-provided data
- [ ] No `os.system()` or unvalidated `subprocess` calls
- [ ] Input validation for user-provided data
- [ ] Rate limiting on resource-intensive operations
- [ ] Error messages don't reveal sensitive information
- [ ] Using `logger` instead of `print()` for production

---

## 📚 Additional Resources

- **OWASP Top 10**: https://owasp.org/www-project-top-ten/
- **CWE-89 SQL Injection**: https://cwe.mitre.org/data/definitions/89.html
- **CWE-94 Code Injection**: https://cwe.mitre.org/data/definitions/94.html
- **CWE-798 Hardcoded Credentials**: https://cwe.mitre.org/data/definitions/798.html
- **Python Security Best Practices**: https://docs.python.org/3/library/security_warnings.html

---

## 🎯 Summary

**Current Security Status**: ✅ GOOD

Your codebase already follows most security best practices. The main areas for improvement are:

1. **Add rate limiting** to prevent command spam
2. **Use structured logging** instead of print statements
3. **Add input validation** for defense in depth

These are enhancements, not critical fixes. Your current code is secure for production use.

---

**Questions?** Review the detailed `SECURITY_RECOMMENDATIONS.md` file for more information.
