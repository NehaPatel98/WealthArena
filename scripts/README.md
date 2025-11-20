# WealthArena Scripts

Utility scripts for managing the WealthArena development environment.

## sync_db_password.ps1

Synchronizes the database password from `data-pipeline/sqlDB.env` to `backend/.env.local`.

### Usage

```powershell
# From project root
.\scripts\sync_db_password.ps1
```

### What it does

1. Reads `SQL_PWD` from `data-pipeline/sqlDB.env`
2. Updates `DB_PASSWORD` in `backend/.env.local`
3. Optionally updates CORS origins to include your local IP
4. Creates `backend/.env.local` if it doesn't exist

### When to use

- After updating the password in `sqlDB.env`
- When `backend/.env.local` has a placeholder password
- To ensure password consistency across services
- After running the main setup script if password sync failed

### Example

```powershell
# Update password in sqlDB.env first
# SQL_PWD=admin

# Then sync to backend
.\scripts\sync_db_password.ps1

# Output:
# ========================================
# Database Password Sync Utility
# ========================================
# Reading password from data-pipeline/sqlDB.env...
#   Found password: admin
# Updating backend/.env.local...
#   Updated DB_PASSWORD from 'your-postgres-password' to 'admin'
# ========================================
# Password Sync Complete!
# ========================================
```

### Notes

- The script preserves all other settings in `backend/.env.local`
- It only updates `DB_PASSWORD` and `CORS_ORIGINS`/`ALLOWED_ORIGINS`
- If `sqlDB.env` doesn't have a valid password, the script will exit with an error
- The backend server must be restarted after syncing for changes to take effect

