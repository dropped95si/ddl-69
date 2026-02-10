# DDL-420-69 - Complete Deployment Setup Guide

## Status: READY FOR PRODUCTION ✓

All security vulnerabilities fixed. Just need to add your API keys and deploy.

---

## PART 1: Get Your API Keys (15 minutes)

### 1. SUPABASE (Required)

**What it is**: Postgres database + authentication + storage  
**Cost**: Free tier available  

Steps:
1. Go to https://app.supabase.com
2. Click "New Project"
3. Choose:
   - Name: `ddl-420-69` or similar
   - Database password: `GenerateStrong!Password123`
   - Region: Pick closest to you
4. Click "Create new project" (wait 2-3 minutes)
5. Go to **Settings → API**
6. Copy:
   - **Project URL** → Add to `.env` as `SUPABASE_URL`
   - **Service Role Secret** → Add as `SUPABASE_SERVICE_ROLE_KEY`
   - **Anon Public** → Save for later (Vercel env var)

**Run schema setup**:
```bash
psql -h [your-host] -U postgres -d postgres < sql/ledger_v1.sql
psql -h [your-host] -U postgres -d postgres < sql/ledger_v2_patch.sql
psql -h [your-host] -U postgres -d postgres < sql/ingest_v1.sql
```

---

### 2. POLYGON.IO (Optional - for market data)

**What it is**: Real-time stock market data API  
**Cost**: Free for basic usage  

Steps:
1. Go to https://polygon.io
2. Click "Get Free API Key"
3. Sign up with email
4. Go to **Dashboard → API Keys**
5. Copy your API key
6. Add to `.env`:
   ```
   POLYGON_API_KEY=pk_your_key_here
   ```

---

### 3. ALPACA MARKETS (Optional - for trading data)

**What it is**: Stock/crypto trading platform API  
**Cost**: Free paper trading  

Steps:
1. Go to https://alpaca.markets
2. Click "Sign Up"
3. Create account
4. Go to **Account → Credentials**
5. Copy:
   - **API Key** → Add to `.env` as `ALPACA_API_KEY`
   - **Secret Key** → Add as `ALPACA_SECRET_KEY`
6. Keep URL as: `https://paper-api.alpaca.markets` (paper trading is free)

---

### 4. DISCORD (Optional - for message ingestion)

**What it is**: Discord bot for pulling messages  
**Cost**: Free  

Steps:
1. Go to https://discord.com/developers/applications
2. Click "New Application"
3. Go to **Bot** tab
4. Click "Add Bot"
5. Under TOKEN, click "Copy"
6. Add to `.env` as `DISCORD_TOKEN`
7. Create a test server to test with

---

## PART 2: Setup Your .env File (5 minutes)

Your `.env` file is ready at: `DDL-420-69/.env`

Fill in the values you just copied:

```bash
# Required
SUPABASE_URL=https://abc123.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJhbGc...
SUPABASE_STORAGE_BUCKET=artifacts

# Optional but recommended
POLYGON_API_KEY=pk_abc123...
ALPACA_API_KEY=ABC123...
ALPACA_SECRET_KEY=XYZ789...

# Optional
DISCORD_TOKEN=ODk2Nzc...
```

**Security**: 
- Never commit `.env` to Git
- Never share these keys
- Rotate keys if compromised

---

## PART 3: Local Testing (10 minutes)

Test that everything works locally:

```bash
cd DDL-420-69

# 1. Test CLI
python -m ddl69.cli.main help

# 2. Test tools status
python -m ddl69.cli.main tools_status

# 3. Test database connection
python -m pytest tests/ -v

# 4. Test imports with your keys
python << 'PYEOF'
import os
from ddl69.core.settings import Settings
from ddl69.ledger.supabase_ledger import SupabaseLedger

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Create settings
s = Settings.from_env()
print(f"Supabase URL: {s.supabase_url[:30]}...")
print(f"Has Polygon key: {bool(s.polygon_api_key)}")
print(f"Has Alpaca key: {bool(s.alpaca_api_key)}")
print("All keys loaded successfully!")
PYEOF
```

---

## PART 4: Deploy to Vercel (5 minutes)

### Option A: Web Dashboard (Easiest)

1. Go to https://vercel.com/new
2. Select **Other**
3. Paste repo: `https://github.com/dropped95si/DDL-420-69`
4. Click **Import**
5. Go to **Settings → Environment Variables**
6. Add variables:
   ```
   SUPABASE_URL=https://abc123.supabase.co
   SUPABASE_ANON_KEY=eyJhbGc... (from Supabase API page)
   ```
7. Click **Deploy**
8. Wait for build to complete
9. Copy deployment URL: `https://ddl-420-69-abc.vercel.app`

### Option B: CLI

```bash
npm install -g vercel
cd DDL-420-69
vercel --prod
```

---

## PART 5: Verify Deployment

Test your live deployment:

```bash
# Test API endpoints
curl https://your-vercel-url.vercel.app/

# Test database connection
curl https://your-vercel-url.vercel.app/api/health

# View logs
vercel logs --follow
```

---

## PART 6: Configure Custom Domain (Optional)

In Vercel Dashboard:
1. Go to **Settings → Domains**
2. Click **Add Domain**
3. Enter your domain (e.g., `ddl69.com`)
4. Follow DNS setup instructions

---

## PART 7: Set Up Monitoring

Enable in Vercel Dashboard:
- **Settings → Monitoring**
- Configure alerts for errors
- Set up log aggregation

---

## Quick Reference: What You'll Have

After setup, you'll have:

```
┌─────────────────────────────────────┐
│   Frontend (Vercel)                 │
│   https://your-vercel-url.vercel.app│
│   (Static HTML + JS Dashboard)      │
└──────────┬──────────────────────────┘
           │
           ├─ API Calls
           │
┌──────────▼──────────────────────────┐
│   Backend (Vercel Serverless)       │
│   Python CLI + API Functions        │
└──────────┬──────────────────────────┘
           │
           ├─ Reads/Writes
           │
┌──────────▼──────────────────────────┐
│   Database (Supabase Postgres)      │
│   Ledger, runs, forecasts, etc      │
└──────────┬──────────────────────────┘
           │
           ├─ Fetches Data
           │
┌──────────▼──────────────────────────┐
│   Market Data APIs                  │
│   Polygon, Alpaca, Yahoo Finance    │
└─────────────────────────────────────┘
```

---

## Troubleshooting

### "API key not found"
- Check `.env` file exists and has correct values
- Run `python -c "import os; print(os.getenv('SUPABASE_URL'))"`

### "Failed to connect to Supabase"
- Verify SUPABASE_URL is correct
- Check network connectivity

### "Vercel deployment failed"
- Check build logs: `vercel logs`
- Verify environment variables are set in Vercel dashboard
- Ensure all required dependencies installed

### "Tests failing"
- Run `pytest tests/ -v` locally
- All 7 tests should pass

---

## Next Steps After Deployment

1. ✓ Test CLI commands in production
2. ✓ Monitor Vercel logs for errors
3. ✓ Create Discord notifications on errors
4. ✓ Set up automated backups
5. ✓ Configure rate limiting

---

## Support

- **GitHub Issues**: https://github.com/dropped95si/DDL-420-69/issues
- **Documentation**: See SECURITY_FIXES.md and README.md
- **Deployment**: See DEPLOY_VERCEL.md

---

**Deployment Status**: READY ✓
**Last Updated**: 2026-02-10
**Version**: v0.2.0
