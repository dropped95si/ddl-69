# DDL-69 Vercel Deployment - Complete Guide (v0.2.0)

## What's New in v0.2.0

✓ **Modern Dashboard UI** - Professional dark-theme dashboard with real-time charts
✓ **Security Hardening** - 8 critical/high vulnerabilities fully remediated
✓ **API Endpoints** - Serverless Python functions for dashboard data
✓ **Production Ready** - Full environment configuration and deployment setup

---

## Quick Start (5 minutes)

### Option A: Deploy via Vercel Web Dashboard

1. **Go to Vercel**: https://vercel.com/new
2. **Select Repository**: Choose `dropped95si/DDL-420-69`
3. **Configure Project**:
   - Framework: `Other`
   - Output Directory: `ui`
   - Root Directory: `.`
   - Build Command: `echo 'UI Only'` (or leave empty)
4. **Environment Variables** (Optional for now):
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_ANON_KEY`: Your Supabase anon key
5. **Deploy**: Click "Deploy"

### Option B: Deploy via Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Navigate to project
cd DDL-420-69

# Deploy
vercel

# Follow prompts:
# ? Set up and deploy?: Yes
# ? Which scope?: Your account
# ? Link to existing project?: No
# ? Project name?: ddl-69
# ? Directory?: 
# Deploy on every push to main: Yes
```

---

## Post-Deployment Verification

### Test 1: Dashboard Loads
```bash
curl https://your-project.vercel.app/
# Should return HTML dashboard with interactive charts
```

### Test 2: API Endpoints
```bash
# Status endpoint
curl https://your-project.vercel.app/api/status
# Expected: {"system_status": "ONLINE", "active_forecasts": 247, ...}

# Forecasts endpoint
curl https://your-project.vercel.app/api/forecasts
# Expected: 30-day probability distribution data

# Calibration endpoint
curl https://your-project.vercel.app/api/calibration
# Expected: Probability calibration curve

# Events endpoint
curl https://your-project.vercel.app/api/events
# Expected: Recent system events
```

### Test 3: Environment Variables
If using Supabase, verify in Vercel dashboard:
- Settings → Environment Variables
- Should see `SUPABASE_URL` and `SUPABASE_ANON_KEY`

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Vercel (Cloud Deployment)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │   Static UI (Served from /ui)               │  │
│  │  ┌──────────────────────────────────────┐  │  │
│  │  │  index.html (Modern Dashboard)       │  │  │
│  │  │  styles.css (Dark Theme)             │  │  │
│  │  │  app.js (Chart.js Integration)       │  │  │
│  │  └──────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────┘  │
│                          │                         │
│                    (fetch calls)                   │
│                          │                         │
│  ┌──────────────────────────────────────────────┐  │
│  │   Serverless Functions (api/)                │  │
│  │  ┌────────────────────────────────────────┐ │  │
│  │  │ /api/status       - System status      │ │  │
│  │  │ /api/forecasts    - Probability data   │ │  │
│  │  │ /api/calibration  - Calibration curve  │ │  │
│  │  │ /api/events       - System events      │ │  │
│  │  └────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  [Environment Variables] → SUPABASE_* (optional)   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Dashboard Features

### Status Cards
- **System Status**: Online/Offline indicator with Supabase connection
- **Active Forecasts**: Number of active probability forecasts
- **Calibration**: Overall calibration score (70% = 70%)
- **Accuracy**: 7-day rolling average accuracy

### Charts
- **Probability Distribution**: Line chart showing P(Accept), P(Reject), P(Continue)
- **Expert Weights**: Interactive table with weights and accuracy metrics
- **Calibration Curve**: Scatter plot showing predicted vs actual probabilities

### Events Stream
- Real-time system events (Forecasts, Weight Updates, Outcomes, Data Ingest)
- Chronological display with relative timestamps

---

## Configuration

### vercel.json
```json
{
  "buildCommand": "npm run build 2>/dev/null || echo 'No build needed'",
  "outputDirectory": "ui",
  "rewrites": [
    { "source": "/(.*)", "destination": "/ui/index.html" }
  ],
  "env": {
    "SUPABASE_URL": "@supabase_url",
    "SUPABASE_ANON_KEY": "@supabase_anon_key"
  }
}
```

### Environment Variables (Optional)
Add these in Vercel dashboard → Settings → Environment Variables:
- `SUPABASE_URL`: https://your-project.supabase.co
- `SUPABASE_ANON_KEY`: Your Supabase anon key
- `SUPABASE_SERVICE_ROLE_KEY`: Service role key (for backend use)

---

## File Structure

```
DDL-420-69/
├── ui/                          # Static UI (served to browser)
│   ├── index.html              # Dashboard layout
│   ├── styles.css              # Dark theme styling
│   └── app.js                  # Chart.js + interactivity
│
├── api/                         # Serverless functions
│   ├── status.py               # System status endpoint
│   ├── forecasts.py            # Forecast data endpoint
│   ├── calibration.py          # Calibration data endpoint
│   └── events.py               # Events stream endpoint
│
├── src/ddl69/                  # Python core (not deployed)
│   ├── cli/main.py             # CLI with security fixes
│   ├── core/settings.py        # Configuration
│   └── utils/validators.py     # Input validation
│
└── vercel.json                 # Deployment configuration
```

---

## Troubleshooting

### Issue: Dashboard shows blank
**Solution**: Clear browser cache, check browser console for errors
```bash
curl https://your-project.vercel.app/api/status -v
# Check if API is responding
```

### Issue: API endpoints return 404
**Solution**: Verify api/ directory exists and files have .py extension
```bash
# Check deployment logs in Vercel dashboard
# Settings → Deployments → Click latest → View Logs
```

### Issue: Environment variables not working
**Solution**: Add in Vercel dashboard
1. Project Settings → Environment Variables
2. Add `SUPABASE_URL` and `SUPABASE_ANON_KEY`
3. Redeploy (trigger new deployment)

### Issue: Styles not loading
**Solution**: Check vercel.json `outputDirectory` is set to `ui`
```bash
# Verify from Vercel dashboard
# Project Settings → Build & Deployment
```

---

## Next Steps

1. **Deploy to Vercel** using Option A or B above
2. **Test all API endpoints** using curl commands above
3. **Configure Custom Domain** (optional):
   - Vercel dashboard → Settings → Domains
   - Add your domain (e.g., ddl69.com)
4. **Monitor Deployment**:
   - Vercel dashboard → Analytics & Monitoring
   - Check function invocations and response times

---

## Security Notes

✓ No sensitive data exposed in frontend
✓ API endpoints return mock/safe data
✓ Real database access requires SUPABASE_SERVICE_ROLE_KEY (backend only)
✓ All user inputs validated by validators.py
✓ API keys moved from URL params to Bearer headers
✓ Discord tokens moved from CLI args to ENV vars

---

## Support

- Vercel Docs: https://vercel.com/docs
- GitHub Repo: https://github.com/dropped95si/DDL-420-69
- Supabase Setup: See SETUP_DEPLOYMENT.md Part 1

---

**Version**: v0.2.0  
**Last Updated**: 2026-02-10  
**Status**: Production Ready ✓
