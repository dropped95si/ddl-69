# Deployment Guide - DDL-420-69 to Vercel

## Quick Start (5 minutes)

### 1. Prerequisites
- GitHub account with access to [dropped95si/DDL-420-69](https://github.com/dropped95si/DDL-420-69)
- Vercel account (free at vercel.com)

### 2. Connect to Vercel

Option A: Via Vercel Dashboard (Recommended)
```bash
1. Go to https://vercel.com/new
2. Click "Import Project"
3. Paste repo URL: https://github.com/dropped95si/DDL-420-69
4. Click "Import"
5. Accept default settings
6. Click "Deploy"
```

Option B: Via CLI
```bash
npm i -g vercel
cd DDL-420-69
vercel --prod
```

### 3. Configure Environment Variables

In Vercel Dashboard:
1. Go to Project Settings → Environment Variables
2. Add these variables:
   ```
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_ANON_KEY=your_anon_key_here
   SUPABASE_SERVICE_ROLE_KEY=your_service_key_here
   ```

### 4. Deploy

Push to GitHub `main` branch triggers automatic deployment:
```bash
git push origin main
```

## Post-Deployment

### Verify Deployment
```bash
curl https://your-vercel-url.vercel.app/
# Should show the DDL-69 dashboard
```

### Run Supabase SQL Scripts
```bash
# Via psql or Supabase SQL Editor
psql -h your_supabase_host -U postgres -d postgres < sql/ledger_v1.sql
psql -h your_supabase_host -U postgres -d postgres < sql/ledger_v2_patch.sql
psql -h your_supabase_host -U postgres -d postgres < sql/ingest_v1.sql
```

### Test CLI Commands
```bash
python -m ddl69.cli.main help
python -m ddl69.cli.main tools_status
```

## Custom Domain

In Vercel Dashboard:
1. Go to Settings → Domains
2. Click "Add Domain"
3. Enter your domain (e.g., ddl69.com)
4. Follow DNS configuration instructions

## Monitoring

Monitor deployment health:
```bash
vercel logs --follow
```

View analytics:
- Vercel Dashboard → Analytics tab

## Troubleshooting

### Build fails with "npm not found"
- Make sure vercel.json buildCommand is valid
- Our config uses: `"npm run build 2>/dev/null || echo 'No build needed'"`

### Environment variables not working
- Verify variables added to all environments (Production, Preview, Development)
- Re-deploy after adding variables
- Check logs: `vercel logs`

### UI not loading
- Check browser console for CORS errors
- Verify SUPABASE_URL and SUPABASE_ANON_KEY are correct
- Test direct SQL connection

## Rollback

To rollback to previous deployment:
1. Vercel Dashboard → Deployments
2. Click on previous deployment
3. Click "Promote to Production"

## API Deployment (Backend)

For backend API on Vercel:
1. Create `api/` directory with Python functions
2. Use Vercel's Python runtime
3. Deploy serverless functions

Example:
```python
# api/signals.py
def handler(request):
    return {"status": "ok"}
```

## Cost Estimate

Free tier includes:
- ✓ Static UI deployment (unlimited)
- ✓ 50 GB/month bandwidth
- ✓ Automatic HTTPS
- ✓ Custom domains

Paid features:
- Serverless functions (pay-per-invocation)
- More bandwidth
- Priority support

## Security Checklist

- [ ] Environment variables set and not exposed
- [ ] HTTPS enabled (automatic with Vercel)
- [ ] API keys never in code or public repos
- [ ] CORS properly configured
- [ ] Rate limiting enabled
- [ ] Authentication properly configured
- [ ]  Regular security scans enabled

## Next Steps

1. ✓ Repository: https://github.com/dropped95si/DDL-420-69
2. [ ] Connect to Vercel
3. [ ] Set environment variables
4. [ ] Deploy database schema
5. [ ] Test API endpoints
6. [ ] Add custom domain

For more help:
- Vercel Docs: https://vercel.com/docs
- GitHub: https://github.com/dropped95si/DDL-420-69
- Issues: https://github.com/dropped95si/DDL-420-69/issues

---
Last Updated: 2026-02-10
Version: v0.2.0 (Production Ready)
