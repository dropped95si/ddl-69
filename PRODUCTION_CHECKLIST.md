# DDL-69 Production Deployment Checklist

**Project**: DDL-69 Probability Engine v0.2.0
**Target**: Vercel Production Deployment
**Status**: ✓ READY FOR DEPLOYMENT

---

## Pre-Deployment (This Conversation)

- [x] Security audit completed (8 vulnerabilities fixed)
- [x] Unit tests passing (7/7)
- [x] UI dashboard created and tested
- [x] API endpoints developed and tested
- [x] Git commits pushed to GitHub
- [x] Environment variables error fixed
- [x] All documentation created
- [x] Performance baseline established

---

## Deployment Steps

### Step 1: Deploy to Vercel
Choose ONE option:

**Option A: Web Dashboard (Recommended)**
- [ ] Go to https://vercel.com
- [ ] Click "Dashboard" → "Add New Project"
- [ ] Click "Import from Git"
- [ ] Paste: https://github.com/dropped95si/DDL-420-69
- [ ] Click "Import"
- [ ] Framework: Select "Other"
- [ ] Click "Deploy"
- [ ] Wait 3-5 minutes
- [ ] When done, click "Visit"

**Option B: Vercel CLI**
- [ ] Run: `npm install -g vercel`
- [ ] Run: `vercel --prod`
- [ ] Follow prompts (defaults are fine)

**Option C: GitHub Auto-Deploy** (Already set up)
- [ ] Every push to main automatically deploys
- [ ] Just push code to GitHub and wait

---

## Post-Deployment Verification

### Immediately After Deploy
- [ ] Note your Vercel URL (e.g., https://ddl-69-abc123.vercel.app/)
- [ ] Test dashboard loads: `curl https://your-url/`
- [ ] Test API: `curl https://your-url/api/status`
- [ ] Open dashboard in browser (check console for errors)
- [ ] Charts render correctly
- [ ] Events list updates

### API Endpoints - Test All 5
```bash
# 1. Health check
curl https://your-url/api/health
# Expected: {"status": "healthy", ...}

# 2. Status endpoint
curl https://your-url/api/status
# Expected: {"system_status": "ONLINE", ...}

# 3. Forecasts endpoint
curl https://your-url/api/forecasts
# Expected: 30-day forecast data

# 4. Calibration endpoint
curl https://your-url/api/calibration
# Expected: Calibration curve data

# 5. Events endpoint
curl https://your-url/api/events
# Expected: Recent events list
```

### Browser Testing
- [ ] Open https://your-url/ in Chrome
- [ ] Open in Firefox
- [ ] Open on mobile device
- [ ] Check responsive layout
- [ ] Open DevTools Console (should be clean, no errors)
- [ ] Check Network tab (all assets load)
- [ ] Click on chart (should be interactive)

---

## Optional: Add Custom Domain (10 minutes)

If you have a domain (e.g., ddl69.com):

1. **In Vercel Dashboard**:
   - Select project → Settings → Domains
   - Click "Add Domain"
   - Enter your domain
   - Follow DNS instructions

2. **In Your Domain Provider** (GoDaddy, Namecheap, etc):
   - Add CNAME record pointing to Vercel
   - Update nameservers if needed

3. **Enable SSL**:
   - Vercel auto-provisions Let's Encrypt cert
   - HTTPS automatic in 48 hours

---

## Optional: Add Supabase Database (30 minutes)

For real database integration:

1. **Create Supabase Project**:
   - Go to https://app.supabase.com
   - Click "New Project"
   - Name: "DDL-69"
   - Region: Choose near you
   - Wait for creation (~2 min)

2. **Run SQL Schemas**:
   - Copy contents of:
     - sql/ledger_v1.sql
     - sql/ledger_v2_patch.sql
     - sql/ingest_v1.sql
   - Go to Supabase → SQL Editor
   - Paste each and execute

3. **Get API Keys**:
   - Settings → API
   - Copy "URL" and "anon key"

4. **Add to Vercel**:
   - Vercel Dashboard → Settings → Environment Variables
   - Add:
     ```
     SUPABASE_URL=https://your-project.supabase.co
     SUPABASE_ANON_KEY=your_key_here
     ```
   - Click save
   - Redeploy (Vercel will auto-redeploy)

---

## Monitoring After Deploy

### Daily (First Week)
- [ ] Check Vercel Analytics dashboard
- [ ] Verify no errors in logs
- [ ] Test /api/health endpoint
- [ ] Monitor response times

### Weekly
- [ ] Review function invocations
- [ ] Check error rates (should be 0%)
- [ ] Verify database connection (if using)

### Monthly
- [ ] Security audit
- [ ] Performance review
- [ ] Backup verification
- [ ] Update dependencies

---

## Rollback Plan

If something breaks:

**Option 1: Quick Rollback**
1. Vercel Dashboard → Deployments
2. Find previous working deployment
3. Click "Promote to Production"
4. Takes 30 seconds

**Option 2: Revert Git & Redeploy**
1. `git revert <commit-hash>`
2. `git push origin main`
3. Vercel auto-redeploys (2-3 min)

**Option 3: Manual Redeploy**
1. `vercel --prod`
2. Deploys from current code

---

## Success Criteria

✓ Dashboard loads without errors
✓ All 5 API endpoints respond
✓ Charts render correctly
✓ Auto-refresh works (check Network tab)
✓ Mobile layout looks good
✓ No console errors
✓ Response times < 1000ms
✓ Zero API errors

---

## Troubleshooting

### Problem: Blank dashboard
**Solution**: 
1. Check browser console (F12 → Console tab)
2. Test API: `curl https://your-url/api/status`
3. If API 404: Functions not deployed, redeploy

### Problem: Charts don't load
**Solution**:
1. Verify Chart.js CDN accessible
2. Check JSON response from /api/forecasts
3. Check browser Network tab

### Problem: Slow response
**Solution**:
1. Check Vercel Analytics
2. Verify cache headers working
3. Upgrade Vercel plan if needed

### Problem: 500 errors
**Solution**:
1. Check Vercel logs
2. Verify Python code in api/ directory
3. Check for missing dependencies

---

## Important URLs

| Item | URL |
|------|-----|
| Vercel Dashboard | https://vercel.com/dashboard |
| GitHub Repo | https://github.com/dropped95si/DDL-420-69 |
| Supabase Console | https://app.supabase.com |
| Your Deployed App | https://your-project.vercel.app |
| Vercel Docs | https://vercel.com/docs |

---

## Success Notification

Once everything is working:

```
✓ Deployment complete
✓ Dashboard live at https://your-url/
✓ All API endpoints working
✓ Ready for real data integration
✓ Auto-monitoring active
```

---

## Next Steps After Live

1. **Set up monitoring** (see MONITORING_OPTIMIZATION.md)
2. **Add Supabase** (optional but recommended)
3. **Custom domain** (optional)
4. **Error tracking** with Sentry (optional)
5. **Analytics** with LogRocket (optional)

---

**Estimated Time to Live**: 
- Option A (Dashboard): 5 minutes
- Option B (CLI): 10 minutes
- Verification: 10 minutes
- **Total**: 15-20 minutes until live

**Estimated Time with Supabase**: Add 30 minutes

---

**Status**: Ready to deploy
**Date**: 2026-02-10
**Version**: v0.2.0
