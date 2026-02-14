# Custom Domain Setup for DDL-69 Dashboard

## Option 1: Setup under stazmediacorp.com

### Add DNS Records:
```
Type: CNAME
Name: ddl69 (or trading, or dashboard)
Value: cname.vercel-dns.com
TTL: 300
```

### Add to Vercel:
```bash
vercel domains add ddl69.stazmediacorp.com
vercel domains add trading.stazmediacorp.com
```

**Result:** 
- https://ddl69.stazmediacorp.com
- https://trading.stazmediacorp.com

## Option 2: Setup under agilera.ai

### Add DNS Records:
```
Type: CNAME  
Name: ddl69 (or trading, or dashboard)
Value: cname.vercel-dns.com
TTL: 300
```

### Add to Vercel:
```bash
vercel domains add ddl69.agilera.ai
vercel domains add trading.agilera.ai
```

**Result:**
- https://ddl69.agilera.ai
- https://trading.agilera.ai

## Step-by-Step Guide:

1. **Choose Your Domain:**
   - stazmediacorp.com (corporate parent)
   - agilera.ai (AI/trading focused)

2. **Add DNS Record:**
   - Log into your DNS provider (Cloudflare/GoDaddy/etc)
   - Add CNAME record pointing subdomain to `cname.vercel-dns.com`

3. **Configure Vercel:**
   ```bash
   cd c:\Users\Stas\Downloads\ddl-69_v0.8
   vercel domains add [YOUR-SUBDOMAIN].[YOUR-DOMAIN]
   ```

4. **Wait for Propagation:**
   - Usually takes 5-30 minutes
   - Check status: `vercel domains ls`

5. **Set as Default:**
   - Open Vercel dashboard
   - Project settings → Domains
   - Set custom domain as production domain

## Current Status:
- ✅ Production deployed: https://ddl-69-[hash].vercel.app
- ⏳ Custom domain: Waiting for configuration
- ✅ Supabase credentials: Configured in Vercel env vars

## Dashboard Features Now Live:
- ✅ Sortable table (click any column header)
- ✅ Charts (confidence distribution + timeframe breakdown)
- ✅ Confidence badges (color-coded: green=80%+, yellow=70-80%, gray=<70%)
- ✅ Links to stazmediacorp.com and agilera.ai in navbar
- ✅ Sidebar filters (All/Day/Swing/Long)
- ✅ Real-time updates (60-second auto-refresh)
- ✅ Walk-forward analysis links
- ✅ Mobile responsive

## Which domain do you want to use?
Tell me and I'll help configure it.
