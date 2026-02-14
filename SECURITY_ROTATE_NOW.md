# SECURITY INCIDENT RESPONSE: ROTATE SECRETS IMMEDIATELY

**DATE:** 2026-02-14
**PRIORITY:** CRITICAL
**STATUS:** PENDING ACTION

---

## 🛑 STOP! READ THIS FIRST

This repository previously contained `.env` files with active API keys and secrets in its history.
Even though these files have been removed from the current working tree, **the secrets they contained are COMPROMISED.**

You must perform the following rotations IMMEDIATELY:

### 1. Supabase Service Role Key
*   **Status:** LEAKED
*   **Action:** Go to Supabase Dashboard -> Project Settings -> API.
*   **Task:** Generate a new `service_role` secret.
*   **Revoke:** The old key is now public. Ensure it is invalidated.

### 2. Alpaca API Keys (Live & Paper)
*   **Status:** LEAKED (Potentially)
*   **Action:** Go to Alpaca Dashboard.
*   **Task:** Regenerate API Key ID and Secret Key for both Live and Paper trading.
*   **Update:** Update your environment variables in Vercel/Local.

### 3. Vercel Tokens
*   **Status:** LEAKED (`.vercel/` directory was present)
*   **Action:** Go to Vercel Dashboard -> Settings -> Tokens.
*   **Task:** Revoke any personal access tokens or project tokens created before today.

### 4. Git History Scrub (Recommended)
*   If this repo is public, simply deleting the files is NOT enough.
*   Use `git filter-repo` or `BFG Repo-Cleaner` to rewrite history and remove `.env` artifacts entirely.
*   *Alternatively:* Delete the public repo and re-push this clean version to a FRESH repository.

---

## 🛡️ Verification Checklist

- [ ] All old keys revoked.
- [ ] New keys generated.
- [ ] New keys stored ONLY in Vercel Environment Variables.
- [ ] Local development uses `.env` (gitignored) populated from `.env.example`.
- [ ] No secrets in `git log -p`.

**DO NOT COMMIT REAL SECRETS AGAIN.**
