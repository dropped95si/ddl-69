#!/bin/bash

# DDL-69 Vercel Deployment Script
# Automates the deployment process

set -e

echo "[*] DDL-69 Vercel Deployment Script v0.2.0"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "[!] Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "[1/5] Verifying project structure..."
if [ ! -f "vercel.json" ]; then
    echo "[ERROR] vercel.json not found. Run from project root."
    exit 1
fi

echo "[1.5/5] Running Integrity Checks..."
bash scripts/verify_integrity.sh
if [ $? -eq 0 ]; then
    echo "[OK] Integrity checks passed"
else
    echo "[ERROR] Integrity checks failed"
    exit 1
fi

echo "[2/5] Running tests..."
python -m pytest tests/ -q
if [ $? -eq 0 ]; then
    echo "[OK] All tests passed"
else
    echo "[ERROR] Tests failed"
    exit 1
fi

echo "[3/5] Checking Git status..."
if [ -z "$(git status --porcelain)" ]; then
    echo "[OK] Working directory clean"
else
    echo "[WARNING] Uncommitted changes detected"
    git status
fi

echo "[4/5] Deploying to Vercel..."
echo ""
echo "Choose deployment method:"
echo "1) Production deployment (deploy to main URL)"
echo "2) Preview deployment (test before promotion)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" == "1" ]; then
    echo "Deploying to PRODUCTION..."
    vercel --prod
elif [ "$choice" == "2" ]; then
    echo "Creating PREVIEW deployment..."
    vercel
else
    echo "[ERROR] Invalid choice"
    exit 1
fi

echo ""
echo "[5/5] Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Verify dashboard loads: https://<project>.vercel.app/"
echo "2. Test API endpoints with curl"
echo "3. Review DEPLOY_VERCEL_COMPLETE.md for post-deployment steps"
