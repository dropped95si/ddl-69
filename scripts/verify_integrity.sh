#!/bin/bash
# scripts/verify_integrity.sh
# CI Tripwire to prevent Truth Mode regression and Secret Leakage

echo "Running Integrity Checks..."

# 1. Truth Mode Copy Tripwire
# Fail if prohibited claims appear
echo "Checking for prohibited copy..."
if grep -RInE "Live metrics|updated every\s*5|production ML pipeline|Supabase ML pipeline" ui/; then
    echo "FAIL: Prohibited 'Live' copy found."
    exit 1
fi

# 2. Secret Tripwire
# Fail if .env* or JWT patterns appear in tracked files
echo "Checking for tracked secrets..."
if git ls-files | grep -E "\.env(\.|$)|\.vercel/"; then
    echo "FAIL: Tracked secret files found."
    exit 1
fi

if grep -RInE "eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}" .; then
    echo "FAIL: Potential JWT tokens found."
    exit 1
fi

echo "PASS: Integrity Checks Passed."
exit 0
