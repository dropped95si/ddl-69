import json

with open(r"c:\Users\Stas\Downloads\ddl-69_v0.8\tmp_audit.json") as f:
    d = json.load(f)

s = d.get("summary", {})
print("=== AUDIT SUMMARY ===")
print(f"  Total: {s.get('total_predictions')}")
print(f"  Avg Confidence: {s.get('avg_confidence', 0) * 100:.1f}%")
print(f"  Avg Expected Return: {s.get('avg_expected_return_pct', 0):.2f}%")
print()

for p in d.get("predictions", [])[:5]:
    m = p.get("metrics", {})
    t = p.get("targets", {})
    ticker = p["ticker"]
    exp = m["expected_return_pct"]
    conf = m["confidence"] * 100
    tp1 = t.get("tp1", "?")
    sl1 = t.get("sl1", "?")
    rr = m.get("risk_reward_ratio", "N/A")
    print(f"  {ticker:>6} | exp_ret={exp:+.2f}% | conf={conf:.1f}% | tp1=${tp1} | sl=${sl1} | rr={rr}")
