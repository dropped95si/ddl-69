import requests
import json

url = "https://ddl-69-okcm4iy0b-stas-projects-794d183b.vercel.app/api/live?timeframe=day"
response = requests.get(url)
data = response.json()

print(f"Source: {data['source']}")
print(f"Count: {data['count']}")
