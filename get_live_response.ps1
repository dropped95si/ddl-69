$url = 'https://ddl-69-3xc7uf2ws-stas-projects-794d183b.vercel.app/api/live'
$r = Invoke-WebRequest -Uri $url -UseBasicParsing
$r.Content | Out-File -FilePath response.txt
Write-Host "Response saved to response.txt"
Write-Host "First 1000 chars:"
$r.Content.Substring(0, [Math]::Min(1000, $r.Content.Length))
