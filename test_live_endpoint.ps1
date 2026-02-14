try {
    $r = Invoke-WebRequest 'https://ddl-69-3xc7uf2ws-stas-projects-794d183b.vercel.app/api/live' -TimeoutSec 15
    $data = $r.Content | ConvertFrom-Json
    Write-Host "HTTP Status: $($r.StatusCode)"
    Write-Host "Error: $($data.error)"
    Write-Host "Message: $($data.message)"
    Write-Host "Count: $($data.count)"
    Write-Host "Source: $($data.source)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
}
