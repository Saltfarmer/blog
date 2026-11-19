# Run this site locally and open it in the browser.
# Works in: blog, micro-salt (Jekyll) and saltfarmer.github.io (plain HTML).
# Usage:  double-click _serve.bat   or   .\_serve.ps1   or   .\_serve.ps1 -Port 4001
# Stop:   Ctrl+C
param([int]$Port = 0)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Test-PortFree([int]$p) {
    $l = $null
    try { $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p); $l.Start(); return $true }
    catch { return $false }
    finally { if ($l) { $l.Stop() } }
}

# Pick a free port: 4000, 4001, 4002...
if ($Port -eq 0) {
    $Port = 4000
    while (-not (Test-PortFree $Port)) { $Port++ }
}

$isJekyll = Test-Path '_config.yml'
$baseurl = ''
if ($isJekyll) {
    $line = Select-String -Path '_config.yml' -Pattern '^\s*baseurl\s*:\s*"?([^"#\s]*)' | Select-Object -First 1
    if ($line) { $baseurl = $line.Matches[0].Groups[1].Value.TrimEnd('/') }
}
$url = "http://127.0.0.1:$Port$baseurl/"

# Open the browser as soon as the site answers (first Jekyll build can take a minute)
$opener = Start-Job -ArgumentList $url -ScriptBlock {
    param($u)
    for ($i = 0; $i -lt 300; $i++) {
        try { Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $u; return }
        catch { Start-Sleep -Seconds 1 }
    }
}

Write-Host ""
Write-Host "  Site:  $url" -ForegroundColor Cyan
Write-Host "  Stop:  Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

try {
    if ($isJekyll) {
        if ((Test-Path 'Gemfile') -and -not (Select-String -Path 'Gemfile' -Pattern 'tzinfo-data' -Quiet)) {
            Write-Host "  Tip: add  gem `"tzinfo-data`", platforms: [:mingw, :x64_mingw, :mswin]  to Gemfile if you see a timezone error." -ForegroundColor Yellow
        }
        bundle check *> $null
        if ($LASTEXITCODE -ne 0) { Write-Host "  Installing gems (first run only)..." -ForegroundColor Yellow; bundle install }
        # Live reload: the page refreshes by itself when you save a file
        $lrPort = 35729 + ($Port - 4000)
        bundle exec jekyll serve --port $Port --livereload --livereload-port $lrPort
    }
    else {
        # Plain HTML site (the gate): a simple static server
        python -m http.server $Port --bind 127.0.0.1
    }
}
finally {
    Stop-Job $opener -ErrorAction SilentlyContinue
    Remove-Job $opener -Force -ErrorAction SilentlyContinue
}
