# Runs the live pipeline. Requires a configured .env (Apify + Tavily/Exa).
$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)
if (Test-Path ".venv\Scripts\Activate.ps1") { . ".venv\Scripts\Activate.ps1" }
python app.py check
if ($LASTEXITCODE -ne 0) { Write-Error "Configuration check failed -- see the messages above."; exit $LASTEXITCODE }
python app.py run-all
