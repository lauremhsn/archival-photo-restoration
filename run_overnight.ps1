#requires -Version 5.1
<#
everything that needs gpu time tonight, in order of how much we'd miss it.
quick + important stuff first, so if it dies at 3am we still got the good bits.

run from the repo root with the venv active:
  .\run_overnight.ps1
#>

param(
    [switch]$SkipFullSet,
    [string]$LogDir = "logs"
)

$ErrorActionPreference = "Continue"

if (-not (Test-Path "sd_predict.py")) {
    throw "run this from the repo root, cant see sd_predict.py"
}
if (-not $env:VIRTUAL_ENV) {
    Write-Warning "venv doesnt look active. ctrl+c and run .venv\Scripts\Activate.ps1 first"
    Start-Sleep -Seconds 5
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$log = Join-Path $LogDir "overnight_$stamp.log"

# dont let windows nap halfway through
powercfg /change standby-timeout-ac 0 2>&1 | Out-Null
powercfg /change hibernate-timeout-ac 0 2>&1 | Out-Null
powercfg /change monitor-timeout-ac 20 2>&1 | Out-Null

function Step($name, $cmd) {
    $line = "=" * 70
    $msg = "`n$line`n$name`nstarted $(Get-Date -Format 'HH:mm:ss')`n$line"
    Write-Host $msg -ForegroundColor Cyan
    Add-Content -Path $log -Value $msg

    $t0 = Get-Date
    & powershell -NoProfile -Command "$cmd" 2>&1 | Tee-Object -FilePath $log -Append
    $mins = ((Get-Date) - $t0).TotalMinutes

    if ($LASTEXITCODE -ne 0) {
        $m = "!! $name failed after $([math]::Round($mins,1)) min, carrying on"
        Write-Warning $m
        Add-Content -Path $log -Value $m
    } else {
        $m = "ok, $([math]::Round($mins,1)) min"
        Write-Host $m -ForegroundColor Green
        Add-Content -Path $log -Value $m
    }
}

Write-Host "logging to $log" -ForegroundColor DarkGray
$begin = Get-Date

# --- test split predictions, 57 images each. these unlock the model comparison
#     and close #15, so they go first.
foreach ($mode in @("real", "none", "shuffled", "generic")) {
    Step "sd predictions, captions=$mode" "python sd_predict.py --captions $mode --split test"
    Step "scoring sd768_$mode" "python -m condcolor.protocol outputs/sd768_$mode --split test"
}

# --- the big one: finish the 563-image zero-shot run. slow, but its the
#     well-powered answer to 'does the title help'.
if (-not $SkipFullSet) {
    Step "full 563 set, both prompt modes" "python sd_tier_a.py"
    Step "title vs generic on the full set" "python sd_compare.py"
}

# --- if rolas unet checkpoint showed up, score it too so the morning table is complete
if (Test-Path "checkpoints/condunet/best.pt") {
    foreach ($mode in @("real", "none", "shuffled")) {
        Step "condunet predictions, captions=$mode" `
             "python -m condcolor.predict checkpoints/condunet/best.pt --split test --captions $mode"
        Step "scoring condunet_$mode" "python -m condcolor.protocol outputs/condunet_$mode --split test"
    }
} else {
    $m = "`nno condunet checkpoint yet, skipped rolas model"
    Write-Host $m -ForegroundColor DarkGray
    Add-Content -Path $log -Value $m
}

$total = ((Get-Date) - $begin).TotalHours
$end = "`nall done in $([math]::Round($total,1)) hours. log: $log"
Write-Host $end -ForegroundColor Green
Add-Content -Path $log -Value $end

Write-Host "`nwhat landed:" -ForegroundColor Cyan
Get-ChildItem results\*.csv -ErrorAction SilentlyContinue | ForEach-Object {
    "  {0}  {1} rows" -f $_.Name, ((Import-Csv $_).Count)
}
Get-ChildItem results_tier_a_sd768_*.csv -ErrorAction SilentlyContinue | ForEach-Object {
    "  {0}  {1} rows" -f $_.Name, ((Import-Csv $_).Count)
}
