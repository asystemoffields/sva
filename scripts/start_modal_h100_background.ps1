param(
    [string]$Name = "sva-h100"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$RunRoot = Join-Path $RepoRoot "results\modal_runs"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunName = "$Name-$Timestamp"
$Stdout = Join-Path $RunRoot "$RunName.stdout.log"
$Stderr = Join-Path $RunRoot "$RunName.stderr.log"
$Result = Join-Path $RunRoot "$RunName.result.txt"
$Meta = Join-Path $RunRoot "$RunName.meta.txt"

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NO_COLOR = "1"

$Args = @(
    "run",
    "--detach",
    "--quiet",
    "--timestamps",
    "--write-result",
    $Result,
    "modal_h100_trainable.py"
)

$Process = Start-Process `
    -FilePath "modal" `
    -ArgumentList $Args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -WindowStyle Hidden `
    -PassThru

@(
    "name=$RunName"
    "pid=$($Process.Id)"
    "started_at=$(Get-Date -Format o)"
    "repo=$RepoRoot"
    "stdout=$Stdout"
    "stderr=$Stderr"
    "result=$Result"
    "command=modal $($Args -join ' ')"
) | Set-Content -Path $Meta -Encoding utf8

Write-Output "Started $RunName"
Write-Output "PID: $($Process.Id)"
Write-Output "stdout: $Stdout"
Write-Output "stderr: $Stderr"
Write-Output "result: $Result"
Write-Output "meta: $Meta"
