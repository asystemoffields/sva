param(
    [string]$App = "sva-trainable-recall-h100",
    [int]$Tail = 200,
    [switch]$Follow
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NO_COLOR = "1"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()

$Args = @("app", "logs", $App, "--tail", "$Tail", "--timestamps")
if ($Follow) {
    $Args += "--follow"
}

& modal @Args
