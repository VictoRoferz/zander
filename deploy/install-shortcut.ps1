# Creates a "Zander" shortcut on the Desktop that launches start-zander.bat.
# Run once (right-click -> Run with PowerShell, or:  powershell -ExecutionPolicy Bypass -File install-shortcut.ps1)
$ErrorActionPreference = "Stop"

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$target  = Join-Path $here "start-zander.bat"
$icon    = Join-Path $here "zander.ico"
$desktop = [Environment]::GetFolderPath("Desktop")
$lnkPath = Join-Path $desktop "Zander.lnk"

$shell    = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($lnkPath)
$shortcut.TargetPath       = $target
$shortcut.WorkingDirectory = $here
$shortcut.Description       = "Start the Zander inspection station (Docker hub + native camera + USB button)"
if (Test-Path $icon) {
    $shortcut.IconLocation = $icon
} else {
    Write-Host "Note: deploy\zander.ico not found - using the default icon. Drop a .ico there and re-run to set a custom one."
}
$shortcut.Save()

Write-Host "Created desktop shortcut: $lnkPath"
Write-Host "Double-click 'Zander' on your Desktop to start everything."
