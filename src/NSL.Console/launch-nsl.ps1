# NSL.exe Launcher with Runtime Config Fix
# This script creates the missing runtime config and launches nsl.exe

$binPath = "$PSScriptRoot\bin\Debug\net8.0"
$runtimeConfigPath = "$binPath\nsl.runtimeconfig.json"
$exePath = "$binPath\nsl.exe"

# Create the missing runtime configuration file
$runtimeConfig = @{
    runtimeOptions = @{
        tfm = "net8.0"
        framework = @{
            name = "Microsoft.NETCore.App"
            version = "8.0.0"
        }
        configProperties = @{
            "System.Reflection.Metadata.MetadataUpdater.IsSupported" = $false
        }
    }
} | ConvertTo-Json -Depth 3

# Ensure the bin directory exists
if (!(Test-Path $binPath)) {
    Write-Host "Building NSL Console first..." -ForegroundColor Yellow
    Set-Location $PSScriptRoot
    dotnet build
}

# Create the runtime config file
Write-Host "Creating runtime configuration..." -ForegroundColor Green
$runtimeConfig | Out-File -FilePath $runtimeConfigPath -Encoding UTF8

# Launch nsl.exe
if (Test-Path $exePath) {
    Write-Host "Launching NSL Console..." -ForegroundColor Green
    & $exePath
} else {
    Write-Host "nsl.exe not found. Building first..." -ForegroundColor Yellow
    Set-Location $PSScriptRoot
    dotnet build
    if (Test-Path $exePath) {
        & $exePath
    } else {
        Write-Host "Failed to build nsl.exe" -ForegroundColor Red
    }
}
