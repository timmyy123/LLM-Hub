$ErrorActionPreference = 'Stop'
$defPath = Join-Path $PSScriptRoot '..\app\src\main\res\values\strings.xml'
if (-not (Test-Path $defPath)) { Write-Error "Default strings.xml not found at $defPath"; exit 2 }
$defXml = [xml](Get-Content $defPath -Raw)
$defNames = @()
if ($defXml.resources -and $defXml.resources.string) {
    foreach ($s in $defXml.resources.string) { if ($s.name) { $defNames += $s.name } }
}
$locFiles = Get-ChildItem -Path (Join-Path $PSScriptRoot '..\app\src\main\res') -Recurse -Filter strings.xml | Where-Object { $_.DirectoryName -match '\\values-' }
$all = @()
foreach ($f in $locFiles) {
    try {
        $x = [xml](Get-Content $f.FullName -Raw)
        if ($x.resources -and $x.resources.string) {
            foreach ($s in $x.resources.string) { if ($s.name) { $all += $s.name } }
        }
    } catch {
        Write-Warning "Failed to parse $($f.FullName): $($_.Exception.Message)"
    }
}
$missing = ($all | Sort-Object -Unique) | Where-Object { $_ -and ($defNames -notcontains $_) }
if ($missing.Count -gt 0) {
    Write-Output "Missing keys in default values/strings.xml:`n"
    $missing | ForEach-Object { Write-Output " - $_" }
    exit 1
} else {
    Write-Output "No missing keys found"
    exit 0
}
