# Fetch the AMI Meeting Corpus manual annotations into data\ami\.
# ~22 MB. Not redistributed with this repository — see README.
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Dest = Join-Path $Root "data"
$Url  = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
$Zip  = Join-Path $Dest "ami_public_manual_1.6.2.zip"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

if (Test-Path (Join-Path $Dest "ami\words")) {
    Write-Host "already present: $Dest\ami"
    exit 0
}

Write-Host "downloading AMI manual annotations (~22 MB)..."
# curl.exe, not the PowerShell alias — Invoke-WebRequest is slow on large files
curl.exe -L --fail -o $Zip $Url

Write-Host "unpacking..."
Expand-Archive -Path $Zip -DestinationPath (Join-Path $Dest "ami") -Force
Remove-Item $Zip

$n = (Get-ChildItem (Join-Path $Dest "ami\words") -Filter *.xml).Count
Write-Host ""
Write-Host "done. $n word files in $Dest\ami"
Write-Host "run:  python src\run.py"
