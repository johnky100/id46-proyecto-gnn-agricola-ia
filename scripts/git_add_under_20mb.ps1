$limite = 20MB # Límite máximo permitido por archivo

$archivos = Get-ChildItem -Path . -Recurse -File -Force |
    Where-Object {
        $_.FullName -notmatch '\\\.git\\' -and
        $_.Length -lt $limite
    }

foreach ($archivo in $archivos) {
    git add -- "$($archivo.FullName)" # Agregar únicamente archivos menores de 20 MB
}

Write-Host ""
Write-Host "Archivos menores de 20 MB agregados al staging."
Write-Host ""

$grandes = Get-ChildItem -Path . -Recurse -File -Force |
    Where-Object {
        $_.FullName -notmatch '\\\.git\\' -and
        $_.Length -ge $limite
    }

if ($grandes.Count -gt 0) {
    Write-Host "Archivos excluidos por superar o alcanzar los 20 MB:"
    
    foreach ($archivo in $grandes) {
        $mb = [math]::Round($archivo.Length / 1MB, 2)
        Write-Host "$mb MB`t$($archivo.FullName)"
    }
}