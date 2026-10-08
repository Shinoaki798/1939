# Read CPU / GPU / disk temperatures, loads, fans and NVMe wear on the 5080 through LibreHardwareMonitorLib
# (needs the PawnIO driver for the CPU sensors and an elevated session, which the SSH session is).
#   ssh gpu 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\AN\Tools\hw_temps.ps1'
# One line per sensor: hardware | type | sensor | value.
param([string]$Lib = "C:\Users\AN\Tools\LibreHardwareMonitor\LibreHardwareMonitorLib.dll")
Add-Type -Path $Lib
$c = New-Object LibreHardwareMonitor.Hardware.Computer
$c.IsCpuEnabled = $true
$c.IsGpuEnabled = $true
$c.IsStorageEnabled = $true
$c.IsMotherboardEnabled = $true
$c.IsMemoryEnabled = $true
$c.Open()

function Show-Hardware($h) {
    $h.Update()
    foreach ($sub in $h.SubHardware) { Show-Hardware $sub }
    foreach ($s in $h.Sensors) {
        $t = "$($s.SensorType)"
        $keep = switch ($t) {
            "Temperature" { $true }
            "Fan"         { $s.Value -gt 0 }
            "Load"        { $s.Name -match "^(CPU Total|GPU Core|GPU Memory|Memory|Used Space)$" }
            "Power"       { $s.Name -match "Package" }
            "Level"       { $true }    # NVMe: percentage used, available spare
            "Data"        { $s.Name -match "^(Data Read|Data Written|Memory Used|Memory Available)$" }
            default       { $false }
        }
        if ($s.Name -match "Limit|Resolution|Threshold|^(Warning|Critical) Temperature$") { $keep = $false }
        if ($keep -and $s.Value -ne $null) {
            "{0} | {1} | {2} | {3:N1}" -f $h.Name, $t, $s.Name, $s.Value
        }
    }
}

foreach ($h in $c.Hardware) { Show-Hardware $h }
$c.Close()
