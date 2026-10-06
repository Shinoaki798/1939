# Start a long-running remote job as a process that outlives the SSH session.
#   ssh gpu 'powershell -NoProfile -Command -' < scripts/start_download.ps1
# To pass downloader arguments, prepend a $DlArgs line:
#   (echo '$DlArgs = "--source hmd_newspapers"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# To run another script from scripts/ instead of run_download.sh, prepend a $Script line:
#   (echo '$Script = "ca_resume.sh"'; cat scripts/start_download.ps1) | ssh gpu 'powershell -NoProfile -Command -'
# Win32_Process.Create runs wsl.exe outside the SSH session's job object.
if (-not $DlArgs) { $DlArgs = '' }
if (-not $Script) { $Script = 'run_download.sh' }
$cmd = "C:\Windows\System32\wsl.exe -d Ubuntu -- bash /home/an/1939/scripts/$Script $DlArgs"
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd }
"ReturnValue=$($r.ReturnValue) ProcessId=$($r.ProcessId) cmd=$cmd"
