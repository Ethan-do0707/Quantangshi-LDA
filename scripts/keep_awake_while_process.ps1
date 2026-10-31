param(
    [Parameter(Mandatory = $true)]
    [int]$TargetProcessId
)

# 仅在目标训练进程存活期间阻止系统自动休眠，不改变永久电源计划。
Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class ExecutionState
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$esContinuous = [uint32]2147483648
$esSystemRequired = [uint32]0x00000001

try {
    [void][ExecutionState]::SetThreadExecutionState($esContinuous -bor $esSystemRequired)
    while (Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}
finally {
    [void][ExecutionState]::SetThreadExecutionState($esContinuous)
}
