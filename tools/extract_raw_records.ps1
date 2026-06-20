param(
    [Parameter(Mandatory = $true)]
    [string]$Subject,

    [Parameter(Mandatory = $true)]
    [ValidateSet("left", "right")]
    [string]$Side,

    [Parameter(Mandatory = $true)]
    [string]$Date,

    [string]$CalibrationRoot = "E:\Research\project\daily_mocap\dataset\code\SmartWear\data\recording",
    [string]$OutputRawRoot = "E:\Research\project\daily_mocap\dataset\code\MocapDB\data\raw",

    # Use this for already-copied local folders, e.g. data\raw\sensor_raw\yinqi_0408\left.
    [string]$SensorRoot,

    # Use this for a connected phone through Windows MTP, e.g. TLR-AL00.
    [string]$MtpDeviceName,
    # Relative to the phone storage root. The script auto-enters the first storage folder
    # shown under the MTP device, e.g. "内部存储" / "Internal storage".
    [string]$MtpDatePath = "Android\data\com.huawei.research.trusport\files\csv",

    [int]$MinFolderSizeMB = 20,
    [int]$MaxDeltaMinutes = 20,
    [int]$MinCsvCount = 6,

    [switch]$DryRun,
    [switch]$Overwrite,
    [switch]$NoSTag
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-TimeStamp {
    param([string]$Text)
    if ($Text -match "(\d{8})(\d{6})") {
        return [datetime]::ParseExact($Matches[1] + $Matches[2], "yyyyMMddHHmmss", $null)
    }
    if ($Text -match "(\d{8})_(\d{6})") {
        return [datetime]::ParseExact($Matches[1] + $Matches[2], "yyyyMMddHHmmss", $null)
    }
    return $null
}

function Parse-CalibrationFiles {
    param([string]$SubjectCalibrationDir, [string]$Date)

    if (-not (Test-Path -LiteralPath $SubjectCalibrationDir)) {
        throw "Calibration subject directory not found: $SubjectCalibrationDir"
    }

    $files = Get-ChildItem -LiteralPath $SubjectCalibrationDir -File -Filter "*.pt" |
        Where-Object { $_.Name -match "_?$Date`_" -or $_.Name -match "_?$Date" }

    $items = @()
    foreach ($file in $files) {
        if ($file.BaseName -match "^(\d+)(?:_(\d+))?_(\d{8})_(\d{6})$") {
            $seq = [int]$Matches[1]
            $calibPart = if ($Matches[2]) { [int]$Matches[2] } else { $null }
            $time = [datetime]::ParseExact($Matches[3] + $Matches[4], "yyyyMMddHHmmss", $null)
            $items += [pscustomobject]@{
                Seq       = $seq
                Part      = $calibPart
                Time      = $time
                Path      = $file.FullName
                Name      = $file.Name
                Length    = $file.Length
            }
        }
    }

    if ($items.Count -eq 0) {
        throw "No calibration .pt files matching date $Date found under $SubjectCalibrationDir"
    }

    $hasRepeatedTrials = @($items | Where-Object { $null -ne $_.Part }).Count -gt 0

    if ($hasRepeatedTrials) {
        $groups = $items | Sort-Object Seq, Part, Time | ForEach-Object {
            $label = if ($null -ne $_.Part) { "$($_.Seq)_$($_.Part)" } else { "$($_.Seq)" }
            [pscustomobject]@{
                Seq      = $label
                SeqIndex = $_.Seq
                Part     = $_.Part
                Time     = $_.Time
                Files    = @($_)
            }
        }
    } else {
        $groups = $items | Group-Object Seq | ForEach-Object {
            $groupItems = @($_.Group | Sort-Object Time)
            [pscustomobject]@{
                Seq      = "$($_.Name)"
                SeqIndex = [int]$_.Name
                Part     = $null
                Time     = ($groupItems | Sort-Object Time -Descending | Select-Object -First 1).Time
                Files    = $groupItems
            }
        } | Sort-Object SeqIndex
    }

    return @($groups)
}

function Get-LocalFolderSize {
    param([string]$Path)
    $stats = Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue |
        Measure-Object -Property Length -Sum
    $csvCount = @(Get-ChildItem -LiteralPath $Path -Recurse -File -Filter "*.csv" -ErrorAction SilentlyContinue).Count
    return [pscustomobject]@{
        Bytes    = [int64]($stats.Sum)
        CsvCount = [int]$csvCount
    }
}

function Get-LocalSensorCandidates {
    param([string]$Root)
    if (-not (Test-Path -LiteralPath $Root)) {
        throw "SensorRoot not found: $Root"
    }

    $candidates = @()
    foreach ($dir in Get-ChildItem -LiteralPath $Root -Directory) {
        $time = Parse-TimeStamp $dir.Name
        if ($null -eq $time) { continue }
        $size = Get-LocalFolderSize $dir.FullName
        $candidates += [pscustomobject]@{
            Name     = $dir.Name
            Time     = $time
            Bytes    = $size.Bytes
            CsvCount = $size.CsvCount
            Kind     = "local"
            Path     = $dir.FullName
            Item     = $null
        }
    }
    return @($candidates | Sort-Object Time)
}

function Get-MtpChildByName {
    param($Folder, [string]$Name)
    foreach ($item in $Folder.Items()) {
        if ($item.Name -eq $Name) { return $item }
    }
    return $null
}

function Resolve-MtpFolder {
    param([string]$DeviceName, [string[]]$Segments)
    $shell = New-Object -ComObject Shell.Application
    $computer = $shell.NameSpace(17)
    if ($null -eq $computer) { throw "Cannot open Shell My Computer namespace." }

    $deviceItem = Get-MtpChildByName $computer $DeviceName
    if ($null -eq $deviceItem) {
        $names = @()
        foreach ($item in $computer.Items()) { $names += $item.Name }
        throw "MTP device '$DeviceName' not found. Visible devices: $($names -join ', ')"
    }

    $folder = $deviceItem.GetFolder
    foreach ($segment in $Segments) {
        $next = Get-MtpChildByName $folder $segment
        if ($null -eq $next) {
            throw "MTP path segment '$segment' not found under '$DeviceName'."
        }
        $folder = $next.GetFolder
    }
    return $folder
}

function Resolve-MtpFolderAutoStorage {
    param([string]$DeviceName, [string[]]$Segments)
    $shell = New-Object -ComObject Shell.Application
    $computer = $shell.NameSpace(17)
    if ($null -eq $computer) { throw "Cannot open Shell My Computer namespace." }

    $deviceItem = Get-MtpChildByName $computer $DeviceName
    if ($null -eq $deviceItem) {
        $names = @()
        foreach ($item in $computer.Items()) { $names += $item.Name }
        throw "MTP device '$DeviceName' not found. Visible devices: $($names -join ', ')"
    }

    $folder = $deviceItem.GetFolder
    $first = Get-MtpChildByName $folder $Segments[0]
    if ($null -eq $first) {
        $storage = $null
        foreach ($item in $folder.Items()) {
            if ($item.IsFolder) {
                $storage = $item
                break
            }
        }
        if ($null -eq $storage) { throw "No storage folder found under MTP device '$DeviceName'." }
        $folder = $storage.GetFolder
    }

    foreach ($segment in $Segments) {
        $next = Get-MtpChildByName $folder $segment
        if ($null -eq $next) {
            throw "MTP path segment '$segment' not found under '$DeviceName'."
        }
        $folder = $next.GetFolder
    }
    return $folder
}

function Measure-MtpItem {
    param($Item)
    $bytes = [int64]0
    $csvCount = 0
    if (-not $Item.IsFolder) {
        try { $bytes += [int64]$Item.Size } catch { }
        if ($Item.Name.ToLower().EndsWith(".csv")) { $csvCount++ }
        return [pscustomobject]@{ Bytes = $bytes; CsvCount = $csvCount }
    }

    $folder = $Item.GetFolder
    foreach ($child in $folder.Items()) {
        $m = Measure-MtpItem $child
        $bytes += [int64]$m.Bytes
        $csvCount += [int]$m.CsvCount
    }
    return [pscustomobject]@{ Bytes = $bytes; CsvCount = $csvCount }
}

function Get-MtpSensorCandidates {
    param([string]$DeviceName, [string]$DatePath, [string]$Date)
    $segments = @($DatePath.Split("\", [System.StringSplitOptions]::RemoveEmptyEntries) + $Date)
    $dateFolder = Resolve-MtpFolderAutoStorage $DeviceName $segments

    $candidates = @()
    foreach ($item in $dateFolder.Items()) {
        if (-not $item.IsFolder) { continue }
        $time = Parse-TimeStamp $item.Name
        if ($null -eq $time) { continue }
        $size = Measure-MtpItem $item
        $candidates += [pscustomobject]@{
            Name     = $item.Name
            Time     = $time
            Bytes    = $size.Bytes
            CsvCount = $size.CsvCount
            Kind     = "mtp"
            Path     = "$DeviceName\$DatePath\$Date\$($item.Name)"
            Item     = $item
        }
    }
    return @($candidates | Sort-Object Time)
}

function Select-SensorMatches {
    param(
        [object[]]$CalibrationGroups,
        [object[]]$SensorCandidates,
        [int]$MinFolderSizeMB,
        [int]$MaxDeltaMinutes,
        [int]$MinCsvCount
    )

    $minBytes = [int64]$MinFolderSizeMB * 1024 * 1024
    $usable = @($SensorCandidates | Where-Object { $_.Bytes -ge $minBytes -and $_.CsvCount -ge $MinCsvCount })
    $usedNames = New-Object "System.Collections.Generic.HashSet[string]"
    $matches = @()

    foreach ($group in $CalibrationGroups) {
        $best = $null
        $bestScore = [double]::PositiveInfinity
        foreach ($candidate in $usable) {
            if ($usedNames.Contains($candidate.Name)) { continue }
            $deltaSeconds = [math]::Abs(($candidate.Time - $group.Time).TotalSeconds)
            if ($deltaSeconds -gt ($MaxDeltaMinutes * 60)) { continue }
            $sizeBonus = [math]::Min(300.0, [math]::Log([math]::Max(1, $candidate.Bytes / 1MB)) * 20.0)
            $score = $deltaSeconds - $sizeBonus
            if ($score -lt $bestScore) {
                $best = $candidate
                $bestScore = $score
            }
        }

        if ($null -ne $best) {
            [void]$usedNames.Add($best.Name)
            $matches += [pscustomobject]@{
                Seq          = $group.Seq
                CalibrationTime = $group.Time
                SensorName   = $best.Name
                SensorTime   = $best.Time
                DeltaSeconds = [math]::Round(($best.Time - $group.Time).TotalSeconds, 1)
                Bytes        = $best.Bytes
                SizeMB       = [math]::Round($best.Bytes / 1MB, 1)
                CsvCount     = $best.CsvCount
                Sensor       = $best
                CalibrationFiles = $group.Files
                Status       = "matched"
            }
        } else {
            $matches += [pscustomobject]@{
                Seq          = $group.Seq
                CalibrationTime = $group.Time
                SensorName   = $null
                SensorTime   = $null
                DeltaSeconds = $null
                Bytes        = 0
                SizeMB       = 0
                CsvCount     = 0
                Sensor       = $null
                CalibrationFiles = $group.Files
                Status       = "unmatched"
            }
        }
    }
    return @($matches)
}

function Copy-CalibrationFiles {
    param([object[]]$CalibrationGroups, [string]$DestinationDir, [switch]$DryRun, [switch]$Overwrite)
    if ($DryRun) { return }
    New-Item -ItemType Directory -Force -Path $DestinationDir | Out-Null
    foreach ($group in $CalibrationGroups) {
        foreach ($file in $group.Files) {
            $target = Join-Path $DestinationDir $file.Name
            if ((Test-Path -LiteralPath $target) -and -not $Overwrite) { continue }
            Copy-Item -LiteralPath $file.Path -Destination $target -Force:$Overwrite
        }
    }
}

function Copy-SensorMatch {
    param($Match, [string]$SideDestination, [switch]$DryRun, [switch]$Overwrite)
    if ($Match.Status -ne "matched") { return }
    if ($DryRun) { return }

    New-Item -ItemType Directory -Force -Path $SideDestination | Out-Null
    $target = Join-Path $SideDestination $Match.SensorName
    if ((Test-Path -LiteralPath $target) -and -not $Overwrite) {
        Write-Host "skip existing $target"
        return
    }
    if ((Test-Path -LiteralPath $target) -and $Overwrite) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }

    if ($Match.Sensor.Kind -eq "local") {
        Copy-Item -LiteralPath $Match.Sensor.Path -Destination $SideDestination -Recurse -Force:$Overwrite
    } elseif ($Match.Sensor.Kind -eq "mtp") {
        $shell = New-Object -ComObject Shell.Application
        $destFolder = $shell.NameSpace($SideDestination)
        if ($null -eq $destFolder) { throw "Cannot open destination folder for Shell copy: $SideDestination" }
        $before = Test-Path -LiteralPath $target
        $destFolder.CopyHere($Match.Sensor.Item, 16)
        $deadline = (Get-Date).AddMinutes(10)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 2
            if (Test-Path -LiteralPath $target) {
                $csvCount = @(Get-ChildItem -LiteralPath $target -Recurse -File -Filter "*.csv" -ErrorAction SilentlyContinue).Count
                if ($csvCount -ge $Match.CsvCount -or ($before -eq $false -and $csvCount -gt 0)) { break }
            }
        }
    }
}

function Remove-PlaceholderSTagFolders {
    param([string]$SideDestination)
    if (-not (Test-Path -LiteralPath $SideDestination)) { return }
    $root = Resolve-Path -LiteralPath $SideDestination
    $targets = @(Get-ChildItem -LiteralPath $root -Directory | ForEach-Object {
        Get-ChildItem -LiteralPath $_.FullName -Directory | Where-Object { $_.Name -like "STag*" }
    })
    foreach ($target in $targets) {
        $resolved = Resolve-Path -LiteralPath $target.FullName
        if (-not $resolved.Path.StartsWith($root.Path)) {
            throw "Refusing to remove outside root: $resolved"
        }
        $bytes = (Get-ChildItem -LiteralPath $target.FullName -Recurse -File -ErrorAction SilentlyContinue |
            Measure-Object -Property Length -Sum).Sum
        if ($bytes -gt 102400) {
            throw "Refusing to remove non-empty-looking STag folder: $($target.FullName) bytes=$bytes"
        }
    }
    $targets | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    Write-Host "Removed $($targets.Count) placeholder STag folders under $SideDestination"
}

$subjectCalibrationDir = Join-Path $CalibrationRoot $Subject
$calibrationGroups = Parse-CalibrationFiles $subjectCalibrationDir $Date

if ($SensorRoot) {
    $sensorCandidates = Get-LocalSensorCandidates $SensorRoot
} elseif ($MtpDeviceName) {
    $sensorCandidates = Get-MtpSensorCandidates $MtpDeviceName $MtpDatePath $Date
} else {
    throw "Provide either -SensorRoot for local data or -MtpDeviceName for a connected phone."
}

$matches = Select-SensorMatches `
    -CalibrationGroups $calibrationGroups `
    -SensorCandidates $sensorCandidates `
    -MinFolderSizeMB $MinFolderSizeMB `
    -MaxDeltaMinutes $MaxDeltaMinutes `
    -MinCsvCount $MinCsvCount

$sensorDest = Join-Path $OutputRawRoot "sensor_raw\$Subject\$Side"
$calibrationDest = Join-Path $OutputRawRoot "calibration\$Subject"
$manifestDir = Join-Path $OutputRawRoot "manifests\$Subject"
$manifestPath = Join-Path $manifestDir "extract_${Side}_${Date}.json"

Write-Host ""
Write-Host "Calibration groups: $($calibrationGroups.Count)"
Write-Host "Sensor candidates:   $($sensorCandidates.Count)"
Write-Host "Matched:             $(@($matches | Where-Object Status -eq 'matched').Count)"
Write-Host "Unmatched:           $(@($matches | Where-Object Status -ne 'matched').Count)"
Write-Host ""

$matches |
    Select-Object Seq, Status, SensorName, DeltaSeconds, SizeMB, CsvCount, CalibrationTime |
    Format-Table -AutoSize

if ($DryRun) {
    Write-Host "Dry run only. Re-run without -DryRun to copy."
} else {
    Copy-CalibrationFiles $calibrationGroups $calibrationDest -Overwrite:$Overwrite
    foreach ($match in $matches) {
        Copy-SensorMatch $match $sensorDest -Overwrite:$Overwrite
    }
    if ($NoSTag) {
        Remove-PlaceholderSTagFolders $sensorDest
    }
}

New-Item -ItemType Directory -Force -Path $manifestDir | Out-Null
$manifest = [pscustomobject]@{
    subject = $Subject
    side = $Side
    date = $Date
    dry_run = [bool]$DryRun
    calibration_root = $CalibrationRoot
    sensor_root = $SensorRoot
    mtp_device_name = $MtpDeviceName
    output_raw_root = $OutputRawRoot
    min_folder_size_mb = $MinFolderSizeMB
    max_delta_minutes = $MaxDeltaMinutes
    min_csv_count = $MinCsvCount
    no_stag = [bool]$NoSTag
    matches = @($matches | ForEach-Object {
        [pscustomobject]@{
            seq = $_.Seq
            status = $_.Status
            calibration_time = if ($_.CalibrationTime) { $_.CalibrationTime.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
            calibration_files = @($_.CalibrationFiles | ForEach-Object { $_.Name })
            sensor_name = $_.SensorName
            sensor_time = if ($_.SensorTime) { $_.SensorTime.ToString("yyyy-MM-dd HH:mm:ss") } else { $null }
            delta_seconds = $_.DeltaSeconds
            size_mb = $_.SizeMB
            csv_count = $_.CsvCount
            source_path = if ($_.Sensor) { $_.Sensor.Path } else { $null }
        }
    })
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Host "Manifest: $manifestPath"
