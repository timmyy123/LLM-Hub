; LLM Hub per-user installer for direct download (Inno Setup 6.3 or later). No administrator rights: it installs to
; %LOCALAPPDATA%\Programs\LLM Hub, adds a Start menu shortcut and registers an uninstaller under Apps > Installed apps.
; Built by package-windows.ps1, which passes:
;   /DAppVersion=0.1.0 /DFileVersion=0.1.0.1 /DArch=x64|arm64 /DSourceDir=<publish folder> /DOutputDir=<folder>
;   /DDesktopDir=<repo>\desktop, and /DSign with /Sllmhub=<sign command> when a code signing certificate is set up.

#define AppName "LLM Hub"
#define AppExe "LLMHub.exe"

[Setup]
; Never change AppId: upgrades and the uninstaller find earlier installs by it.
AppId={{9C30354B-877D-4343-88DF-FEA9C2D48717}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=LLM Hub
AppPublisherURL=https://github.com/timmyy123/LLM-Hub
VersionInfoVersion={#FileVersion}
VersionInfoProductName={#AppName}
PrivilegesRequired=lowest
DefaultDirName={userpf}\{#AppName}
DisableDirPage=yes
DisableProgramGroupPage=yes
DisableReadyPage=yes
#if Arch == "arm64"
ArchitecturesAllowed=arm64
ArchitecturesInstallIn64BitMode=arm64
#else
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
#endif
MinVersion=10.0.19041
OutputDir={#OutputDir}
OutputBaseFilename=LLMHub-{#AppVersion}-win-{#Arch}-setup
SetupIconFile={#DesktopDir}\shared\icons\llmhub.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
LZMAUseSeparateProcess=yes
CloseApplications=force
RestartApplications=no
ShowLanguageDialog=auto
#ifdef Sign
SignTool=llmhub
SignedUninstaller=yes
SignToolRetryCount=4
SignToolRetryDelay=5000
#endif

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "es"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "fr"; MessagesFile: "compiler:Languages\French.isl"
Name: "de"; MessagesFile: "compiler:Languages\German.isl"
Name: "it"; MessagesFile: "compiler:Languages\Italian.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
en.DeleteData=Also delete your downloaded models, chats and settings?%n%nThey are in %1. Choose No to keep them for a later install.
es.DeleteData=¿Eliminar también los modelos descargados, los chats y la configuración?%n%nEstán en %1. Elige No para conservarlos si vuelves a instalar la aplicación.
ptbr.DeleteData=Excluir também os modelos baixados, as conversas e as configurações?%n%nEles ficam em %1. Escolha Não para mantê-los caso instale o app de novo.
fr.DeleteData=Supprimer aussi les modèles téléchargés, les discussions et les réglages ?%n%nIls se trouvent dans %1. Choisissez Non pour les conserver en vue d'une réinstallation.
de.DeleteData=Auch die heruntergeladenen Modelle, Chats und Einstellungen löschen?%n%nSie liegen in %1. Wähle Nein, um sie für eine spätere Installation zu behalten.
it.DeleteData=Eliminare anche i modelli scaricati, le chat e le impostazioni?%n%nSi trovano in %1. Scegli No per conservarli in vista di una nuova installazione.
ru.DeleteData=Удалить также скачанные модели, чаты и настройки?%n%nОни находятся в папке %1. Выберите «Нет», чтобы сохранить их для повторной установки.

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; Excludes: "*.pdb"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Models can take many gigabytes, so an interactive uninstall offers to remove the app's data too (default: keep it).
// A silent uninstall (/VERYSILENT, used by upgrades and scripts) always keeps it.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Data: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    Data := ExpandConstant('{localappdata}\LLMHub');
    if DirExists(Data) and not UninstallSilent then
      if MsgBox(FmtMessage(CustomMessage('DeleteData'), [Data]), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(Data, True, True, True);
  end;
end;
