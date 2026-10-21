"""The Excel add-in's manifests (PLAN.md R3), one per environment.

    python -m ops.addin_manifest          writes web/public/excel/manifest.xml and manifest-staging.xml

An add-in manifest tells Excel where the add-in's task pane lives and what
its button looks like. It is the "add-in only" XML manifest, which Excel on
Windows, on the Mac and on the web all take for sideloading (DEPLOY.md
"Excel add-in" says how a tester installs it). The pane is a page of the
web app (``/excel``), so there is nothing else to host.

Each environment's manifest has its own id, so a tester can have production
and staging side by side, and its own pages: production on variater.com,
staging on the staging branch's preview (behind Vercel's login, so only the
Vercel team can open it). ``tests/test_addin_manifest.py`` fails when a
committed manifest is not what this writes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import quoteattr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "public" / "excel"

# Changing an id makes Excel treat it as a different add-in: never change one
ENVIRONMENTS = {
    "production": {"file": "manifest.xml", "id": "6f0b1f8e-3c2a-4d7e-9a51-2b8f7c4e1d90",
                   "origin": "https://variater.com", "name": "Variater",
                   # Clerk's own host: its session check may pass through the pane, which would open any
                   # host not listed here in the system browser instead
                   "domains": ("https://clerk.variater.com",)},
    "staging": {"file": "manifest-staging.xml", "id": "a3d5c7e9-1b2f-4a6c-8e0d-5f7b9c1a3e24",
                "origin": "https://fse-ml-git-staging-aadit7.vercel.app", "name": "Variater (staging)",
                # Vercel's login stands in front of staging; the pane may open it. Staging's Clerk is the
                # development instance on clerk.accounts.dev, whose own name isn't known here: see DEPLOY.md
                "domains": ("https://vercel.com",)},
}
VERSION = "1.0.0.0"
DESCRIPTION = ("Bring checked quarterly results into Excel: statements read from a results filing, "
               "reviewed by you, with subtotals as formulas and every figure's page in a note.")
BUTTON = "Variater"
GROUP = "Variater"
TOOLTIP = "Open Variater: sign in and import a covered company's checked results."
GET_STARTED = "Variater is ready"
GET_STARTED_TEXT = "Open Variater from the Home tab to sign in and import checked results."
ICON_SIZES = (16, 32, 80)


def manifest(environment: str) -> str:
    env = ENVIRONMENTS[environment]
    origin = env["origin"]
    icon = lambda size: f"{origin}/brand/addin-{size}.png"  # noqa: E731
    pane = f"{origin}/excel"
    domains = "".join(f"\n    <AppDomain>{d}</AppDomain>" for d in (origin, *env["domains"]))
    images = "".join(f'\n          <bt:Image size="{n}" resid="Icon.{n}"/>' for n in ICON_SIZES)
    image_resources = "".join(f'\n        <bt:Image id="Icon.{n}" DefaultValue={quoteattr(icon(n))}/>'
                              for n in ICON_SIZES)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!-- The Variater Excel add-in, {environment} (PLAN.md R3). Written by python -m ops.addin_manifest; do not edit. -->
<OfficeApp xmlns="http://schemas.microsoft.com/office/appforoffice/1.1"
           xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xmlns:bt="http://schemas.microsoft.com/office/officeappbasictypes/1.0"
           xmlns:ov="http://schemas.microsoft.com/office/taskpaneappversionoverrides"
           xsi:type="TaskPaneApp">
  <Id>{env["id"]}</Id>
  <Version>{VERSION}</Version>
  <ProviderName>Variater</ProviderName>
  <DefaultLocale>en-US</DefaultLocale>
  <DisplayName DefaultValue={quoteattr(env["name"])}/>
  <Description DefaultValue={quoteattr(DESCRIPTION)}/>
  <IconUrl DefaultValue={quoteattr(icon(32))}/>
  <HighResolutionIconUrl DefaultValue={quoteattr(f"{origin}/brand/addin-64.png")}/>
  <SupportUrl DefaultValue={quoteattr(f"{origin}/support")}/>
  <AppDomains>{domains}
  </AppDomains>
  <Hosts>
    <Host Name="Workbook"/>
  </Hosts>
  <Requirements>
    <Sets DefaultMinVersion="1.1">
      <Set Name="ExcelApi" MinVersion="1.7"/>
    </Sets>
  </Requirements>
  <DefaultSettings>
    <SourceLocation DefaultValue={quoteattr(pane)}/>
  </DefaultSettings>
  <Permissions>ReadWriteDocument</Permissions>
  <VersionOverrides xmlns="http://schemas.microsoft.com/office/taskpaneappversionoverrides" xsi:type="VersionOverridesV1_0">
    <Hosts>
      <Host xsi:type="Workbook">
        <DesktopFormFactor>
          <GetStarted>
            <Title resid="GetStarted.Title"/>
            <Description resid="GetStarted.Description"/>
            <LearnMoreUrl resid="GetStarted.LearnMoreUrl"/>
          </GetStarted>
          <ExtensionPoint xsi:type="PrimaryCommandSurface">
            <OfficeTab id="TabHome">
              <Group id="Variater.Group">
                <Label resid="Group.Label"/>
                <Icon>{images}
                </Icon>
                <Control xsi:type="Button" id="Variater.Open">
                  <Label resid="Button.Label"/>
                  <Supertip>
                    <Title resid="Button.Label"/>
                    <Description resid="Button.Tooltip"/>
                  </Supertip>
                  <Icon>{images}
                  </Icon>
                  <Action xsi:type="ShowTaskpane">
                    <TaskpaneId>VariaterPane</TaskpaneId>
                    <SourceLocation resid="Pane.Url"/>
                  </Action>
                </Control>
              </Group>
            </OfficeTab>
          </ExtensionPoint>
        </DesktopFormFactor>
      </Host>
    </Hosts>
    <Resources>
      <bt:Images>{image_resources}
      </bt:Images>
      <bt:Urls>
        <bt:Url id="GetStarted.LearnMoreUrl" DefaultValue={quoteattr(f"{origin}/support")}/>
        <bt:Url id="Pane.Url" DefaultValue={quoteattr(pane)}/>
      </bt:Urls>
      <bt:ShortStrings>
        <bt:String id="GetStarted.Title" DefaultValue={quoteattr(GET_STARTED)}/>
        <bt:String id="Group.Label" DefaultValue={quoteattr(GROUP)}/>
        <bt:String id="Button.Label" DefaultValue={quoteattr(BUTTON)}/>
      </bt:ShortStrings>
      <bt:LongStrings>
        <bt:String id="GetStarted.Description" DefaultValue={quoteattr(GET_STARTED_TEXT)}/>
        <bt:String id="Button.Tooltip" DefaultValue={quoteattr(TOOLTIP)}/>
      </bt:LongStrings>
    </Resources>
  </VersionOverrides>
</OfficeApp>
"""


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for environment, env in ENVIRONMENTS.items():
        (OUT / env["file"]).write_text(manifest(environment), encoding="utf-8")
        print(f"wrote web/public/excel/{env['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
