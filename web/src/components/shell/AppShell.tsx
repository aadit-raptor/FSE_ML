import { CommandSearch } from "./CommandSearch";
import { Shortcuts } from "./Shortcuts";
import { StatusBar } from "./StatusBar";
import { StepBar } from "./StepBar";
import { TopBar } from "./TopBar";
import { WorkspaceProvider } from "./workspace";

/** The Guided shell: mode tabs, step row, search, content, status bar. */
export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <WorkspaceProvider>
      <div className="flex h-dvh flex-col overflow-hidden bg-canvas">
        <TopBar />
        <StepBar />
        <main id="content" className="min-h-0 flex-1 overflow-auto">
          {children}
        </main>
        <StatusBar />
      </div>
      <CommandSearch />
      <Shortcuts />
    </WorkspaceProvider>
  );
}
