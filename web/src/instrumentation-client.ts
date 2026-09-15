// Runs before the app becomes interactive (Next.js instrumentation-client):
// starts error tracking when the build has a Sentry DSN (lib/monitoring.ts).
import { initMonitoring } from "@/lib/monitoring";

try {
  initMonitoring();
} catch {
  // Monitoring must never stop the app from loading
}
