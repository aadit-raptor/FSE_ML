/**
 * Server limits the inputs respect (PLAN.md 1.6). Mirrors api/limits.py, where
 * the numbers are explained; the API refuses anything larger with a message.
 */
export const MAX_SIMULATION_PATHS = 100_000;
