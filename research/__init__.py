"""Equity research (PLAN.md phase R): sector models written once as data.

A model is a table of rows and periods with a rule in each cell
(`definition`); the server works it out (`evaluate`) and the same rows are
written as live Excel formulas (`workbook`). Both read one parsed formula
(`formula`), so they cannot drift apart. Nothing here imports `core`,
`lbo_engine` or `simulation`.
"""
