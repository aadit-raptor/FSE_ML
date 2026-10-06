"""Sourced starting assumptions by region, sector and size (PLAN.md 4.3).

- ``catalogue``: the published data sets read (Aswath Damodaran's industry
  averages, NYU Stern, one file per data set and region), which country
  belongs to which region, and how thin data falls back;
- ``damodaran``: reading those workbooks into compact tables;
- ``starting``: a new deal's starting figures from the tables, the economic
  data of PLAN.md 4.2 and the country tax rates, each with its source,
  sample and date;
- ``refresh``: the scheduled ``benchmarks-refresh`` that stores the tables;
- ``record``: recording real workbooks as trimmed test fixtures.
"""
