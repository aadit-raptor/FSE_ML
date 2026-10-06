"""Economic data by country, and exchange rates (PLAN.md 4.2).

``catalogue`` says which economies, indicators and sources; ``connectors``
reads each source (IMF, World Bank, OECD, BIS, ECB, FRED) into ``Series``;
``refresh`` is the nightly task that stores them (db/economy.py); ``views``
turns what is stored into a country's current figures, the current
reference rates a floating tranche starts from, and exchange rates.

Market data never reaches a model run by itself: it fills an input the user
sees (a tranche's reference rate), which the deal then stores like any other.
"""
