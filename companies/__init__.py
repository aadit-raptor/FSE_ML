"""Company data from filings in many countries (PLAN.md 4.1).

One interface (``companies.sources``) with a connector per source:

- ``sec``: SEC EDGAR company facts (United States; foreign filers' 20-F/40-F);
- ``esef``: ESEF annual reports collected by filings.xbrl.org (EU, UK and
  other listed companies), read from their xBRL-JSON;
- ``companies_house``: UK Companies House accounts filed as inline XBRL
  (needs ``COMPANIES_HOUSE_API_KEY``);
- ``edinet``: Japan's EDINET annual securities reports, read from their
  XBRL-to-CSV files (needs ``EDINET_API_KEY``).

Each connector turns a filing into the same **summary figures**
(``companies.items.SUMMARY_FIELDS``) per fiscal year, in millions of the
filing's own currency, with the accounting standard, the fiscal year end and
a link to the filing. Anything else is document upload (PLAN.md 6.2).
"""
