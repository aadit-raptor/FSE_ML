"""File uploads (PLAN.md 6.1): PDF, Excel and Word files, private to their owner.

- ``rules``: every number and why (sizes, quotas, how long a file is kept).
- ``scan``: what makes a file refused, and the scanner itself.
- ``store``: where the bytes are kept (Supabase Storage, or a directory locally).
- ``service``: the steps of an upload, a download link, a deletion and the sweep.

Rows are in ``db/files.py``; the endpoints in ``api/routers/files.py``.
"""
