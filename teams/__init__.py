"""Teams (PLAN.md 7.2): who people are, and the emails sent when someone is mentioned.

The database holds teams, shares and comments (db/teams.py) by user id only.
``directory`` asks the identity provider for a person's name or email address
when a screen or an email needs it and keeps neither; ``mail`` sends; ``notices``
works through the mentions owed an email, in batches inside the free plan.
"""
