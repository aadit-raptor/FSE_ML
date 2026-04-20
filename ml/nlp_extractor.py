"""
NLP-based extraction of financial assumptions from unstructured text.
Uses the Anthropic API for zero-shot extraction — no training needed.
Also supports direct 10-K text pasting.
"""

import json
import re
import requests
import os
from dataclasses import dataclass
from typing import Optional, Dict

ANTHROPIC_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

EXTRACTION_PROMPT = """You are a financial analyst assistant. 
Extract the following quantitative financial metrics from the text below.
Return ONLY a valid JSON object with these exact keys.
If a value cannot be found, use null.
All percentage values should be expressed as decimals (e.g. 5% = 0.05).
All dollar values should be in millions USD.

Keys to extract:
- revenue_growth_mean: management's guided or implied annual revenue growth rate
- revenue_growth_low: low end of guided growth range (if provided)
- revenue_growth_high: high end of guided growth range (if provided)
- ebitda_margin: current or guided EBITDA margin
- gross_margin: current or guided gross margin
- capex_pct: capex as % of revenue
- da_pct: depreciation and amortization as % of revenue
- ltm_revenue: last twelve months revenue
- ltm_ebitda: last twelve months EBITDA
- net_debt: total debt minus cash
- guidance_confidence: your assessment of management's confidence (low/medium/high)
- key_risks: array of 3 key risks mentioned (as strings)
- key_growth_drivers: array of 3 key growth drivers mentioned (as strings)

Text to analyze:
{text}

Return only the JSON, no preamble or explanation."""


def extract_from_text(text: str, 
                       max_chars: int = 8000) -> dict:
    """
    Extract financial assumptions from any financial text.
    text: earnings call transcript, 10-K excerpt, CIM summary, etc.
    """
    if not ANTHROPIC_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Get a free API key at console.anthropic.com"
        )

    # Truncate to avoid token limits
    truncated = text[:max_chars]
    if len(text) > max_chars:
        truncated += "\n[Text truncated for length]"

    prompt = EXTRACTION_PROMPT.format(text=truncated)

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )

    if response.status_code != 200:
        raise ValueError(f"API error {response.status_code}: {response.text}")

    content = response.json()['content'][0]['text']

    # Clean any markdown code fences
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)

    try:
        result = json.loads(content.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse API response as JSON: {e}\n{content}")

    return result


def apply_extraction_to_session(extracted: dict, session_state) -> list:
    """
    Maps extracted values to session state fields.
    Returns list of (field_name, value, description) tuples for user review.
    """
    changes = []

    def apply(key, ss_key, label, scale=1.0):
        val = extracted.get(key)
        if val is not None:
            scaled = val * scale
            setattr(session_state, ss_key, scaled) if hasattr(session_state, ss_key) \
                else session_state.__setitem__(ss_key, scaled)
            changes.append((label, scaled, key))

    # Map to Page 1 session state keys
    apply('revenue_growth_mean', 'd_growth',        'Revenue growth mean (%)', scale=100)
    apply('ebitda_margin',       None,               'EBITDA margin (%)',       scale=100)
    apply('gross_margin',        'd_gross_margin',   'Gross margin (%)',        scale=100)
    apply('capex_pct',           'd_capex',          'Capex / revenue (%)',     scale=100)
    apply('da_pct',              'd_da',             'D&A / revenue (%)',       scale=100)

    # Map to Monte Carlo Page 5 session state keys
    apply('revenue_growth_mean', 'mc_growth_mean', 'MC growth mean (%)',       scale=100)

    # Growth uncertainty from range if provided
    low  = extracted.get('revenue_growth_low')
    high = extracted.get('revenue_growth_high')
    if low is not None and high is not None:
        implied_std = (high - low) / 4  # 95% CI spans ~4 std devs
        session_state['mc_growth_std'] = implied_std * 100
        changes.append(('MC growth std dev (%)', implied_std * 100, 'derived'))

    return changes