/** Labels for the forecasting inputs (keys from core/forecasting.py). */

export type RowDef = { key: string; label: string };
export type Group = { title: string; rows: RowDef[] };

export const HISTORY_GROUPS: Group[] = [
  {
    title: "Income statement, $M",
    rows: [
      { key: "h_rev", label: "Revenue" },
      { key: "h_cogs", label: "Cost of sales (negative)" },
      { key: "h_rd", label: "R&D (negative)" },
      { key: "h_sga", label: "SG&A (negative)" },
      { key: "h_int_inc", label: "Interest income" },
      { key: "h_int_exp", label: "Interest expense (negative)" },
      { key: "h_other", label: "Other income" },
      { key: "h_tax", label: "Taxes (negative)" },
      { key: "h_da", label: "D&A (add-back)" },
      { key: "h_sbc", label: "SBC (add-back)" },
    ],
  },
  {
    title: "Balance sheet, $M",
    rows: [
      { key: "h_cash", label: "Cash" },
      { key: "h_ar", label: "Receivables" },
      { key: "h_inv", label: "Inventory" },
      { key: "h_ocurr", label: "Other current assets" },
      { key: "h_ppe", label: "PP&E, net" },
      { key: "h_nca", label: "Other non-current assets" },
      { key: "h_lta", label: "Goodwill, other LT assets" },
      { key: "h_ap", label: "Payables" },
      { key: "h_ocl", label: "Other current liabilities" },
      { key: "h_def", label: "Deferred revenue" },
      { key: "h_ltd", label: "Long-term debt" },
      { key: "h_ncl", label: "Other non-current liabilities" },
      { key: "h_cs", label: "Common stock" },
      { key: "h_re", label: "Retained earnings" },
      { key: "h_oci", label: "Other comprehensive income" },
    ],
  },
  {
    title: "Cash flow, $M",
    rows: [
      { key: "h_capex", label: "Capex" },
      { key: "h_divs", label: "Dividends" },
      { key: "h_buybacks", label: "Buybacks" },
    ],
  },
];

export const ASSUMPTION_GROUPS: Group[] = [
  {
    title: "Growth and margins, %",
    rows: [
      { key: "rev_g", label: "Revenue growth" },
      { key: "gm", label: "Gross margin" },
      { key: "rd", label: "R&D, % of sales" },
      { key: "sga", label: "SG&A, % of sales" },
      { key: "tax", label: "Tax rate" },
    ],
  },
  {
    title: "Cash flow, % of revenue",
    rows: [
      { key: "da", label: "D&A" },
      { key: "sbc", label: "SBC" },
      { key: "capex", label: "Capex" },
    ],
  },
  {
    title: "Working capital",
    rows: [
      { key: "ar_d", label: "Receivable days" },
      { key: "inv_d", label: "Inventory days" },
      { key: "ap_d", label: "Payable days" },
      { key: "ocl_pct", label: "Other current liab, % rev" },
      { key: "def_pct", label: "Deferred revenue, % rev" },
      { key: "nca_pct", label: "Other NCA, % rev" },
    ],
  },
  {
    title: "Financing and other",
    rows: [
      { key: "other_inc", label: "Other income, $M" },
      { key: "divs", label: "Dividends, $M" },
      { key: "buybacks", label: "Buybacks, $M" },
      { key: "ltd_chg", label: "Debt net change, $M" },
      { key: "r_cash", label: "Rate on cash, %" },
      { key: "r_debt", label: "Rate on debt, %" },
      { key: "min_cash", label: "Minimum cash, $M" },
    ],
  },
];

export const HISTORY_LABEL = Object.fromEntries(HISTORY_GROUPS.flatMap((g) => g.rows.map((r) => [r.key, r.label])));
export const ASSUMPTION_LABEL = Object.fromEntries(ASSUMPTION_GROUPS.flatMap((g) => g.rows.map((r) => [r.key, r.label])));
