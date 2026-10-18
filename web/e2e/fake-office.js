/*
 * A stand-in for Office.js (PLAN.md R3) that the browser tests serve in place of Microsoft's CDN file
 * (web/e2e/excel.spec.ts). It does what Excel does for the parts the add-in uses, and records them:
 *
 * - the dialog: displayDialogAsync opens the page in a window of its own, as Excel on the web does, and the
 *   page's messageParent reaches the pane only from this site's own origin;
 * - the workbook: every value, formula, number format, bold, font colour and size, fill, alignment, note,
 *   column width, row height, frozen pane and sheet-scoped defined name the pane asks for, kept per sheet; window.__excelWritten() answers them in the shape
 *   tests/addin_cells_check.py compares with the download.
 *
 * Plain script, no imports: it runs where Office.js would.
 */
(function () {
  "use strict";
  var DIALOG = "__fakeOfficeDialog";
  var sheets = [];

  function letters(col) {
    var out = "";
    while (col > 0) {
      var r = (col - 1) % 26;
      out = String.fromCharCode(65 + r) + out;
      col = Math.floor((col - 1) / 26);
    }
    return out;
  }
  function columnNumber(text) {
    return text.split("").reduce(function (n, ch) { return n * 26 + ch.charCodeAt(0) - 64; }, 0);
  }

  function makeSheet(name) {
    var sheet = { name: name, cells: {}, widths: {}, heights: {}, names: {}, freeze: null, active: false };
    sheets.push(sheet);
    function cell(address) {
      if (!sheet.cells[address]) sheet.cells[address] = {};
      return sheet.cells[address];
    }
    var ws = {
      name: name,
      getRange: function (address) {
        var column = /^([A-Z]+):\1$/.exec(address);
        if (column) {
          // Excel's conversion: n characters are n x 7 + 5 pixels, a pixel 0.75 points
          return { format: { set columnWidth(points) { sheet.widths[column[1]] = Math.round((points / 0.75 - 5) / 7 * 1000) / 1000; } } };
        }
        var whole = /^(\d+):\1$/.exec(address);
        if (whole) {
          return { format: { set rowHeight(points) { sheet.heights[whole[1]] = points; } } };
        }
        if (!/^[A-Z]+\d+$/.test(address)) throw new Error("fake Excel takes one cell at a time: " + address);
        var target = cell(address);
        return {
          set values(v) {
            var value = v[0][0];
            delete target.formula;
            // as Excel does, a leading apostrophe marks words and is not part of them; unmarked words starting
            // with = are a formula, which is what the add-in must never write for words
            if (typeof value === "string" && value.charAt(0) === "'") target.value = value.slice(1);
            else if (typeof value === "string" && value.charAt(0) === "=") target.formula = value;
            else target.value = value;
          },
          set formulas(v) {
            delete target.value;
            target.formula = v[0][0];
          },
          set numberFormat(v) {
            target.format = v[0][0];
          },
          format: {
            font: {
              set bold(b) { if (b) target.bold = true; else delete target.bold; },
              set color(c) { target.color = String(c).replace(/^#/, ""); },
              set size(n) { target.size = n; },
            },
            fill: { set color(c) { target.fill = String(c).replace(/^#/, ""); } },
            set horizontalAlignment(a) { target.align = String(a).toLowerCase(); },
            set wrapText(w) { if (w) target.wrap = true; },
          },
          address: address,
        };
      },
      freezePanes: {
        freezeAt: function (range) {
          var m = /^A1:([A-Z]+)(\d+)$/.exec(range);
          if (!m) throw new Error("fake Excel freezes from A1: " + range);
          sheet.freeze = letters(columnNumber(m[1]) + 1) + (Number(m[2]) + 1);
        },
        unfreeze: function () { sheet.freeze = null; },
      },
      names: {
        // worksheet-scoped names, as ws.names.add makes them (ExcelApi 1.4)
        add: function (defined, range) {
          if (sheet.names[defined]) throw new Error("fake Excel: the name " + defined + " is taken on " + name);
          var m = /^([A-Z]+)(\d+)$/.exec(range.address);
          sheet.names[defined] = "$" + m[1] + "$" + m[2];
        },
      },
      notes: {
        add: function (address, content) {
          var m = /^'((?:[^']|'')+)'!([A-Z]+\d+)$/.exec(address);
          if (!m || m[1].replace(/''/g, "'") !== name) throw new Error("fake Excel: note on another sheet " + address);
          cell(m[2]).note = content;
        },
      },
      activate: function () {
        sheets.forEach(function (s) { s.active = false; });
        sheet.active = true;
      },
    };
    return ws;
  }

  var worksheets = {
    items: [],
    load: function () {},
    add: function (name) {
      if (worksheets.items.some(function (ws) { return ws.name.toLowerCase() === name.toLowerCase(); })) {
        throw new Error("fake Excel: a sheet named " + name + " exists");
      }
      var ws = makeSheet(name);
      worksheets.items.push(ws);
      return ws;
    },
  };
  // A new workbook has one sheet, as Excel's does
  worksheets.add("Sheet1");

  window.Excel = {
    run: function (batch) {
      return Promise.resolve(batch({ workbook: { worksheets: worksheets }, sync: function () { return Promise.resolve(); } }));
    },
  };

  window.__excelWritten = function () {
    return {
      sheets: sheets.filter(function (s) { return s.name !== "Sheet1"; }).map(function (s) {
        return { name: s.name, cells: s.cells, widths: s.widths, heights: s.heights, names: s.names, freeze: s.freeze,
                 active: s.active };
      }),
    };
  };

  var listeners = [];
  window.addEventListener("message", function (event) {
    if (event.origin !== window.location.origin || !event.data || typeof event.data[DIALOG] !== "string") return;
    listeners.forEach(function (fn) { fn(event.data[DIALOG]); });
  });

  window.Office = {
    EventType: { DialogMessageReceived: "dialogMessageReceived", DialogEventReceived: "dialogEventReceived" },
    onReady: function () {
      return Promise.resolve({ host: window.opener ? null : "Excel", platform: "OfficeOnline" });
    },
    context: {
      diagnostics: { platform: "OfficeOnline" },
      requirements: { isSetSupported: function () { return true; } },
      ui: {
        displayDialogAsync: function (url, options, callback) {
          var opened = window.open(url, "office-dialog", "width=420,height=640");
          var handlers = {};
          var timer = setInterval(function () {
            if (opened && opened.closed) {
              clearInterval(timer);
              if (handlers.dialogEventReceived) handlers.dialogEventReceived({ error: 12006 });
            }
          }, 250);
          listeners.push(function (message) {
            if (handlers.dialogMessageReceived) handlers.dialogMessageReceived({ message: message, origin: window.location.origin });
          });
          callback({
            status: "succeeded",
            value: {
              addEventHandler: function (event, handler) { handlers[event] = handler; },
              close: function () { clearInterval(timer); if (opened) opened.close(); },
            },
          });
        },
        messageParent: function (message, options) {
          if (!window.opener) throw new Error("fake Office: messageParent outside a dialog");
          var target = (options && options.targetOrigin) || window.location.origin;
          var payload = {};
          payload[DIALOG] = message;
          window.opener.postMessage(payload, target);
        },
      },
    },
  };
})();
