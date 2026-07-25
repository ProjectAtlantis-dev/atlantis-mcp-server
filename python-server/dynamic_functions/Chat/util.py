"""General utility functions."""

from typing import Any, Dict, List

import atlantis


@public
async def get_unused() -> List[Dict[str, Any]]:
    """Prints all functions wo callers"""
    rows = atlantis.get_uncalled_dynamic_functions()
    await atlantis.client_data("Unused functions", rows)
    await atlantis.client_script("""
(function() {
  var tables = Array.from(document.querySelectorAll(".bot-table"));
  var table = tables.reverse().find(function(candidate) {
    return candidate.querySelector('.bot-table-cell[data-metacol="visibility"]');
  });
  if (!table) return;

  table.querySelectorAll(".bot-table-row-hover").forEach(function(row) {
    var visibilityCell = row.querySelector('[data-metacol="visibility"]');
    if (!visibilityCell || visibilityCell.textContent.trim() !== "hidden") return;

    row.querySelectorAll(".bot-table-cell").forEach(function(cell) {
      cell.style.color = "#666";
      cell.style.textShadow = "none";
      cell.querySelectorAll("*").forEach(function(child) {
        child.style.color = "#666";
      });
    });
  });
})();
""")
    return rows
