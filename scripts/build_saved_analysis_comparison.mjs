import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [inputPath, outputPath, qaDir] = process.argv.slice(2);
if (!inputPath || !outputPath || !qaDir) throw new Error("Usage: builder <input.json> <output.xlsx> <qa-dir>");
const payload = JSON.parse(await fs.readFile(inputPath, "utf8"));
const workbook = Workbook.create();

function colName(index) {
  let n = index + 1;
  let name = "";
  while (n) {
    n -= 1;
    name = String.fromCharCode(65 + (n % 26)) + name;
    n = Math.floor(n / 26);
  }
  return name;
}

function addSheet(name, rows) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const safeRows = rows.length ? rows : [["Status"], ["No data found"]];
  const columns = Math.max(...safeRows.map((row) => row.length));
  const normalized = safeRows.map((row) => [...row, ...Array(columns - row.length).fill("")]);
  const last = colName(columns - 1);
  sheet.getRange(`A1:${last}${normalized.length}`).values = normalized;
  sheet.getRange(`A1:${last}1`).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${last}${normalized.length}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#D9E2F3" },
  };
  if (normalized.length > 1) {
    sheet.getRange(`A2:${last}${normalized.length}`).format = { verticalAlignment: "top", wrapText: true };
    for (const header of ["UniqueId", "Audio", "Caller", "Destination", "call_unique_id", "tenant_id"]) {
      const textColumn = normalized[0].indexOf(header);
      if (textColumn >= 0) {
        sheet.getRangeByIndexes(1, textColumn, normalized.length - 1, 1).format.numberFormat = "@";
      }
    }
  }
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(name === "Comparison" ? 6 : 2);
  sheet.getRange(`A1:${last}${normalized.length}`).format.autofitColumns();
  for (let col = 0; col < columns; col += 1) {
    let width = 20;
    if (name === "Comparison") {
      const header = String(normalized[0][col] || "");
      width = col === 0 ? 22 : header === "Audio" ? 38 : header.includes("reason") ? 55 : header.includes("comparison") ? 22 : 18;
    }
    if (name === "Raw Data") width = col < 2 ? 22 : (col === 8 ? 70 : 24);
    sheet.getRangeByIndexes(0, col, normalized.length, 1).format.columnWidth = width;
  }
  sheet.getRange(`A1:${last}1`).format.rowHeight = 34;
  if (name === "Comparison" && normalized.length > 1) {
    const audioColumn = normalized[0].indexOf("Audio");
    if (audioColumn >= 0) {
      sheet.getRangeByIndexes(1, audioColumn, normalized.length - 1, 1).format.font = {
        color: "#0563C1",
        underline: true,
      };
    }
    const comparisonColumn = normalized[0].indexOf("needs_follow_up comparison");
    if (comparisonColumn >= 0) {
      const statusRange = sheet.getRangeByIndexes(1, comparisonColumn, normalized.length - 1, 1);
      statusRange.conditionalFormats.add("containsText", {
        text: "DIFFERENT",
        format: { fill: "#FDE2E2", font: { color: "#B91C1C", bold: true } },
      });
      statusRange.conditionalFormats.add("containsText", {
        text: "MATCH",
        format: { fill: "#DCFCE7", font: { color: "#166534", bold: true } },
      });
      statusRange.conditionalFormats.add("containsText", {
        text: "MISSING/INVALID",
        format: { fill: "#FEF3C7", font: { color: "#92400E", bold: true } },
      });
    }
  }
  return sheet;
}

addSheet("Comparison", payload.comparisonRows);
addSheet("Raw Data", payload.rawRows);

await fs.mkdir(qaDir, { recursive: true });
const comparisonLastCol = colName(payload.comparisonRows[0].length - 1);
console.log((await workbook.inspect({ kind: "table", range: `Comparison!A1:${comparisonLastCol}${Math.min(payload.comparisonRows.length, 12)}`, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 20 })).ndjson);
console.log((await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" })).ndjson);
await fs.mkdir(outputPath.substring(0, Math.max(outputPath.lastIndexOf("/"), outputPath.lastIndexOf("\\"))), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
for (const sheetName of ["Comparison", "Raw Data"]) {
  const colCount = sheetName === "Comparison" ? payload.comparisonRows[0].length : payload.rawRows[0].length;
  const previewRange = sheetName === "Comparison" ? `A1:${colName(colCount - 1)}3` : `A1:${colName(colCount - 1)}1`;
  const preview = await workbook.render({ sheetName, range: previewRange, scale: 0.75, format: "png" });
  await fs.writeFile(`${qaDir}/${sheetName.replace(" ", "_")}.png`, new Uint8Array(await preview.arrayBuffer()));
}
