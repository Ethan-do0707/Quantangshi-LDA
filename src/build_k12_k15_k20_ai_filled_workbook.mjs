import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import JSZip from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/jszip/lib/index.js";
import { FileBlob, SpreadsheetFile } from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = process.cwd();
const taskRoot = path.join(root, "model_selection", "K12_K15_K20_human_adjudication");
const aiDir = path.join(taskRoot, "AI_review");
const auditDir = path.join(aiDir, "audit", "post_edit");
const inputPath = path.join(taskRoot, "QTS_K12_K15_K20_Human_Review.before_AI_review.xlsx");
const outputPath = path.join(taskRoot, "QTS_K12_K15_K20_Human_Review_AI_Filled.xlsx");
await fs.mkdir(auditDir, { recursive: true });

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const nonEmpty = rows.filter((values) => values.some((value) => value !== ""));
  const headers = nonEmpty[0];
  const body = nonEmpty.slice(1);
  return {
    headers,
    rows: body.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]))),
  };
}

async function loadCsv(filePath) {
  return parseCsv(await fs.readFile(filePath, "utf8"));
}

function columnLetter(number) {
  let value = number;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

function typedValue(header, value) {
  if (value === "") return null;
  if (/^(Unblinded_K|Unblinded_Topic_ID|K|Topic_Count|Priority|.*_Count)$/.test(header)) return Number(value);
  if (/^(AI_Provisional_(Word_Coherence|Poem_Coherence|Word_Poem_Agreement|Labelability|Literary_Interpretability|Semantic_Specificity|Overbreadth|Internal_Mixedness)|.*_Ratio|Mean_.*)$/.test(header)) return Number(value);
  return value;
}

const colors = {
  navy: "#17324D",
  white: "#FFFFFF",
  slate: "#334155",
  grid: "#D9E1E7",
  paleYellow: "#FFF4CC",
  paleGreen: "#E5F4EA",
  paleRed: "#FDE8E8",
  paleGray: "#F5F7F9",
};
const font = "Microsoft YaHei";

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: colors.navy,
    font: { name: font, bold: true, color: colors.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    rowHeightPx: 48,
    borders: { preset: "all", style: "thin", color: colors.grid },
  };
}

function writeTableSheet(workbook, name, parsed, tableName, widths = {}, rowHeightPx = 54) {
  const sheet = workbook.worksheets.add(name);
  const headers = parsed.headers;
  const rows = parsed.rows.map((record) => headers.map((header) => typedValue(header, record[header])));
  const lastColumn = columnLetter(headers.length);
  const lastRow = rows.length + 1;
  sheet.getRange("A1:" + lastColumn + lastRow).values = [headers, ...rows];
  styleHeader(sheet, "A1:" + lastColumn + "1");
  if (rows.length) {
    sheet.getRange("A2:" + lastColumn + lastRow).format = {
      font: { name: font, size: 9, color: colors.slate },
      verticalAlignment: "top",
      wrapText: true,
      rowHeightPx,
      borders: { preset: "all", style: "thin", color: colors.grid },
    };
  }
  headers.forEach((header, index) => {
    let width = widths[header];
    if (!width && /Evidence|Comment|Rationale/.test(header)) width = 300;
    if (!width && /Notes|Subthemes|Risk/.test(header)) width = 360;
    sheet.getRange(columnLetter(index + 1) + "1:" + columnLetter(index + 1) + lastRow).format.columnWidthPx = width ?? 118;
  });
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add("A1:" + lastColumn + lastRow, true, tableName);
  return sheet;
}

const inputBytes = await fs.readFile(inputPath);
const inputSha256 = crypto.createHash("sha256").update(inputBytes).digest("hex");
const expectedInputSha256 = "04ae77e31b25691741a219ef3de953597803f1ab56f7c573952702d68ab7927a";
if (inputSha256 !== expectedInputSha256) {
  throw new Error("备份工作簿哈希异常：" + inputSha256);
}

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
const originalSheets = workbook.worksheets.items.map((sheet) => sheet.name);
const expectedOriginalSheets = [
  "Instructions",
  "Blind_Topic_Review",
  "Blind_Poem_Evidence",
  "Model_Level_Review",
  "K12_to_K15_Splits",
  "K15_to_K20_Splits",
  "Fragmentation_Review",
  "Semantic_Family_Coverage",
  "Final_Evidence_Matrix",
  "Final_Decision",
  "Hidden_AI_Reference",
];
if (JSON.stringify(originalSheets) !== JSON.stringify(expectedOriginalSheets)) {
  throw new Error("原工作表清单意外：" + originalSheets.join(", "));
}

const provisional = await loadCsv(path.join(aiDir, "AI_topic_review_unblinded.csv"));
const summary = await loadCsv(path.join(aiDir, "AI_model_summary.csv"));
const splitMerge = await loadCsv(path.join(aiDir, "AI_split_merge_review.csv"));
const fragmentation = await loadCsv(path.join(aiDir, "AI_fragmentation_review.csv"));

const provisionalSheet = writeTableSheet(
  workbook,
  "AI_Provisional_Review",
  provisional,
  "AIProvisionalReviewTable",
  {
    blind_topic_id: 92,
    Unblinded_K: 82,
    Unblinded_Topic_ID: 105,
    AI_Provisional_Label: 245,
    AI_Provisional_Suspected_Subthemes: 330,
    AI_Provisional_Evidence: 520,
    AI_Provisional_Notes: 520,
    AI_Challenge_Status: 145,
    AI_Challenge_Comment: 500,
    AI_Missing_Semantic_Risk: 420,
  },
  88,
);
const typeColumn = provisional.headers.indexOf("AI_Provisional_Topic_Type") + 1;
const needsColumn = provisional.headers.indexOf("AI_Provisional_Needs_Splitting") + 1;
const challengeColumn = provisional.headers.indexOf("AI_Challenge_Status") + 1;
for (let row = 2; row <= provisional.rows.length + 1; row += 1) {
  const record = provisional.rows[row - 2];
  const typeCell = provisionalSheet.getRange(columnLetter(typeColumn) + row);
  if (record.AI_Provisional_Topic_Type === "CLEAR") typeCell.format.fill = colors.paleGreen;
  if (record.AI_Provisional_Topic_Type === "BROAD") typeCell.format.fill = colors.paleYellow;
  if (["MIXED", "NOISE"].includes(record.AI_Provisional_Topic_Type)) typeCell.format.fill = colors.paleRed;
  if (record.AI_Provisional_Needs_Splitting === "YES") {
    provisionalSheet.getRange(columnLetter(needsColumn) + row).format.fill = colors.paleRed;
  }
  if (record.AI_Challenge_Status !== "NOT_REQUIRED") {
    provisionalSheet.getRange(columnLetter(challengeColumn) + row).format.fill = colors.paleYellow;
  }
}

const summarySheet = writeTableSheet(
  workbook,
  "AI_Model_Summary",
  summary,
  "AIModelSummaryTable",
  { AI_Overall_Status: 210 },
  46,
);
summary.headers.forEach((header, index) => {
  const column = columnLetter(index + 1);
  if (/_Ratio$/.test(header)) {
    summarySheet.getRange(column + "2:" + column + (summary.rows.length + 1)).format.numberFormat = "0.0%";
  }
  if (/^Mean_/.test(header)) {
    summarySheet.getRange(column + "2:" + column + (summary.rows.length + 1)).format.numberFormat = "0.00";
  }
});

writeTableSheet(
  workbook,
  "AI_Split_Merge_Review",
  splitMerge,
  "AISplitMergeReviewTable",
  {
    lower_blind_topic_ids: 190,
    higher_blind_topic_ids: 190,
    AI_Provisional_Split_Judgment: 220,
    AI_Rationale: 560,
    AI_Text_Evidence: 480,
  },
  92,
);

writeTableSheet(
  workbook,
  "AI_Fragmentation_Review",
  fragmentation,
  "AIFragmentationReviewTable",
  { AI_Provisional_Fragmentation_Status: 235, AI_Rationale: 600 },
  86,
);

const disagreementHeaders = [
  "blind_topic_id",
  "metric",
  "AI_value",
  "Researcher_value",
  "agreement_status",
];
const disagreementSheet = workbook.worksheets.add("AI_Researcher_Disagreement");
disagreementSheet.getRange("A1:E3").values = [
  disagreementHeaders,
  ["", "", "", "", ""],
  ["", "", "", "", ""],
];
styleHeader(disagreementSheet, "A1:E1");
disagreementSheet.getRange("A2:E3").format = {
  font: { name: font, size: 9, color: colors.slate },
  fill: colors.paleGray,
  wrapText: true,
  rowHeightPx: 42,
  borders: { preset: "all", style: "thin", color: colors.grid },
};
disagreementSheet.freezePanes.freezeRows(1);
disagreementSheet.tables.add("A1:E3", true, "AIResearcherDisagreementTable");
disagreementHeaders.forEach((header, index) => {
  const width = /value|status/.test(header) ? 220 : 140;
  disagreementSheet.getRange(columnLetter(index + 1) + "1:" + columnLetter(index + 1) + "3").format.columnWidthPx = width;
});

const previewRanges = {
  AI_Provisional_Review: "A1:U10",
  AI_Model_Summary: "A1:" + columnLetter(summary.headers.length) + "4",
  AI_Split_Merge_Review: "A1:L10",
  AI_Fragmentation_Review: "A1:I10",
  AI_Researcher_Disagreement: "A1:E3",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const rendered = await workbook.render({ sheetName, range, scale: 0.72, format: "png" });
  await fs.writeFile(path.join(auditDir, sheetName + ".png"), new Uint8Array(await rendered.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 22000,
  tableMaxRows: 4,
  tableMaxCols: 10,
  tableMaxCellChars: 120,
});
await fs.writeFile(path.join(auditDir, "workbook_inspection.json"), JSON.stringify(inspection, null, 2) + "\n", "utf8");
const formulaTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
const inspectionText = JSON.stringify(inspection);
const formulaErrors = formulaTokens.filter((token) => inspectionText.includes(token));
await fs.writeFile(
  path.join(auditDir, "workbook_formula_errors.json"),
  JSON.stringify({ error_count: formulaErrors.length, errors: formulaErrors }, null, 2) + "\n",
  "utf8",
);

// artifact-tool 负责内容、样式、表格和渲染；下段仅恢复其当前未公开的原工作表可见性状态。
const originalArchive = await JSZip.loadAsync(inputBytes);
const originalWorkbookXml = await originalArchive.file("xl/workbook.xml").async("string");
const originalStates = new Map();
for (const tag of originalWorkbookXml.match(/<(?:[A-Za-z0-9_]+:)?sheet\b[^>]*\/>/g) ?? []) {
  const name = tag.match(/\bname="([^"]+)"/)?.[1];
  const state = tag.match(/\bstate="([^"]+)"/)?.[1] ?? "visible";
  if (name) originalStates.set(name, state);
}

const rawXlsx = await SpreadsheetFile.exportXlsx(workbook);
const rawPath = path.join(auditDir, "_ai_filled_before_visibility_hardening.xlsx");
await rawXlsx.save(rawPath);
const archive = await JSZip.loadAsync(await fs.readFile(rawPath));
const workbookXml = await archive.file("xl/workbook.xml").async("string");
const hardenedXml = workbookXml.replace(/<(?:[A-Za-z0-9_]+:)?sheet\b[^>]*\/>/g, (tag) => {
  const name = tag.match(/\bname="([^"]+)"/)?.[1];
  if (!name) return tag;
  const withoutState = tag.replace(/\sstate="[^"]*"/g, "");
  const desiredState = originalStates.get(name) ?? "visible";
  if (desiredState === "visible") return withoutState;
  return withoutState.replace(/\s*\/>$/, ' state="' + desiredState + '"/>');
});
archive.file("xl/workbook.xml", hardenedXml);
const finalBytes = await archive.generateAsync({
  type: "uint8array",
  compression: "DEFLATE",
  compressionOptions: { level: 6 },
});
await fs.writeFile(outputPath, finalBytes);
await fs.unlink(rawPath);
try {
  await fs.unlink(rawPath + ".inspect.ndjson");
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

const outputSha256 = crypto.createHash("sha256").update(finalBytes).digest("hex");
const result = {
  status: "AI_FILLED_WORKBOOK_CREATED",
  input_sha256: inputSha256,
  output_sha256: outputSha256,
  original_sheet_count: originalSheets.length,
  output_sheet_count: originalSheets.length + Object.keys(previewRanges).length,
  original_sheet_states: Object.fromEntries(originalStates),
  new_visible_sheets: Object.keys(previewRanges),
  formula_error_count: formulaErrors.length,
  output: outputPath,
};
await fs.writeFile(path.join(auditDir, "workbook_build_result.json"), JSON.stringify(result, null, 2) + "\n", "utf8");
console.log(JSON.stringify(result, null, 2));
