import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = path.resolve("model_selection/lower_k_semantic_audit");
const sourcePath = path.join(root, "data", "researcher_review_workbook_source.csv");
const payloadPath = path.join(root, "data", "researcher_review_workbook_payload.json");
const outputPath = path.join(root, "QTS_Lower_K_Researcher_Review.xlsx");
const previewDir = path.join(root, "_workbook_preview");

const csvText = await fs.readFile(sourcePath, "utf8");
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Topic Review" });
const reviewSheet = workbook.worksheets.getItem("Topic Review");
workbook.comments.setSelf({ displayName: "AI 辅助审核" });

// 把 JSON 记录写成独立工作表，避免在浏览器端重复解析复杂 CSV。
function addRecordSheet(name, records) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (!records.length) return sheet;
  const headers = Object.keys(records[0]);
  const matrix = [headers, ...records.map((record) => headers.map((header) => record[header] ?? null))];
  const lastColumn = columnName(headers.length);
  sheet.getRange(`A1:${lastColumn}${matrix.length}`).values = matrix;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: "#244A64",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
    verticalAlignment: "center",
    horizontalAlignment: "center",
    borders: { bottom: { style: "thin", color: "#17384D" } },
  };
  sheet.getRange(`A2:${lastColumn}${matrix.length}`).format = {
    verticalAlignment: "top",
    wrapText: true,
    borders: { insideHorizontal: { style: "thin", color: "#E2E8F0" } },
  };
  sheet.freezePanes.freezeRows(1);
  sheet.getRange(`A1:${lastColumn}1`).format.rowHeight = 42;
  return sheet;
}

function columnName(number) {
  let value = number;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

const instructions = workbook.worksheets.add("Instructions");
instructions.showGridLines = false;
instructions.getRange("A1:B13").values = [
  ["《全唐诗》Lower-K 研究者语义复核", null],
  [null, null],
  ["项目", "说明"],
  ["审核状态", "AI_REVIEW / PROVISIONAL_REVIEW / NEEDS_RESEARCHER_CONFIRMATION；AI 分数不是人工审核结果。"],
  ["审核范围", "C_K5、C_K10、C_K15 共 30 Topics；每题已读 Top30 words、Top20 代表诗及 C_K20 alignment。"],
  ["Researcher 字段", "Topic Review 的 Q:Y 列全部保持空白，供研究者填写。"],
  ["评分", "Word Coherence、Poem Coherence、Interpretability、Overbreadth 均为 1–5。"],
  ["布尔值", "Mixed Flag 与 UnderSplit Flag 只填 TRUE 或 FALSE。"],
  ["Overall Type", "只填 CLEAR、BROAD、MIXED、NOISE。"],
  ["证据入口", "Topic Review 保留 Top30 与 Top20 来源路径；逐题全文证据见 review_cards。"],
  ["不可变性", "本工作簿未重新训练模型，也未修改冻结矩阵、词表或 C_K20 taxonomy。"],
  ["当前建议", "K15_COMPETITIVE_WITH_CAVEATS；下一步 REVIEW_K15_TO_K20_FINE_GRID。"],
  ["运行边界", "本轮未运行 K16–19，等待研究者确认。"],
];
instructions.getRange("A1:B1").merge();
instructions.getRange("A1:B1").format = { fill: "#17384D", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
instructions.getRange("A1:B1").format.rowHeight = 32;
instructions.getRange("A3:B3").format = { fill: "#DCE8EF", font: { bold: true, color: "#17384D" }, borders: { preset: "outside", style: "thin", color: "#8BA7B8" } };
instructions.getRange("A4:B13").format = { wrapText: true, verticalAlignment: "top", borders: { insideHorizontal: { style: "thin", color: "#D8E1E6" } } };
instructions.getRange("A1:A13").format.columnWidth = 23;
instructions.getRange("B1:B13").format.columnWidth = 82;

reviewSheet.showGridLines = false;
reviewSheet.freezePanes.freezeRows(1);
reviewSheet.freezePanes.freezeColumns(4);
reviewSheet.getRange("A1:P1").format = {
  fill: "#244A64",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
reviewSheet.getRange("Q1:Y1").format = {
  fill: "#8A5A22",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
reviewSheet.getRange("A2:P31").format = { fill: "#F4F8FA", verticalAlignment: "top", wrapText: true };
reviewSheet.getRange("Q2:Y31").format = { fill: "#FFF4D6", verticalAlignment: "top", wrapText: true };
reviewSheet.getRange("A1:Y31").format.borders = {
  insideHorizontal: { style: "thin", color: "#E2E8F0" },
  bottom: { style: "thin", color: "#CBD5E1" },
};
reviewSheet.getRange("R2:U31").dataValidation = { rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 } };
reviewSheet.getRange("V2:W31").dataValidation = { rule: { type: "list", values: ["TRUE", "FALSE"] } };
reviewSheet.getRange("X2:X31").dataValidation = { rule: { type: "list", values: ["CLEAR", "BROAD", "MIXED", "NOISE"] } };
reviewSheet.getRange("A1:Y1").format.rowHeight = 48;
reviewSheet.getRange("A1:B31").format.columnWidth = 9;
reviewSheet.getRange("C1:C31").format.columnWidth = 31;
reviewSheet.getRange("D1:D31").format.columnWidth = 13;
reviewSheet.getRange("E1:H31").format.columnWidth = 14;
reviewSheet.getRange("I1:J31").format.columnWidth = 13;
reviewSheet.getRange("K1:L31").format.columnWidth = 22;
reviewSheet.getRange("M1:N31").format.columnWidth = 35;
reviewSheet.getRange("O1:P31").format.columnWidth = 48;
reviewSheet.getRange("Q1:Q31").format.columnWidth = 28;
reviewSheet.getRange("R1:U31").format.columnWidth = 20;
reviewSheet.getRange("V1:X31").format.columnWidth = 19;
reviewSheet.getRange("Y1:Y31").format.columnWidth = 42;
reviewSheet.tables.add("A1:Y31", true, "LowerKResearcherReviewTable");

const summarySheet = addRecordSheet("K Summary", payload.summary);
summarySheet.getRange("A1:N4").format.columnWidth = 18;
summarySheet.getRange("B2:M4").format.numberFormat = "0.0000";
summarySheet.getRange("H2:M4").format.numberFormat = "0.0%";
summarySheet.getRange("N1:N4").format.columnWidth = 31;

const k15Sheet = addRecordSheet("K15 vs K20", payload.k15_audit);
k15Sheet.getRange("A1:A16").format.columnWidth = 12;
k15Sheet.getRange("B1:B16").format.columnWidth = 33;
k15Sheet.getRange("C1:D16").format.columnWidth = 45;
k15Sheet.getRange("E1:J16").format.columnWidth = 25;
k15Sheet.getRange("K1:K16").format.columnWidth = 65;
k15Sheet.getRange("L1:M16").format.columnWidth = 28;

const coverageSheet = addRecordSheet("Core Coverage", payload.coverage);
coverageSheet.getRange("A1:A10").format.columnWidth = 14;
coverageSheet.getRange("B1:B10").format.columnWidth = 38;
coverageSheet.getRange("C1:C10").format.columnWidth = 16;
coverageSheet.getRange("D1:D10").format.columnWidth = 40;
coverageSheet.getRange("E1:H10").format.columnWidth = 28;

const prioritySheet = addRecordSheet("Priority Queue", payload.priority);
prioritySheet.getRange("A1:D31").format.columnWidth = 15;
prioritySheet.getRange("E1:E31").format.columnWidth = 40;
prioritySheet.getRange("F1:H31").format.columnWidth = 24;
prioritySheet.getRange("I1:I31").format.columnWidth = 65;

// 导出前直接检查最终内存工作簿，确保研究者字段没有被预填。
const blankInspection = await workbook.inspect({
  kind: "table",
  range: "'Topic Review'!Q1:Y31",
  include: "values,formulas",
  tableMaxRows: 31,
  tableMaxCols: 9,
  maxChars: 20000,
});
const blankTable = blankInspection.ndjson.trim().split(/\r?\n/).map((line) => JSON.parse(line)).find((item) => item.kind === "table");
const researcherValues = (blankTable?.values ?? []).slice(1).flat();
const researcherFieldsBlank = researcherValues.length === 270 && researcherValues.every((value) => value === null || value === "");
if (!researcherFieldsBlank) {
  throw new Error("Researcher_* 字段存在非空值，停止导出。");
}

const errorScan = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "最终公式错误扫描",
});

await fs.mkdir(previewDir, { recursive: true });
const previews = [];
for (const [sheetName, range] of [
  ["Topic Review", "A1:Y12"],
  ["Instructions", "A1:B13"],
  ["K Summary", "A1:N4"],
  ["K15 vs K20", "A1:M10"],
  ["Core Coverage", "A1:H10"],
  ["Priority Queue", "A1:I15"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const previewPath = path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  previews.push(previewPath);
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.writeFile(path.join(previewDir, "blank_inspection.ndjson"), blankInspection.ndjson, "utf8");
await fs.writeFile(path.join(previewDir, "formula_errors.ndjson"), errorScan.ndjson, "utf8");
await fs.writeFile(
  path.join(previewDir, "verification.json"),
  JSON.stringify({
    workbook: outputPath,
    source: sourcePath,
    sheetCount: 6,
    reviewRows: 30,
    researcherColumns: 9,
    researcherBlankCells: researcherValues.length,
    researcherFieldsBlank,
    formulaErrorScan: errorScan.ndjson,
    previews,
  }, null, 2),
  "utf8",
);

console.log(JSON.stringify({ outputPath, sheetCount: 6, reviewRows: 30, researcherFieldsBlank, previews }, null, 2));
