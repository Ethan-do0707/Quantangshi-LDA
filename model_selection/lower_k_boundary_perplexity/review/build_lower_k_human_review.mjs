import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputRoot = path.resolve("model_selection/lower_k_boundary_perplexity");
const sourcePath = path.join(outputRoot, "review", "lower_k_human_review_source.csv");
const outputPath = path.join(outputRoot, "review", "lower_k_human_review.xlsx");
const previewDir = path.join(outputRoot, "review", "_workbook_preview");

const csvText = await fs.readFile(sourcePath, "utf8");
const workbook = await Workbook.fromCSV(csvText, { sheetName: "Review" });
const review = workbook.worksheets.getItem("Review");
const instructions = workbook.worksheets.add("Instructions");
const summary = workbook.worksheets.add("Summary");

workbook.comments.setSelf({ displayName: "User" });

instructions.showGridLines = false;
instructions.getRange("A1").values = [["《全唐诗》LDA 低 K 人工审核"]];
instructions.getRange("A3:B12").values = [
  ["项目", "说明"],
  ["目的", "检查 K=5/10/15 是否因主题数过少而出现过宽、混合或欠分。"],
  ["评分范围", "semantic / poem coherence / interpretability / overbreadth 均为 1–5。"],
  ["Overbreadth=1", "语义相对集中。"],
  ["Overbreadth=5", "明显把多个不同文学语义场挤在同一 Topic。"],
  ["Mixed flag", "若 Topic 明显混合多个可区分语义场，填写 TRUE。"],
  ["Under-split flag", "若 K 太小迫使多个语义场合并，填写 TRUE。"],
  ["证据", "Top30 words、prevalence、NPMI 和逐 Topic 审核卡均已预填。"],
  ["方法限制", "自动 alignment 仅为诊断，不等同人工语义裁决。"],
  ["填写原则", "只填写有把握的判断；不确定内容写入 notes。"],
];
instructions.getRange("A1:B1").format = { font: { bold: true, color: "#17384D", size: 16 }, verticalAlignment: "center", borders: { bottom: { style: "thin", color: "#8BA7B8" } } };
instructions.getRange("A3:B3").format = { fill: "#DCE8EF", font: { bold: true, color: "#17384D" }, borders: { preset: "outside", style: "thin", color: "#8BA7B8" } };
instructions.getRange("A4:B12").format = { wrapText: true, verticalAlignment: "top", borders: { preset: "inside", style: "thin", color: "#D8E1E6" } };
instructions.getRange("A1:A12").format.columnWidth = 22;
instructions.getRange("B1:B12").format.columnWidth = 78;
instructions.getRange("A1:B1").format.rowHeight = 30;

review.showGridLines = false;
review.freezePanes.freezeRows(1);
review.freezePanes.freezeColumns(4);
const reviewUsed = review.getUsedRange();
review.getRange("A1:P1").format = { fill: "#244A64", font: { bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: "#17384D" } };
review.getRange("A2:F31").format = { fill: "#F5F8FA" };
review.getRange("G2:H31").format = { fill: "#FFF8E7", wrapText: true, verticalAlignment: "top" };
review.getRange("I2:O31").format = { fill: "#FFF4D6", verticalAlignment: "top" };
review.getRange("P2:P31").format = { fill: "#F5F8FA", wrapText: true };
review.getRange("E2:E31").format.numberFormat = "0.00%";
review.getRange("F2:F31").format.numberFormat = "0.0000";
review.getRange("I2:L31").dataValidation = { rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 } };
review.getRange("M2:N31").dataValidation = { rule: { type: "list", values: ["TRUE", "FALSE"] } };
review.getRange("A1:P31").format.borders = { insideHorizontal: { style: "thin", color: "#E2E8F0" }, bottom: { style: "thin", color: "#CBD5E1" } };
review.getRange("A1:A31").format.columnWidth = 12;
review.getRange("B1:D31").format.columnWidth = 11;
review.getRange("E1:F31").format.columnWidth = 14;
review.getRange("G1:G31").format.columnWidth = 62;
review.getRange("H1:H31").format.columnWidth = 26;
review.getRange("I1:L31").format.columnWidth = 18;
review.getRange("M1:N31").format.columnWidth = 15;
review.getRange("O1:O31").format.columnWidth = 38;
review.getRange("P1:P31").format.columnWidth = 42;
review.getRange("A1:P1").format.rowHeight = 42;
review.tables.add("A1:P31", true, "LowerKHumanReviewTable");

summary.showGridLines = false;
summary.getRange("A1").values = [["低 K 人工审核进度与汇总"]];
summary.getRange("A3:G6").values = [
  ["K", "Topics", "Completed", "Mean Interpretability", "Mean Overbreadth", "Mixed Ratio", "Under-split Ratio"],
  [5, null, null, null, null, null, null],
  [10, null, null, null, null, null, null],
  [15, null, null, null, null, null, null],
];
summary.getRange("B4").formulas = [["=COUNTIF('Review'!$B$2:$B$31,A4)"]];
summary.getRange("B4:B6").fillDown();
summary.getRange("C4").formulas = [["=COUNTIFS('Review'!$B$2:$B$31,A4,'Review'!$I$2:$I$31,\">0\",'Review'!$J$2:$J$31,\">0\",'Review'!$K$2:$K$31,\">0\",'Review'!$L$2:$L$31,\">0\",'Review'!$M$2:$M$31,\"<>\",'Review'!$N$2:$N$31,\"<>\")"]];
summary.getRange("C4:C6").fillDown();
summary.getRange("D4").formulas = [["=IF(C4=0,\"\",SUMIFS('Review'!$K$2:$K$31,'Review'!$B$2:$B$31,A4,'Review'!$I$2:$I$31,\">0\",'Review'!$J$2:$J$31,\">0\",'Review'!$K$2:$K$31,\">0\",'Review'!$L$2:$L$31,\">0\",'Review'!$M$2:$M$31,\"<>\",'Review'!$N$2:$N$31,\"<>\")/C4)"]];
summary.getRange("D4:D6").fillDown();
summary.getRange("E4").formulas = [["=IF(C4=0,\"\",SUMIFS('Review'!$L$2:$L$31,'Review'!$B$2:$B$31,A4,'Review'!$I$2:$I$31,\">0\",'Review'!$J$2:$J$31,\">0\",'Review'!$K$2:$K$31,\">0\",'Review'!$L$2:$L$31,\">0\",'Review'!$M$2:$M$31,\"<>\",'Review'!$N$2:$N$31,\"<>\")/C4)"]];
summary.getRange("E4:E6").fillDown();
summary.getRange("F4").formulas = [["=IF(C4=0,\"\",COUNTIFS('Review'!$B$2:$B$31,A4,'Review'!$M$2:$M$31,\"TRUE\")/C4)"]];
summary.getRange("F4:F6").fillDown();
summary.getRange("G4").formulas = [["=IF(C4=0,\"\",COUNTIFS('Review'!$B$2:$B$31,A4,'Review'!$N$2:$N$31,\"TRUE\")/C4)"]];
summary.getRange("G4:G6").fillDown();
summary.getRange("A8:G10").values = [
  ["状态说明", null, null, null, null, null, null],
  ["当前评分字段保持空白，符合项目“人工标签与评分由研究者填写”的既有决策。", null, null, null, null, null, null],
  ["填写 Review 后，本页公式自动更新；Completed 应分别达到 5、10、15。", null, null, null, null, null, null],
];
summary.getRange("A8:G8").merge();
summary.getRange("A9:G9").merge();
summary.getRange("A10:G10").merge();
summary.getRange("A1:G1").format = { font: { bold: true, color: "#17384D", size: 16 }, borders: { bottom: { style: "thin", color: "#8BA7B8" } } };
summary.getRange("A3:G3").format = { fill: "#DCE8EF", font: { bold: true, color: "#17384D" }, wrapText: true, horizontalAlignment: "center" };
summary.getRange("A4:G6").format = { borders: { preset: "inside", style: "thin", color: "#D8E1E6" } };
summary.getRange("D4:E6").format.numberFormat = "0.00";
summary.getRange("F4:G6").format.numberFormat = "0.0%";
summary.getRange("A8:G8").format = { fill: "#F2E8D5", font: { bold: true, color: "#5B4636" } };
summary.getRange("A9:G10").format = { fill: "#FFF9ED", wrapText: true, font: { color: "#5B4636" } };
summary.getRange("A1:G10").format.columnWidth = 20;
summary.getRange("A1:A10").format.columnWidth = 12;
summary.getRange("A1:G1").format.rowHeight = 30;

// 临时填写一行验证公式会随人工输入更新，核验后立即恢复空白。
review.getRange("I2:N2").values = [[4, 4, 4, 5, "TRUE", "TRUE"]];
const liveFormulaCheck = await workbook.inspect({ kind: "table", range: "Summary!A3:G6", include: "values,formulas", tableMaxRows: 6, tableMaxCols: 8, maxChars: 3000 });
const liveTable = liveFormulaCheck.ndjson.trim().split(/\r?\n/).map((line) => JSON.parse(line)).find((item) => item.kind === "table");
const liveK5 = liveTable?.values?.[1];
if (!liveK5 || liveK5[2] !== 1 || liveK5[3] !== 4 || liveK5[4] !== 5 || liveK5[5] !== 1 || liveK5[6] !== 1) {
  throw new Error(`汇总公式临时输入测试失败：${JSON.stringify(liveK5)}`);
}
review.getRange("I2:N2").values = [[null, null, null, null, null, null]];

await fs.mkdir(previewDir, { recursive: true });
const previews = [];
for (const [sheetName, range] of [["Review", "A1:P12"], ["Instructions", "A1:B12"], ["Summary", "A1:G10"]]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const previewPath = path.join(previewDir, `${sheetName}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
  previews.push(previewPath);
}

const inspection = await workbook.inspect({ kind: "table", range: "Summary!A1:G10", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 8, maxChars: 5000 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
await fs.writeFile(path.join(previewDir, "inspection.ndjson"), inspection.ndjson, "utf8");
await fs.writeFile(path.join(previewDir, "formula_errors.ndjson"), errors.ndjson, "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.writeFile(path.join(previewDir, "verification.json"), JSON.stringify({ workbook: outputPath, source: sourcePath, sheetCount: 3, reviewRows: 30, previews, liveFormulaInputCheck: liveK5, formulaErrorScan: errors.ndjson }, null, 2), "utf8");
console.log(JSON.stringify({ outputPath, previewDir, reviewRows: 30, sheetCount: 3 }));
