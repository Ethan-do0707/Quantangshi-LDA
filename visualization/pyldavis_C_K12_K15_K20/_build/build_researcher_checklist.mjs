import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

// 解析带引号的标准 CSV，避免标签或备注中的逗号破坏列结构。
function parseCsv(text) {
  const rows = [];
  let row = [], cell = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { row.push(cell); cell = ""; }
    else if (ch === '\n') { row.push(cell.replace(/\r$/, "")); rows.push(row); row = []; cell = ""; }
    else cell += ch;
  }
  if (cell.length || row.length) { row.push(cell.replace(/\r$/, "")); rows.push(row); }
  const header = rows.shift().map((value) => value.replace(/^\uFEFF/, ""));
  return rows.filter((values) => values.some((value) => value !== "")).map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ""])));
}

const buildDir = path.dirname(fileURLToPath(import.meta.url));
const root = process.argv[2] ? path.resolve(process.argv[2]) : path.resolve(buildDir, "../../..");
const outputRoot = path.join(root, "visualization", "pyldavis_C_K12_K15_K20");
const sourcePath = path.join(outputRoot, "data", "pyldavis_three_model_topic_diagnostic_summary.csv");
const outputPath = path.join(outputRoot, "researcher", "K12_K15_K20_pyldavis_researcher_checklist.xlsx");
const renderDir = path.join(buildDir, "renders");
const sourceRows = parseCsv(await fs.readFile(sourcePath, "utf8"));

if (sourceRows.length !== 47) throw new Error(`研究者核查表预期 47 个 Topic，实际 ${sourceRows.length}`);

const workbook = Workbook.create();
const guide = workbook.worksheets.add("使用说明");
const checklist = workbook.worksheets.add("Topic核查表");
guide.showGridLines = false;
checklist.showGridLines = false;
guide.tabColor = "#315F72";
checklist.tabColor = "#B46B3B";

guide.getRange("A1:F1").merge();
guide.getRange("A1").values = [["《全唐诗》C_K12 / C_K15 / C_K20 pyLDAvis 研究者核查表"]];
guide.getRange("A1:F1").format = { fill: "#315F72", font: { bold: true, color: "#FFFFFF", size: 16 }, verticalAlignment: "center" };
guide.getRange("A1:F1").format.rowHeight = 32;
const guideRows = [
  ["用途", "记录人工视觉判断；不用于自动选择 K，也不覆盖既有文学语义审核。"],
  ["第一步", "从 comparison HTML 分别打开 C_K12、C_K15、C_K20，并确认 Topic ID。"],
  ["第二步", "观察 bubble 面积；它表示 prevalence，不表示主题质量或文学重要性。"],
  ["第三步", "只在同一个 K 内观察 bubble 的距离、邻近与重叠；不同 K 的 PC 坐标不能直接比较。"],
  ["第四步", "点击 Topic，依次查看 λ=1.0、λ≈0.6、λ≈0.2 的 Top 30 terms。"],
  ["第五步", "结合 nearest-topic cosine、既有标签与代表诗，填写末四列。"],
  ["研究者字段", "visual_separation、visual_overlap、term_distinctiveness、researcher_notes 均保持空白等待填写。"],
  ["标签状态", "K20 为已冻结人工 taxonomy；K12/K15 仅有 AI_PROVISIONAL 标签，因此 researcher_label_if_available 留空。"],
];
guide.getRange(`A3:B${guideRows.length + 2}`).values = guideRows;
guide.getRange("A3:A10").format = { fill: "#E7EFF2", font: { bold: true, color: "#274B59" }, verticalAlignment: "top" };
guide.getRange("B3:B10").format = { wrapText: true, verticalAlignment: "top" };
guide.getRange("A3:B10").format.borders = { preset: "all", style: "thin", color: "#D5DEE1" };
guide.getRange("A:A").format.columnWidth = 18;
guide.getRange("B:B").format.columnWidth = 90;
guide.getRange("A3:B10").format.autofitRows();

const headers = [
  "K", "topic_id", "researcher_label_if_available", "prevalence_percent", "nearest_topic_id", "nearest_topic_cosine",
  "relevant_terms_lambda_1_0", "relevant_terms_lambda_0_6", "relevant_terms_lambda_0_2",
  "visual_separation", "visual_overlap", "term_distinctiveness", "researcher_notes",
];
const data = sourceRows.map((row) => [
  Number(row.K), row.topic_id, row.researcher_label, Number(row.prevalence), row.nearest_topic_id, Number(row.nearest_topic_cosine),
  row.top_terms_lambda_1_0, row.top_terms_lambda_0_6, row.top_terms_lambda_0_2, "", "", "", "",
]);
checklist.getRange("A1:M1").merge();
checklist.getRange("A1").values = [["C_K12 / C_K15 / C_K20 Topic 视觉核查（47 Topics）"]];
checklist.getRange("A1:M1").format = { fill: "#315F72", font: { bold: true, color: "#FFFFFF", size: 15 }, verticalAlignment: "center" };
checklist.getRange("A2:M2").merge();
checklist.getRange("A2").values = [["末四列由研究者填写。λ=0.6 是主要观察点，不是最优参数；相近或重叠只记为“建议人工复核”。"]];
checklist.getRange("A2:M2").format = { fill: "#E7EFF2", font: { color: "#274B59", italic: true }, wrapText: true };
checklist.getRange("A4:M4").values = [headers];
checklist.getRange("A4:M4").format = { fill: "#B46B3B", font: { bold: true, color: "#FFFFFF" }, wrapText: true, verticalAlignment: "center" };
checklist.getRange(`A5:M${data.length + 4}`).values = data;
checklist.getRange(`A5:M${data.length + 4}`).format = { verticalAlignment: "top" };
checklist.getRange(`C5:C${data.length + 4}`).format.wrapText = true;
checklist.getRange(`G5:M${data.length + 4}`).format.wrapText = true;
checklist.getRange(`A4:M${data.length + 4}`).format.borders = { preset: "all", style: "thin", color: "#D9DEDB" };
checklist.getRange(`D5:D${data.length + 4}`).setNumberFormat("0.000");
checklist.getRange(`F5:F${data.length + 4}`).setNumberFormat("0.0000");
checklist.getRange(`A5:M${data.length + 4}`).conditionalFormats.add("expression", {
  formula: "=MOD(ROW(),2)=0", format: { fill: "#F7F9F8" },
});
checklist.getRange(`J5:J${data.length + 4}`).dataValidation = { rule: { type: "list", values: ["清晰", "部分", "不清晰", "待复核"] } };
checklist.getRange(`K5:K${data.length + 4}`).dataValidation = { rule: { type: "list", values: ["无明显重叠", "部分重叠", "高度重叠", "待复核"] } };
checklist.getRange(`L5:L${data.length + 4}`).dataValidation = { rule: { type: "list", values: ["清晰", "一般", "不足", "待复核"] } };
checklist.freezePanes.freezeRows(4);
checklist.freezePanes.freezeColumns(2);
checklist.getRange("A:A").format.columnWidth = 7;
checklist.getRange("B:B").format.columnWidth = 11;
checklist.getRange("C:C").format.columnWidth = 28;
checklist.getRange("D:F").format.columnWidth = 17;
checklist.getRange("G:I").format.columnWidth = 54;
checklist.getRange("J:L").format.columnWidth = 19;
checklist.getRange("M:M").format.columnWidth = 40;
checklist.getRange("A1:M4").format.rowHeight = 27;
checklist.getRange(`A5:M${data.length + 4}`).format.rowHeight = 88;

await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(renderDir, { recursive: true });
const blob = await SpreadsheetFile.exportXlsx(workbook);
await blob.save(outputPath);

// 重新导入并检查导出文件，确保值、工作表和视觉渲染可读取。
const imported = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const inspect = await imported.inspect({ kind: "sheet,region", sheetId: "Topic核查表", range: "A1:M12", maxChars: 6500, tableMaxRows: 12, tableMaxCols: 13 });
await fs.writeFile(path.join(buildDir, "workbook_inspect.txt"), inspect.ndjson, "utf8");
for (const sheetName of ["使用说明", "Topic核查表"]) {
  const preview = await imported.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(renderDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({ outputPath, rows: data.length, sheets: ["使用说明", "Topic核查表"], renderDir }, null, 2));
