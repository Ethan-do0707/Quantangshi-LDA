/** 使用 artifact-tool 生成作者时期 Topic 趋势结构主工作簿。 */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const scriptPath = fileURLToPath(import.meta.url);
const root = path.resolve(process.env.QTS_REPO_ROOT ?? path.resolve(path.dirname(scriptPath), ".."));
const configPath = path.resolve(root, process.argv[2] ?? "config/trend_structure_config.yaml");
const config = JSON.parse(await fs.readFile(configPath, "utf8"));
const outputRoot = path.resolve(root, config.output_root);
const dataRoot = path.join(outputRoot, "data");
const previewRoot = path.join(outputRoot, ".artifact_work", "previews");
await fs.mkdir(previewRoot, { recursive: true });


function parseCsv(text) {
  /** 解析 RFC 4180 风格 CSV，保留引号内换行与逗号。 */
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
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
      row.push(field.endsWith("\r") ? field.slice(0, -1) : field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}


function coerce(value) {
  /** 把 CSV 标量恢复为工作簿可用的布尔值、数值或文本。 */
  if (value === "") return null;
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}


async function readCsv(name) {
  /** 读取一个分析 CSV，并返回表头和类型化数据。 */
  const text = await fs.readFile(path.join(dataRoot, `${name}.csv`), "utf8");
  const rows = parseCsv(text);
  const headers = rows[0];
  const data = rows.slice(1).filter((row) => row.length > 1 || row[0] !== "").map((row) => row.map(coerce));
  return { headers, data };
}


function columnName(index) {
  /** 把零基列号转换为 Excel 列名。 */
  let value = index + 1;
  let name = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}


function findColumn(headers, field) {
  /** 返回字段对应的一基 Excel 列名。 */
  const index = headers.indexOf(field);
  if (index < 0) throw new Error(`缺少字段：${field}`);
  return columnName(index);
}


const sheetSpecs = [
  ["Period_Counts", "period_counts", "三套审计样本的四时期诗歌数与作者数"],
  ["Period_Topic_Means", "period_topic_means", "六规格 Period × Topic 描述统计；author period 不是作品创作年代"],
  ["Bootstrap_CI", "period_topic_bootstrap_ci", "按时期以作者为 cluster 的2,000次 bootstrap 95%区间"],
  ["Trend_Structure", "topic_trend_structure", "20 Topic × 六规格的 peak、rank、shape 与相邻时期差值"],
  ["Trend_Robustness", "topic_trend_robustness", "六规格综合稳健性；规则来自 trend_structure_config.yaml"],
  ["Poem_vs_Author", "poem_vs_author_trend_comparison", "每套样本 poem-weighted 与 author-balanced 的四期曲线诊断"],
  ["Author_Concentration", "period_topic_author_concentration", "各时期 Topic probability mass 的作者集中度"],
  ["CLEAR_Topics", "clear_topic_trend_structure", "taxonomy 中全部 CLEAR Topics 的趋势结构摘要"],
  ["Candidate_Trends", "trend_candidate_topics", "供后续人工审核的 candidate robust trends；不构成文学史解释"],
  ["Sensitivity_Flags", "sensitivity_flags", "聚合、样本、方向与 metadata 风险标记"],
  ["Data_Dictionary", "data_dictionary", "核心分析表的字段级定义"],
];

const datasets = {};
for (const [, fileName] of sheetSpecs) datasets[fileName] = await readCsv(fileName);
const summary = JSON.parse(await fs.readFile(path.join(dataRoot, "analysis_summary.json"), "utf8"));

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
const sheets = new Map([["Overview", overview]]);
for (const [sheetName] of sheetSpecs) sheets.set(sheetName, workbook.worksheets.add(sheetName));

const colors = {
  navy: "#17324D",
  teal: "#2D6F73",
  paleTeal: "#DDEEEF",
  paleBlue: "#EAF1F7",
  gold: "#C99A3D",
  paleGold: "#F7EBCF",
  red: "#A94442",
  paleRed: "#F8DDDA",
  green: "#3F7D52",
  paleGreen: "#DDEEDD",
  gray: "#5C6770",
  paleGray: "#EEF1F3",
  white: "#FFFFFF",
};


function applyTextConditionalFormatting(sheet, headers, dataLength) {
  /** 对风险、稳健性和标记字段应用可更新的条件格式。 */
  const targets = [
    ["trend_robustness", "HIGH", colors.paleGreen, colors.green],
    ["trend_robustness", "MODERATE", colors.paleGold, "#7A5B10"],
    ["trend_robustness", "LOW", colors.paleRed, colors.red],
    ["representativeness_risk", "HIGH", colors.paleRed, colors.red],
    ["representativeness_risk", "MODERATE", colors.paleGold, "#7A5B10"],
    ["concentration_level", "HIGH", colors.paleRed, colors.red],
    ["concentration_level", "MODERATE", colors.paleGold, "#7A5B10"],
  ];
  for (const [field, text, fill, fontColor] of targets) {
    const index = headers.indexOf(field);
    if (index < 0 || dataLength === 0) continue;
    const column = columnName(index);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text,
      format: { fill, font: { color: fontColor, bold: true } },
    });
  }
  const flagIndex = headers.indexOf("sensitivity_flags");
  if (flagIndex >= 0 && dataLength > 0) {
    const column = columnName(flagIndex);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("notContainsText", {
      text: "NONE",
      format: { fill: colors.paleGold, font: { color: "#7A5B10" } },
    });
  }
}


function formatDataSheet(sheet, title, note, dataset, tableName) {
  /** 写入并格式化一张机器可读数据表。 */
  const { headers, data } = dataset;
  const lastColumn = columnName(headers.length - 1);
  const lastRow = data.length + 4;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${lastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A2:${lastColumn}2`).merge();
  sheet.getRange("A2").values = [[note]];
  sheet.getRange(`A4:${lastColumn}4`).values = [headers];
  if (data.length > 0) sheet.getRange(`A5:${lastColumn}${lastRow}`).values = data;
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: colors.navy,
    font: { name: "Aptos Display", size: 16, bold: true, color: colors.white },
    rowHeight: 30,
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastColumn}2`).format = {
    fill: colors.paleBlue,
    font: { name: "Aptos", size: 10, italic: true, color: colors.gray },
    wrapText: true,
    rowHeight: 30,
    verticalAlignment: "center",
  };
  sheet.getRange(`A4:${lastColumn}4`).format = {
    fill: colors.teal,
    font: { name: "Aptos", size: 10, bold: true, color: colors.white },
    wrapText: true,
    rowHeight: 44,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: colors.teal },
  };
  if (data.length > 0) {
    sheet.getRange(`A5:${lastColumn}${lastRow}`).format = {
      font: { name: "Aptos", size: 9, color: "#1F2933" },
      verticalAlignment: "center",
      borders: { insideHorizontal: { style: "thin", color: "#E5E7EB" } },
    };
    const table = sheet.tables.add(`A4:${lastColumn}${lastRow}`, true, tableName);
    table.style = "TableStyleMedium2";
  }
  sheet.freezePanes.freezeRows(4);
  sheet.getRange(`A4:${lastColumn}${Math.min(lastRow, 80)}`).format.autofitColumns();
  for (let index = 0; index < headers.length; index += 1) {
    const field = headers[index];
    const column = columnName(index);
    const range = data.length > 0 ? sheet.getRange(`${column}5:${column}${lastRow}`) : null;
    if (/count$|_order$|iterations$/.test(field) && range) range.format.numberFormat = "#,##0";
    if (/(probability|_mean$|_value$|_difference$|_delta$|_ci_low$|_ci_high$|bootstrap_sd|bootstrap_mean|smd|hhi|pearson|spearman|normalized_range|rmse)/.test(field) && range) {
      range.format.numberFormat = "0.000000";
    }
    if (/(ratio$|share$)/.test(field) && range) range.format.numberFormat = "0.0%";
    if (/(description|notes|risk_reason|bootstrap_algorithm|sensitivity_flags|main_direction_pattern)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 42;
      if (range) range.format.wrapText = true;
    } else if (/(topic_label|final_topic_label)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 28;
    } else if (/(author_id_internal|specification|interpretation_scope)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 20;
    } else {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 14;
    }
  }
  applyTextConditionalFormatting(sheet, headers, data.length);
  return { lastColumn, lastRow };
}


const sheetBounds = new Map();
for (let index = 0; index < sheetSpecs.length; index += 1) {
  const [sheetName, fileName, note] = sheetSpecs[index];
  const title = sheetName.replaceAll("_", " ");
  const bounds = formatDataSheet(
    sheets.get(sheetName),
    title,
    note,
    datasets[fileName],
    `QTS_${String(index + 1).padStart(2, "0")}_${sheetName.replaceAll("_", "")}`,
  );
  sheetBounds.set(sheetName, bounds);
}


function cellFormula(sheetName, headers, field, startRow, endRow) {
  /** 返回带单引号页签名的绝对列范围。 */
  const column = findColumn(headers, field);
  return `'${sheetName}'!$${column}$${startRow}:$${column}$${endRow}`;
}


// Overview 只展示可核查 KPI；公式直接引用底表。
overview.showGridLines = false;
overview.getRange("A1:I1").merge();
overview.getRange("A1").values = [["QTS Author-Period Topic Trend Structure"]];
overview.getRange("A2:I2").merge();
overview.getRange("A2").values = [["按作者所属时期划分的 C_K20 主题分布；不是单篇作品创作年代。CORE 为主要描述规格，STRICT/EXPANDED 为敏感性。"]];
overview.getRange("A1:I1").format = {
  fill: colors.navy,
  font: { name: "Aptos Display", size: 19, bold: true, color: colors.white },
  rowHeight: 34,
  verticalAlignment: "center",
};
overview.getRange("A2:I2").format = {
  fill: colors.paleBlue,
  font: { name: "Aptos", size: 10, italic: true, color: colors.gray },
  wrapText: true,
  rowHeight: 34,
  verticalAlignment: "center",
};

overview.getRange("A4:C4").values = [["Sample", "Poems", "Authors"]];
overview.getRange("A5:A7").values = [["STRICT"], ["CORE"], ["EXPANDED"]];
const countHeaders = datasets.period_counts.headers;
const countRows = datasets.period_counts.data.length + 4;
const countSampleRange = cellFormula("Period_Counts", countHeaders, "sample", 5, countRows);
const countPoemRange = cellFormula("Period_Counts", countHeaders, "poem_count", 5, countRows);
const countAuthorRange = cellFormula("Period_Counts", countHeaders, "author_count", 5, countRows);
for (let row = 5; row <= 7; row += 1) {
  overview.getRange(`B${row}`).formulas = [[`=SUMIF(${countSampleRange},A${row},${countPoemRange})`]];
  overview.getRange(`C${row}`).formulas = [[`=SUMIF(${countSampleRange},A${row},${countAuthorRange})`]];
}

overview.getRange("E4:H4").values = [["CORE Peak", "初唐", "盛唐", "中唐"]];
overview.getRange("E5:E6").values = [["POEM"], ["AUTHOR"]];
overview.getRange("I4").values = [["晚唐"]];
const trendHeaders = datasets.topic_trend_structure.headers;
const trendRows = datasets.topic_trend_structure.data.length + 4;
const trendSampleRange = cellFormula("Trend_Structure", trendHeaders, "sample", 5, trendRows);
const trendAggregationRange = cellFormula("Trend_Structure", trendHeaders, "aggregation", 5, trendRows);
const trendPeakRange = cellFormula("Trend_Structure", trendHeaders, "peak_period", 5, trendRows);
for (let row = 5; row <= 6; row += 1) {
  for (let column = 6; column <= 9; column += 1) {
    const header = `${columnName(column - 1)}4`;
    overview.getRange(`${columnName(column - 1)}${row}`).formulas = [[
      `=COUNTIFS(${trendSampleRange},"CORE",${trendAggregationRange},$E${row},${trendPeakRange},${header})`,
    ]];
  }
}

overview.getRange("E8:F8").values = [["Trend Robustness", "Topic Count"]];
overview.getRange("E9:E11").values = [["HIGH"], ["MODERATE"], ["LOW"]];
const robustHeaders = datasets.topic_trend_robustness.headers;
const robustRows = datasets.topic_trend_robustness.data.length + 4;
const robustRange = cellFormula("Trend_Robustness", robustHeaders, "trend_robustness", 5, robustRows);
for (let row = 9; row <= 11; row += 1) overview.getRange(`F${row}`).formulas = [[`=COUNTIF(${robustRange},E${row})`]];
overview.getRange("H8:I8").values = [["Candidate Trends", "Count"]];
overview.getRange("H9").values = [["CLEAR candidates"]];
overview.getRange("I9").formulas = [[`=COUNTA('Candidate_Trends'!$A$5:$A$${datasets.trend_candidate_topics.data.length + 4})`]];

overview.getRange("A10:C10").values = [["Method", "Value", "Interpretation"]];
overview.getRange("A11:C15").values = [
  ["Bootstrap iterations", config.bootstrap_iterations, "作者为 cluster；同一作者的全部入选诗随 cluster 进入"],
  ["Random seed", config.random_seed, "固定 seed，可逐值复现"],
  ["Analytical specifications", 6, "3 samples × 2 aggregation estimands"],
  ["Topic dimensions", 20, "完整概率向量；Mixed/Noise 未删除"],
  ["Pairwise p-values", 0, "本阶段不运行时期两两显著性检验"],
];
overview.getRange("A18:I18").merge();
overview.getRange("A18").values = [["Interpretation boundary"]];
overview.getRange("A19:I22").merge();
overview.getRange("A19").values = [[
  "所有 peak、rank、shape、bootstrap-supported direction 和 concentration 均是描述性统计结构。author_period_primary 表示作者所属时期，不得改写为诗歌创作年代；candidate trends 只供下一阶段人工审核。",
]];

for (const range of ["A4:C4", "E4:I4", "E8:F8", "H8:I8", "A10:C10", "A18:I18"]) {
  overview.getRange(range).format = {
    fill: colors.teal,
    font: { name: "Aptos", size: 10, bold: true, color: colors.white },
    rowHeight: 24,
    verticalAlignment: "center",
  };
}
overview.getRange("A5:C7").format = { fill: colors.paleTeal, font: { name: "Aptos", size: 11 }, rowHeight: 24 };
overview.getRange("E5:I6").format = { fill: colors.paleBlue, font: { name: "Aptos", size: 11 }, rowHeight: 24 };
overview.getRange("E9:F11").format = { fill: colors.paleGold, font: { name: "Aptos", size: 11 }, rowHeight: 24 };
overview.getRange("H9:I9").format = { fill: colors.paleGreen, font: { name: "Aptos", size: 11 }, rowHeight: 24 };
overview.getRange("A11:C15").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 26 };
overview.getRange("A19:I22").format = { fill: colors.paleGold, font: { name: "Aptos", size: 11, color: "#5E4610" }, wrapText: true, verticalAlignment: "center" };
overview.getRange("B5:C7").format.numberFormat = "#,##0";
overview.getRange("F5:I6").format.numberFormat = "#,##0";
overview.getRange("F9:F11").format.numberFormat = "#,##0";
overview.getRange("I9").format.numberFormat = "#,##0";
overview.getRange("A:A").format.columnWidth = 24;
overview.getRange("B:B").format.columnWidth = 15;
overview.getRange("C:C").format.columnWidth = 42;
overview.getRange("D:D").format.columnWidth = 3;
overview.getRange("E:E").format.columnWidth = 22;
overview.getRange("F:I").format.columnWidth = 13;
overview.freezePanes.freezeRows(2);

const overviewInspect = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:I22",
  include: "values,formulas",
  tableMaxRows: 24,
  tableMaxCols: 10,
  maxChars: 12000,
});
const errorInspect = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "最终公式错误扫描",
  maxChars: 8000,
});

const previewFiles = [];
for (const sheetName of ["Overview", ...sheetSpecs.map(([name]) => name)]) {
  const bounds = sheetName === "Overview" ? { lastColumn: "I", lastRow: 22 } : sheetBounds.get(sheetName);
  const endRow = Math.min(bounds.lastRow, 24);
  const blob = await workbook.render({
    sheetName,
    range: `A1:${bounds.lastColumn}${endRow}`,
    scale: 1,
    format: "png",
  });
  const previewPath = path.join(previewRoot, `${sheetName}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
  previewFiles.push(previewPath);
}

const workbookPath = path.join(outputRoot, "QTS_Author_Period_Trend_Structure.xlsx");
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(workbookPath);

const workbookBytes = await fs.readFile(workbookPath);
const workbookHash = crypto.createHash("sha256").update(workbookBytes).digest("hex");
const verification = {
  workbook_path: workbookPath,
  workbook_sha256: workbookHash,
  workbook_bytes: workbookBytes.length,
  sheet_count: 12,
  rendered_sheet_count: previewFiles.length,
  rendered_previews: previewFiles,
  overview_inspect: overviewInspect.ndjson,
  formula_error_scan: errorInspect.ndjson,
};
await fs.writeFile(path.join(outputRoot, "workbook_verification.json"), JSON.stringify(verification, null, 2), "utf8");

const manifestPath = path.join(outputRoot, "analysis_manifest.json");
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
manifest.outputs["QTS_Author_Period_Trend_Structure.xlsx"] = {
  sha256: workbookHash,
  bytes: workbookBytes.length,
};
manifest.outputs["workbook_verification.json"] = {
  sha256: crypto.createHash("sha256").update(await fs.readFile(path.join(outputRoot, "workbook_verification.json"))).digest("hex"),
  bytes: (await fs.stat(path.join(outputRoot, "workbook_verification.json"))).size,
};
await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

console.log(JSON.stringify({
  workbookPath,
  workbookHash,
  sheetCount: 12,
  renderedSheetCount: previewFiles.length,
  overviewInspect: overviewInspect.ndjson,
  formulaErrorScan: errorInspect.ndjson,
}, null, 2));
