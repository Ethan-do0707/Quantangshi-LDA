/** 使用 artifact-tool 生成作者时期效应量与统计验证主工作簿。 */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const scriptPath = fileURLToPath(import.meta.url);
const root = path.resolve(process.env.QTS_REPO_ROOT ?? path.resolve(path.dirname(scriptPath), ".."));
const configPath = path.resolve(root, process.argv[2] ?? "config/effect_validation_config.yaml");
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
  /** 读取分析 CSV，并返回表头和类型化数据。 */
  const text = await fs.readFile(path.join(dataRoot, `${name}.csv`), "utf8");
  const rows = parseCsv(text);
  return {
    headers: rows[0],
    data: rows.slice(1).filter((row) => row.length > 1 || row[0] !== "").map((row) => row.map(coerce)),
  };
}


function project(dataset, fields, predicate = () => true) {
  /** 从机器可读底表选择字段并可选过滤记录。 */
  const indexes = fields.map((field) => {
    const index = dataset.headers.indexOf(field);
    if (index < 0) throw new Error(`缺少字段：${field}`);
    return index;
  });
  const records = dataset.data.filter((row) => predicate(Object.fromEntries(dataset.headers.map((header, index) => [header, row[index]]))));
  return { headers: fields, data: records.map((row) => indexes.map((index) => row[index])) };
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


function fieldColumn(headers, field) {
  /** 返回字段的一基 Excel 列名。 */
  const index = headers.indexOf(field);
  if (index < 0) throw new Error(`缺少字段：${field}`);
  return columnName(index);
}


const source = {
  omnibus: await readCsv("clear_topic_period_omnibus"),
  primary: await readCsv("clear_topic_primary_contrasts"),
  pairwise: await readCsv("clear_topic_all_pairwise"),
  effects: await readCsv("all_topic_effect_sizes"),
  sensitivity: await readCsv("clear_topic_effect_sensitivity_matrix"),
  clr: await readCsv("raw_vs_clr_direction_agreement"),
  clearSummary: await readCsv("clear_topics_validation_summary"),
  queue: await readCsv("author_influence_check_queue"),
  dictionary: await readCsv("data_dictionary"),
};

const bootstrapFields = [
  "topic_id", "topic_label", "aggregation", "contrast", "raw_delta", "bootstrap_ci_low",
  "bootstrap_ci_high", "bootstrap_se", "bootstrap_direction", "bootstrap_iterations", "bootstrap_unit",
];
const poemAuthorFields = [
  "topic_id", "topic_label", "contrast", "raw_effect_poem", "poem_ci_low", "poem_ci_high",
  "raw_effect_author", "author_ci_low", "author_ci_high", "descriptive_clustered_smd", "hedges_g",
  "cliff_delta", "aggregation_agreement", "aggregation_conflict", "validation_status",
];
const validationFields = [
  "topic_id", "topic_label", "contrast", "direction", "bootstrap_support_poem", "bootstrap_support_author",
  "q_poem", "q_author", "aggregation_agreement", "sample_robustness", "sample_sign_reversal",
  "representativeness_risk", "clr_agreement", "validation_status", "validation_reason",
];

const sheetSpecs = [
  ["Omnibus", source.omnibus, "9个 CLEAR Topics 的 CORE_POEM 与 CORE_AUTHOR omnibus Wald 检验；两种 aggregation 分属不同 BH-FDR family。"],
  ["Primary_Contrasts", source.primary, "E-H、H-M、M-L、E-L；raw Δ 定义为 B−A，同时报告作者 cluster bootstrap CI、标准化效应与稳健推断。"],
  ["All_Pairwise", source.pairwise, "CLEAR Topics 的六个时期对比；E-M、H-L 为 secondary，未纳入 primary contrast FDR family。"],
  ["Effect_Sizes", source.effects, "20 Topics × 六规格 × 六对比的完整描述性附录；MIXED/NOISE 未删除。"],
  ["Bootstrap_CI", project(source.effects, bootstrapFields, (record) => record.sample === "CORE" && record.topic_type === "CLEAR" && record.contrast_role === "PRIMARY"), "CORE CLEAR planned contrasts 的作者 cluster bootstrap 区间；抽样单位为 author。"],
  ["Poem_vs_Author", project(source.primary, poemAuthorFields), "并列诗等权与作者等权 estimand；差异不表示其中一个估计必然更正确。"],
  ["Sample_Sensitivity", source.sensitivity, "STRICT/CORE/EXPANDED × POEM/AUTHOR 六规格的效应方向、区间、符号反转与幅度敏感性。"],
  ["CLR_Sensitivity", source.clr, "raw probability 为主；CLR 只在 CORE CLEAR Topics 临时执行，未覆盖原始概率。"],
  ["CLEAR_Topics", source.clearSummary, "9个 CLEAR Topics 的四类 planned contrast 状态、聚合冲突、样本敏感性与 CLR 摘要。"],
  ["Validation_Status", project(source.primary, validationFields), "STRONG/SUPPORTED/TENTATIVE/UNSUPPORTED/CONFLICTING 由多维证据共同决定。"],
  ["Author_Check_Flags", source.queue, "下一阶段 Author Influence / Leave-One-Author-Out 队列；priority 数字越小越优先。"],
  ["Data_Dictionary", source.dictionary, "正式 CSV 的字段级定义与数据类型。"],
];

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
const sheets = new Map([["Overview", overview]]);
for (const [sheetName] of sheetSpecs) sheets.set(sheetName, workbook.worksheets.add(sheetName));

const colors = {
  navy: "#16324F",
  teal: "#2F6F73",
  paleTeal: "#DCEDEF",
  paleBlue: "#E9F1F7",
  gold: "#C38A2E",
  paleGold: "#F8EBCB",
  red: "#A33B3B",
  paleRed: "#F7DDDA",
  green: "#2F6B45",
  paleGreen: "#DCEBDD",
  gray: "#5B6670",
  paleGray: "#EEF1F3",
  white: "#FFFFFF",
};


function applyConditionalFormatting(sheet, headers, dataLength) {
  /** 对状态、风险、一致性与 FDR 字段设置动态条件格式。 */
  const rules = [
    ["validation_status", "STRONG", colors.paleGreen, colors.green],
    ["validation_status", "SUPPORTED", colors.paleTeal, colors.teal],
    ["validation_status", "TENTATIVE", colors.paleGold, "#78580D"],
    ["validation_status", "UNSUPPORTED", colors.paleGray, colors.gray],
    ["validation_status", "CONFLICTING", colors.paleRed, colors.red],
    ["representativeness_risk", "MODERATE", colors.paleGold, "#78580D"],
    ["representativeness_risk", "HIGH", colors.paleRed, colors.red],
    ["aggregation_agreement", "HIGH", colors.paleGreen, colors.green],
    ["aggregation_agreement", "LOW", colors.paleRed, colors.red],
  ];
  for (const [field, text, fill, fontColor] of rules) {
    const index = headers.indexOf(field);
    if (index < 0 || dataLength === 0) continue;
    const column = columnName(index);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text,
      format: { fill, font: { color: fontColor, bold: true } },
    });
  }
  for (const field of ["fdr_pass", "direction_agreement", "clr_agreement"]) {
    const index = headers.indexOf(field);
    if (index < 0 || dataLength === 0) continue;
    const column = columnName(index);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text: "true",
      format: { fill: colors.paleGreen, font: { color: colors.green, bold: true } },
    });
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text: "false",
      format: { fill: colors.paleRed, font: { color: colors.red, bold: true } },
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
    rowHeight: 32,
    verticalAlignment: "center",
  };
  sheet.getRange(`A4:${lastColumn}4`).format = {
    fill: colors.teal,
    font: { name: "Aptos", size: 10, bold: true, color: colors.white },
    wrapText: true,
    rowHeight: 46,
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
  for (let index = 0; index < headers.length; index += 1) {
    const field = headers[index];
    const column = columnName(index);
    const range = data.length > 0 ? sheet.getRange(`${column}5:${column}${lastRow}`) : null;
    if (/(^n_|count$|_count$|_order$|iterations$|priority$|df_num$|df_denom$)/.test(field) && range) range.format.numberFormat = "#,##0";
    if (/(raw_|_delta$|_effect|_mean|_se$|_ci_|hedges|cliff|wald|model_t|p_value|q_value|^p_|^q_|relative_change|smd)/.test(field) && range) range.format.numberFormat = "0.000000";
    if (/(share$)/.test(field) && range) range.format.numberFormat = "0.0%";
    if (/(reason|description|algorithm|scope|specifications)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 42;
      if (range) range.format.wrapText = true;
    } else if (/topic_label/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 28;
    } else if (/(topic_id|specification|aggregation|contrast|direction|status|magnitude|risk|family)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 19;
    } else {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 14;
    }
  }
  applyConditionalFormatting(sheet, headers, data.length);
  return { lastColumn, lastRow, headers };
}


const sheetBounds = new Map();
for (let index = 0; index < sheetSpecs.length; index += 1) {
  const [sheetName, dataset, note] = sheetSpecs[index];
  sheetBounds.set(
    sheetName,
    formatDataSheet(
      sheets.get(sheetName),
      sheetName.replaceAll("_", " "),
      note,
      dataset,
      `QTS_EV_${String(index + 1).padStart(2, "0")}_${sheetName.replaceAll("_", "")}`,
    ),
  );
}


function absoluteColumnRange(sheetName, dataset, field) {
  /** 返回数据表第5行起的绝对引用范围。 */
  const column = fieldColumn(dataset.headers, field);
  return `'${sheetName}'!$${column}$5:$${column}$${dataset.data.length + 4}`;
}


// Overview 的 KPI 由底表公式驱动，便于复核和后续刷新。
overview.showGridLines = false;
overview.getRange("A1:L1").merge();
overview.getRange("A1").values = [["QTS Author-Period Effect Size & Statistical Validation"]];
overview.getRange("A2:L2").merge();
overview.getRange("A2").values = [["CORE 为正式推断；STRICT/EXPANDED 为敏感性。所有差值为 period B − period A；author period 不是诗歌创作年代。"]];
overview.getRange("A1:L1").format = {
  fill: colors.navy,
  font: { name: "Aptos Display", size: 19, bold: true, color: colors.white },
  rowHeight: 36,
  verticalAlignment: "center",
};
overview.getRange("A2:L2").format = {
  fill: colors.paleBlue,
  font: { name: "Aptos", size: 10, italic: true, color: colors.gray },
  wrapText: true,
  rowHeight: 36,
  verticalAlignment: "center",
};

overview.getRange("A4:C4").values = [["Omnibus family", "FDR pass", "Total CLEAR"]];
overview.getRange("A5:A6").values = [["CORE_POEM"], ["CORE_AUTHOR"]];
const omnibusAggregation = absoluteColumnRange("Omnibus", source.omnibus, "aggregation");
const omnibusQ = absoluteColumnRange("Omnibus", source.omnibus, "q_value");
for (let row = 5; row <= 6; row += 1) {
  const aggregation = row === 5 ? "POEM" : "AUTHOR";
  overview.getRange(`B${row}`).formulas = [[`=COUNTIFS(${omnibusAggregation},"${aggregation}",${omnibusQ},"<=0.05")`]];
  overview.getRange(`C${row}`).formulas = [[`=COUNTIF(${omnibusAggregation},"${aggregation}")`]];
}

overview.getRange("E4:J4").values = [["Contrast", "STRONG", "SUPPORTED", "TENTATIVE", "UNSUPPORTED", "CONFLICTING"]];
overview.getRange("E5:E8").values = [["E-H"], ["H-M"], ["M-L"], ["E-L"]];
const validationContrast = absoluteColumnRange("Validation_Status", sheetSpecs.find(([name]) => name === "Validation_Status")[1], "contrast");
const validationStatus = absoluteColumnRange("Validation_Status", sheetSpecs.find(([name]) => name === "Validation_Status")[1], "validation_status");
for (let row = 5; row <= 8; row += 1) {
  for (let column = 6; column <= 10; column += 1) {
    overview.getRange(`${columnName(column - 1)}${row}`).formulas = [[
      `=COUNTIFS(${validationContrast},$E${row},${validationStatus},${columnName(column - 1)}$4)`,
    ]];
  }
}

overview.getRange("A10:C10").values = [["Review queue", "Count", "Meaning"]];
overview.getRange("A11:A12").values = [["Supported candidates"], ["Author-check rows"]];
overview.getRange("B11").formulas = [[`=COUNTIF(${validationStatus},"STRONG")+COUNTIF(${validationStatus},"SUPPORTED")`]];
overview.getRange("B12").formulas = [[`=COUNTA('Author_Check_Flags'!$A$5:$A$${source.queue.data.length + 4})`]];
overview.getRange("C11:C12").values = [["STATISTICALLY_SUPPORTED_CANDIDATE；仍非文学史确认"], ["下一阶段 Author Influence / LOAO 检查项"]];

overview.getRange("A15:C15").values = [["Method", "Value", "Audit note"]];
overview.getRange("A16:C22").values = [
  ["Bootstrap iterations", config.bootstrap_iterations, "作者为 cluster；固定时期内重抽作者"],
  ["Random seed", config.random_seed, "全部随机过程可复现"],
  ["Formal Topics", 9, "仅 taxonomy 中 CLEAR"],
  ["Formal estimands", 2, "CORE_POEM CR1；CORE_AUTHOR HC3"],
  ["Primary contrasts", 4, "E-H、H-M、M-L、E-L"],
  ["Sensitivity specifications", 6, "3 samples × 2 aggregations；不新增 p-value family"],
  ["CLR true zeros", 0, "未使用 pseudocount；CLR 只作临时敏感性"],
];
overview.getRange("E11:L11").merge();
overview.getRange("E11").values = [["Interpretation boundary"]];
overview.getRange("E12:L17").merge();
overview.getRange("E12").values = [["效应状态综合 raw Δ、作者 cluster bootstrap、Hedges' g、Cliff's delta、BH-FDR、POEM/AUTHOR 一致性、六规格敏感性、CLR 与 representativeness risk。统计支持不等于文学史重要性，也不构成因果推断。"]];

for (const range of ["A4:C4", "E4:J4", "A10:C10", "A15:C15", "E11:L11"]) {
  overview.getRange(range).format = {
    fill: colors.teal,
    font: { name: "Aptos", size: 10, bold: true, color: colors.white },
    rowHeight: 25,
    verticalAlignment: "center",
  };
}
overview.getRange("A5:C6").format = { fill: colors.paleTeal, font: { name: "Aptos", size: 11 }, rowHeight: 25 };
overview.getRange("E5:J8").format = { font: { name: "Aptos", size: 10 }, rowHeight: 24 };
overview.getRange("A11:C12").format = { fill: colors.paleGreen, font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 30 };
overview.getRange("A16:C22").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 27 };
overview.getRange("E12:L17").format = { fill: colors.paleGold, font: { name: "Aptos", size: 11, color: "#5B430B" }, wrapText: true, verticalAlignment: "center" };
overview.getRange("B5:C6").format.numberFormat = "#,##0";
overview.getRange("F5:J8").format.numberFormat = "#,##0";
overview.getRange("B11:B12").format.numberFormat = "#,##0";
overview.getRange("A:A").format.columnWidth = 25;
overview.getRange("B:B").format.columnWidth = 14;
overview.getRange("C:C").format.columnWidth = 44;
overview.getRange("D:D").format.columnWidth = 3;
overview.getRange("E:E").format.columnWidth = 16;
overview.getRange("F:J").format.columnWidth = 14;
overview.getRange("K:L").format.columnWidth = 14;
overview.freezePanes.freezeRows(2);

const chart = overview.charts.add("bar", overview.getRange("E4:J8"));
chart.title = "Validation status counts by planned contrast";
chart.hasLegend = true;
chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
chart.yAxis = { numberFormatCode: "0", min: 0, max: 9 };
chart.setPosition("E20", "L36");

const formulaInspect = await workbook.inspect({
  kind: "formula",
  sheetId: "Overview",
  range: "A1:L36",
  maxChars: 12000,
  options: { maxResults: 100 },
});
const overviewInspect = await workbook.inspect({
  kind: "region",
  sheetId: "Overview",
  range: "A1:L36",
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
  const bounds = sheetName === "Overview" ? { lastColumn: "L", lastRow: 36, headers: [] } : sheetBounds.get(sheetName);
  const previewLastColumn = sheetName === "Overview" ? "L" : columnName(Math.min(bounds.headers.length, 14) - 1);
  const previewEndRow = Math.min(bounds.lastRow, 24);
  const blob = await workbook.render({
    sheetName,
    range: `A1:${previewLastColumn}${previewEndRow}`,
    scale: 1,
    format: "png",
  });
  const previewPath = path.join(previewRoot, `${sheetName}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
  previewFiles.push(previewPath);
}

const workbookPath = path.join(outputRoot, "QTS_Author_Period_Effect_Size_Statistical_Validation.xlsx");
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(workbookPath);
const workbookBytes = await fs.readFile(workbookPath);
const workbookHash = crypto.createHash("sha256").update(workbookBytes).digest("hex");
const verification = {
  workbook_path: workbookPath,
  workbook_sha256: workbookHash,
  workbook_bytes: workbookBytes.length,
  sheet_count: 13,
  rendered_sheet_count: previewFiles.length,
  rendered_previews: previewFiles,
  overview_inspect: overviewInspect.ndjson,
  formula_inspect: formulaInspect.ndjson,
  formula_error_scan: errorInspect.ndjson,
};
const verificationPath = path.join(outputRoot, "workbook_verification.json");
await fs.writeFile(verificationPath, JSON.stringify(verification, null, 2), "utf8");

const manifestPath = path.join(outputRoot, "analysis_manifest.json");
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
manifest.outputs["QTS_Author_Period_Effect_Size_Statistical_Validation.xlsx"] = {
  sha256: workbookHash,
  bytes: workbookBytes.length,
};
manifest.outputs["workbook_verification.json"] = {
  sha256: crypto.createHash("sha256").update(await fs.readFile(verificationPath)).digest("hex"),
  bytes: (await fs.stat(verificationPath)).size,
};
await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

console.log(JSON.stringify({
  workbookPath,
  workbookHash,
  sheetCount: 13,
  renderedSheetCount: previewFiles.length,
  formulaInspect: formulaInspect.ndjson,
  overviewInspect: overviewInspect.ndjson,
  formulaErrorScan: errorInspect.ndjson,
}, null, 2));
