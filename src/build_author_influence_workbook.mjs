/** 使用 artifact-tool 生成作者影响与 LOAO 主工作簿。 */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptPath = fileURLToPath(import.meta.url);
const root = path.resolve(process.env.QTS_REPO_ROOT ?? path.resolve(path.dirname(scriptPath), ".."));
const configPath = path.resolve(root, process.argv[2] ?? "config/author_influence_config.yaml");
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
  /** 把 CSV 标量恢复为布尔值、数值或文本。 */
  if (value === "") return null;
  if (value === "True" || value === "true") return true;
  if (value === "False" || value === "false") return false;
  if (/^-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

async function readCsv(name) {
  /** 读取分析 CSV，并返回表头与类型化数据。 */
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
  const records = dataset.data.filter((row) => {
    const record = Object.fromEntries(dataset.headers.map((header, index) => [header, row[index]]));
    return predicate(record);
  });
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
  /** 返回指定字段的一基 Excel 列名。 */
  const index = headers.indexOf(field);
  if (index < 0) throw new Error(`缺少字段：${field}`);
  return columnName(index);
}

const source = {
  influence: await readCsv("clear_topic_author_influence_summary"),
  loao: await readCsv("loao_effects"),
  topAuthors: await readCsv("top_influential_authors"),
  exclusions: await readCsv("top_contributor_exclusion_effects"),
  bootstrap: await readCsv("top_exclusion_bootstrap_ci"),
  conflict: await readCsv("aggregation_conflict_decomposition"),
  candidates: await readCsv("author_validated_candidate_trends"),
  dictionary: await readCsv("data_dictionary"),
};

const influenceFields = [
  "topic_id", "topic_label", "contrast", "original_poem_delta", "original_author_delta",
  "poem_loao_max_abs_change", "author_loao_max_abs_change", "poem_sign_flip_count",
  "author_sign_flip_count", "poem_max_attenuation", "author_max_attenuation", "top1_share",
  "top3_share", "top5_share", "top1_exclusion_delta_poem", "top3_exclusion_delta_poem",
  "top5_exclusion_delta_poem", "top1_exclusion_delta_author", "top3_exclusion_delta_author",
  "top5_exclusion_delta_author", "author_robustness_status", "sample_influence_robust", "robustness_reason",
  "primary_exclusion_scheme", "per_period_author_robustness_status", "per_period_robustness_reason",
  "exclusion_scheme_status_agreement",
];

const exclusionFields = [
  "sample", "topic_id", "topic_label", "contrast", "period_a", "period_b", "aggregation",
  "exclusion_scheme", "N", "excluded_author_count", "excluded_authors", "excluded_poem_count",
  "excluded_topic_mass", "original_delta", "new_delta", "absolute_change", "attenuation_ratio",
  "sign_flip", "sign_flip_concern",
];

const sheetSpecs = [
  ["Influence_Summary", project(source.influence, influenceFields), "9个 CLEAR Topics × 4 planned contrasts；GLOBAL_TOP 主分类，PER_PERIOD_TOP 敏感性分列。未加方案前缀的 Top 摘要均为 GLOBAL_TOP。"],
  ["LOAO_All", source.loao, "CORE 逐作者剔除全表；removed_author_id 是删除单位，删除时移除该作者在当前样本中的全部诗。"],
  ["Top_Influential_Authors", source.topAuthors, "每个 Topic×contrast×aggregation 按 absolute effect change 排列的前10位作者；只表示数值影响。"],
  ["Top_Exclusion", project(source.exclusions, exclusionFields), "GLOBAL_TOP 合并两时期排序；PER_PERIOD_TOP 在两侧分别排序；贡献者按 Topic mass 而非诗数确定。完整 author_id 列保存在对应 CSV。"],
  ["Top_Exclusion_Bootstrap", source.bootstrap, "仅13组既有统计候选的 Top1/3/5 exclusion；固定种子作者 cluster bootstrap，不扩张 p-value 搜索。"],
  ["Aggregation_Conflict", source.conflict, "T07/T11/T16/T19 及全部既有方向冲突的逐作者贡献分解；POEM 与 AUTHOR 回答不同加权问题。"],
  ["T03", project(source.influence, influenceFields, (record) => record.topic_id === "topic_03"), "T03 四组 planned contrasts；重点核查 E-H、M-L、E-L。"],
  ["T06", project(source.influence, influenceFields, (record) => record.topic_id === "topic_06"), "T06 四组 planned contrasts；重点核查 E-L。"],
  ["T08", project(source.influence, influenceFields, (record) => record.topic_id === "topic_08"), "T08 四组 planned contrasts；重点核查 E-L。"],
  ["T16", project(source.influence, influenceFields, (record) => record.topic_id === "topic_16"), "T16 四组 planned contrasts；保留 aggregation conflict 与 H-M/E-L 作者影响诊断。"],
  ["Final_Assessment", source.candidates, "13组既有统计候选；GLOBAL_TOP 为主分类，PER_PERIOD_TOP 敏感性分列。9组保留候选均通过两方案诊断；不表示删作者后的 CI 均不跨零。"],
  ["Data_Dictionary", source.dictionary, "正式 CSV 的字段级定义与数据类型。"],
];

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
const sheets = new Map([["Overview", overview]]);
for (const [sheetName] of sheetSpecs) sheets.set(sheetName, workbook.worksheets.add(sheetName));

const colors = {
  navy: "#17324D", teal: "#2B6F73", paleTeal: "#DDEEEF", blue: "#2F6FDB",
  paleBlue: "#E8F0FB", green: "#2D6A4F", paleGreen: "#DCECDD", gold: "#B7791F",
  paleGold: "#F8EBCB", red: "#A33B3B", paleRed: "#F6DDDA", gray: "#5D6772",
  paleGray: "#EEF1F3", white: "#FFFFFF",
};

function applyConditionalFormatting(sheet, headers, dataLength) {
  /** 为状态与方向反转字段设置动态条件格式。 */
  const rules = [
    ["author_robustness_status", "AUTHOR_ROBUST", colors.paleGreen, colors.green],
    ["author_robustness_status", "CONCENTRATED_BUT_ROBUST", colors.paleBlue, colors.blue],
    ["author_robustness_status", "AUTHOR_SENSITIVE", colors.paleRed, colors.red],
    ["author_robustness_status", "ESTIMAND_DEPENDENT", colors.paleGold, colors.gold],
    ["author_robustness_status", "INCONCLUSIVE", colors.paleGray, colors.gray],
    ["final_author_influence_assessment", "UPGRADE_KEEP", colors.paleGreen, colors.green],
    ["final_author_influence_assessment", "KEEP_WITH_CAUTION", colors.paleBlue, colors.blue],
    ["final_author_influence_assessment", "DOWNGRADE_AUTHOR_SENSITIVE", colors.paleRed, colors.red],
    ["final_author_influence_assessment", "ESTIMAND_DEPENDENT", colors.paleGold, colors.gold],
  ];
  for (const [field, text, fill, fontColor] of rules) {
    const index = headers.indexOf(field);
    if (index < 0 || dataLength === 0) continue;
    const column = columnName(index);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text, format: { fill, font: { color: fontColor, bold: true } },
    });
  }
  for (const field of ["sign_flip", "sign_flip_concern", "finding_level_bd_eligible", "sample_influence_robust"]) {
    const index = headers.indexOf(field);
    if (index < 0 || dataLength === 0) continue;
    const column = columnName(index);
    sheet.getRange(`${column}5:${column}${dataLength + 4}`).conditionalFormats.add("containsText", {
      text: "true", format: { fill: field.includes("flip") ? colors.paleRed : colors.paleGreen, font: { bold: true } },
    });
  }
}

function formatDataSheet(sheet, title, note, dataset, tableName) {
  /** 写入并格式化机器可读数据表。 */
  const { headers, data } = dataset;
  const lastColumn = columnName(headers.length - 1);
  const titleLastColumn = columnName(Math.min(headers.length, 14) - 1);
  const lastRow = data.length + 4;
  sheet.showGridLines = false;
  sheet.getRange(`A1:${titleLastColumn}1`).merge();
  sheet.getRange("A1").values = [[title]];
  sheet.getRange(`A2:${titleLastColumn}2`).merge();
  sheet.getRange("A2").values = [[note]];
  sheet.getRange(`A4:${lastColumn}4`).values = [headers];
  if (data.length > 0) sheet.getRange(`A5:${lastColumn}${lastRow}`).values = data;
  sheet.getRange(`A1:${titleLastColumn}1`).format = { fill: colors.navy, font: { name: "Aptos Display", size: 16, bold: true, color: colors.white }, rowHeight: 30, verticalAlignment: "center" };
  sheet.getRange(`A2:${titleLastColumn}2`).format = { fill: colors.paleBlue, font: { name: "Aptos", size: 10, italic: true, color: colors.gray }, wrapText: true, rowHeight: 34, verticalAlignment: "center" };
  sheet.getRange(`A4:${lastColumn}4`).format = { fill: colors.teal, font: { name: "Aptos", size: 10, bold: true, color: colors.white }, wrapText: true, rowHeight: 48, verticalAlignment: "center", borders: { preset: "outside", style: "thin", color: colors.teal } };
  if (data.length > 0) {
    sheet.getRange(`A5:${lastColumn}${lastRow}`).format = { font: { name: "Aptos", size: 9, color: "#1F2933" }, verticalAlignment: "center", borders: { insideHorizontal: { style: "thin", color: "#E5E7EB" } } };
    const table = sheet.tables.add(`A4:${lastColumn}${lastRow}`, true, tableName);
    table.style = "TableStyleMedium2";
  }
  sheet.freezePanes.freezeRows(4);
  for (let index = 0; index < headers.length; index += 1) {
    const field = headers[index];
    const column = columnName(index);
    const range = data.length > 0 ? sheet.getRange(`${column}5:${column}${lastRow}`) : null;
    if (/(^N$|rank$|_rank$|count$|_count$|iterations$|seed$)/.test(field) && range) range.format.numberFormat = "#,##0";
    if (/(delta|change|attenuation|probability|topic_mass$|hhi|gini|weight|contribution|ci_low|ci_high|score)/.test(field) && range) range.format.numberFormat = "0.000000";
    if (/(share$|_share$)/.test(field) && range) range.format.numberFormat = "0.0%";
    if (field === "file" || field === "field") {
      sheet.getRange(`${column}:${column}`).format.columnWidth = field === "file" ? 54 : 48;
    } else if (/^(author_id_internal|removed_author_id)$/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 28;
    } else if (/_exclusion_result$/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 56;
      if (range) range.format.wrapText = true;
    } else if (/(robustness_status|final_author_influence_assessment)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 36;
    } else if (/(reason|description|note|interpretation)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 42;
      if (range) range.format.wrapText = true;
    } else if (/(excluded_authors|excluded_author_ids|authors$)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 48;
      if (range) range.format.wrapText = true;
    } else if (/topic_label/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 28;
    } else if (/(author$|removed_author$)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 18;
    } else if (/(topic_id|aggregation|contrast|period|status|assessment|scheme|direction)/.test(field)) {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 20;
    } else {
      sheet.getRange(`${column}:${column}`).format.columnWidth = 14;
    }
  }
  if (headers.some((field) => /_exclusion_result$/.test(field)) && data.length > 0) {
    sheet.getRange(`A5:${lastColumn}${lastRow}`).format.rowHeight = 48;
  }
  const excludedNamesIndex = headers.indexOf("excluded_authors");
  if (excludedNamesIndex >= 0) {
    for (let index = 0; index < data.length; index += 1) {
      const visibleWidth = [...String(data[index][excludedNamesIndex] ?? "")].reduce((sum, character) => sum + (character.charCodeAt(0) > 255 ? 2 : 1), 0);
      sheet.getRange(`A${index + 5}:${lastColumn}${index + 5}`).format.rowHeight = Math.max(20, Math.min(96, Math.ceil(visibleWidth / 44) * 14 + 6));
    }
  }
  applyConditionalFormatting(sheet, headers, data.length);
  return { lastColumn, lastRow, headers };
}

const sheetBounds = new Map();
for (let index = 0; index < sheetSpecs.length; index += 1) {
  const [sheetName, dataset, note] = sheetSpecs[index];
  sheetBounds.set(sheetName, formatDataSheet(sheets.get(sheetName), sheetName.replaceAll("_", " "), note, dataset, `QTS_AI_${String(index + 1).padStart(2, "0")}_${sheetName.replaceAll("_", "")}`));
}

function absoluteColumnRange(sheetName, dataset, field) {
  /** 返回数据表第5行起的绝对引用范围。 */
  const column = fieldColumn(dataset.headers, field);
  return `'${sheetName}'!$${column}$5:$${column}$${dataset.data.length + 4}`;
}

overview.showGridLines = false;
overview.getRange("A1:L1").merge();
overview.getRange("A1").values = [["QTS Author Influence / Leave-One-Author-Out"]];
overview.getRange("A2:L2").merge();
overview.getRange("A2").values = [["CORE 正式 LOAO；以 author_id_internal 删除整簇。主分类使用 GLOBAL_TOP Top1/3/5，PER_PERIOD_TOP 单独作敏感性诊断；高 influence 不表示文学史地位或因果作用。"]];
overview.getRange("A1:L1").format = { fill: colors.navy, font: { name: "Aptos Display", size: 19, bold: true, color: colors.white }, rowHeight: 36, verticalAlignment: "center" };
overview.getRange("A2:L2").format = { fill: colors.paleBlue, font: { name: "Aptos", size: 10, italic: true, color: colors.gray }, wrapText: true, rowHeight: 38, verticalAlignment: "center" };

overview.getRange("A4:C4").values = [["Candidate status", "Count", "Audit meaning"]];
overview.getRange("A5:A9").values = [["AUTHOR_ROBUST"], ["CONCENTRATED_BUT_ROBUST"], ["AUTHOR_SENSITIVE"], ["ESTIMAND_DEPENDENT"], ["INCONCLUSIVE"]];
overview.getRange("C5:C9").values = [
  ["方向与幅度对单作者及 Top contributor 排除稳健"],
  ["贡献集中，但不依赖单一作者且 Top1/3/5 后仍保留"],
  ["单作者或多个 Top contributor 排除造成强衰减/反向"],
  ["POEM 与 AUTHOR estimand 保留系统差异"],
  ["现有信息不足以稳定分类"],
];
const candidateStatus = absoluteColumnRange("Final_Assessment", source.candidates, "author_robustness_status");
for (let row = 5; row <= 9; row += 1) overview.getRange(`B${row}`).formulas = [[`=COUNTIF(${candidateStatus},A${row})`]];

overview.getRange("E4:G4").values = [["Final assessment", "Count", "Next-stage rule"]];
overview.getRange("E5:E9").values = [["UPGRADE_KEEP"], ["KEEP_WITH_CAUTION"], ["DOWNGRADE_AUTHOR_SENSITIVE"], ["ESTIMAND_DEPENDENT"], ["INSUFFICIENT"]];
overview.getRange("G5:G9").values = [
  ["可进入 B/D finding-level robustness"],
  ["可进入，但必须保留集中度警告"],
  ["本阶段降级，不作下一阶段主候选"],
  ["如后续复核，须分 estimand 报告"],
  ["不进入下一阶段"],
];
const finalAssessment = absoluteColumnRange("Final_Assessment", source.candidates, "final_author_influence_assessment");
for (let row = 5; row <= 9; row += 1) overview.getRange(`F${row}`).formulas = [[`=COUNTIF(${finalAssessment},E${row})`]];

overview.getRange("A12:C12").values = [["Method", "Value", "Audit note"]];
overview.getRange("A13:C20").values = [
  ["CORE authors", 207, "作者级等权表中的唯一 author_id_internal"],
  ["CORE poems", 10841, "正式 LOAO 样本"],
  ["CLEAR Topics", 9, "taxonomy 冻结"],
  ["Primary contrasts", 4, "E-H、H-M、M-L、E-L"],
  ["LOAO rows", source.loao.data.length, "9×4×2 estimands×涉及时期作者"],
  ["Top exclusion schemes", 2, "GLOBAL_TOP 与 PER_PERIOD_TOP"],
  ["Bootstrap iterations", config.bootstrap_iterations, "只对13组候选 Top1/3/5 exclusion"],
  ["Random seed", config.random_seed, "固定且可复现"],
];
overview.getRange("E12:L12").merge();
overview.getRange("E12").values = [["Interpretation boundary"]];
overview.getRange("E13:L18").merge();
overview.getRange("E13").values = [["author period ≠ poem date；author influence ≠ literary importance。高影响可能来自创作量、现存作品量、《全唐诗》收录量或 Topic probability，不能拆解为单一原因。工作簿只保存 influence / robustness diagnostic；未运行 B_K20、D_K20，也未生成文学史结论。"]];

for (const range of ["A4:C4", "E4:G4", "A12:C12", "E12:L12"]) {
  overview.getRange(range).format = { fill: colors.teal, font: { name: "Aptos", size: 10, bold: true, color: colors.white }, rowHeight: 25, verticalAlignment: "center" };
}
overview.getRange("A5:C9").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 28 };
overview.getRange("E5:G9").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 28 };
overview.getRange("A13:C20").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 27 };
overview.getRange("E13:L18").format = { fill: colors.paleGold, font: { name: "Aptos", size: 11, color: "#5B430B" }, wrapText: true, verticalAlignment: "center" };
overview.getRange("B5:B9").format.numberFormat = "#,##0";
overview.getRange("F5:F9").format.numberFormat = "#,##0";
overview.getRange("B13:B20").format.numberFormat = "#,##0";
overview.getRange("B20").format.numberFormat = "0";
overview.getRange("A23:C23").values = [["Classification heuristic", "Threshold", "Scope"]];
overview.getRange("A24:C29").values = [
  ["Strong attenuation", config.thresholds.attenuation_strong, "LOAO / GLOBAL Top1/3/5"],
  ["High Top1 mass share", config.thresholds.top1_share_high, "任一时期"],
  ["High Top5 mass share", config.thresholds.top5_share_high, "任一时期"],
  ["Minimum Top5 remaining", config.thresholds.top5_remaining_ratio_minimum, "两种估计量均须满足"],
  ["Sign-flip concern floor", config.thresholds.near_zero_original_delta, "原始 |delta|；raw flip 始终保留"],
  ["Attenuation NA floor", config.thresholds.relative_change_minimum_original, "原始 |delta| 低于此值时不计算比率"],
];
overview.getRange("A23:C23").format = { fill: colors.teal, font: { name: "Aptos", size: 10, bold: true, color: colors.white }, rowHeight: 25 };
overview.getRange("A24:C29").format = { font: { name: "Aptos", size: 10 }, wrapText: true, rowHeight: 28 };
overview.getRange("B24:B27").format.numberFormat = "0%";
overview.getRange("B28:B29").format.numberFormat = "0.000000";
overview.getRange("A:A").format.columnWidth = 29;
overview.getRange("B:B").format.columnWidth = 12;
overview.getRange("C:C").format.columnWidth = 44;
overview.getRange("D:D").format.columnWidth = 3;
overview.getRange("E:E").format.columnWidth = 31;
overview.getRange("F:F").format.columnWidth = 12;
overview.getRange("G:G").format.columnWidth = 40;
overview.getRange("H:L").format.columnWidth = 12;
overview.freezePanes.freezeRows(2);

const chart = overview.charts.add("bar", overview.getRange("A4:B9"));
chart.title = "Author robustness status counts (13 candidates)";
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
chart.yAxis = { numberFormatCode: "0", min: 0, max: 13 };
chart.setPosition("E20", "L36");

const formulaInspect = await workbook.inspect({ kind: "formula", sheetId: "Overview", range: "A1:L36", maxChars: 12000, options: { maxResults: 100 } });
const overviewInspect = await workbook.inspect({ kind: "region", sheetId: "Overview", range: "A1:L36", maxChars: 12000 });
const errorInspect = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "最终公式错误扫描", maxChars: 8000 });

const previewFiles = [];
for (const sheetName of ["Overview", ...sheetSpecs.map(([name]) => name)]) {
  const bounds = sheetName === "Overview" ? { lastColumn: "L", lastRow: 36, headers: [] } : sheetBounds.get(sheetName);
  const previewLastColumn = sheetName === "Overview" ? "L" : columnName(Math.min(bounds.headers.length, 14) - 1);
  const previewEndRow = sheetName === "Overview" ? 36 : Math.min(bounds.lastRow, 24);
  const blob = await workbook.render({ sheetName, range: `A1:${previewLastColumn}${previewEndRow}`, scale: 1, format: "png" });
  const previewPath = path.join(previewRoot, `${sheetName}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
  previewFiles.push(previewPath);
}
const supplementalPreviews = [];
for (const [sheetName, range] of [["Final_Assessment", "P4:S17"], ["Influence_Summary", "U4:AA24"]]) {
  const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const previewPath = path.join(previewRoot, `${sheetName}_scheme_audit.png`);
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
  supplementalPreviews.push(previewPath);
}

const workbookPath = path.join(outputRoot, "QTS_Author_Influence_LOAO.xlsx");
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(workbookPath);
const workbookBytes = await fs.readFile(workbookPath);
const workbookHash = crypto.createHash("sha256").update(workbookBytes).digest("hex");
const verification = {
  workbook_path: workbookPath, workbook_sha256: workbookHash, workbook_bytes: workbookBytes.length,
  sheet_count: 13, rendered_sheet_count: previewFiles.length, rendered_previews: previewFiles,
  supplemental_previews: supplementalPreviews,
  overview_inspect: overviewInspect.ndjson, formula_inspect: formulaInspect.ndjson, formula_error_scan: errorInspect.ndjson,
};
const verificationPath = path.join(outputRoot, "workbook_verification.json");
await fs.writeFile(verificationPath, JSON.stringify(verification, null, 2), "utf8");

const manifestPath = path.join(outputRoot, "analysis_manifest.json");
const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));
manifest.outputs["QTS_Author_Influence_LOAO.xlsx"] = { sha256: workbookHash, bytes: workbookBytes.length };
manifest.outputs["workbook_verification.json"] = {
  sha256: crypto.createHash("sha256").update(await fs.readFile(verificationPath)).digest("hex"),
  bytes: (await fs.stat(verificationPath)).size,
};
await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), "utf8");

console.log(JSON.stringify({
  workbookPath, workbookHash, sheetCount: 13, renderedSheetCount: previewFiles.length,
  formulaInspect: formulaInspect.ndjson, overviewInspect: overviewInspect.ndjson, formulaErrorScan: errorInspect.ndjson,
}, null, 2));
process.exitCode = 0;
