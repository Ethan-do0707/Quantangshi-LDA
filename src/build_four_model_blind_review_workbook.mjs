import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = process.cwd();
const auditRoot = path.join(root, "model_selection", "four_model_blind_semantic_audit");
const dataDir = path.join(auditRoot, "data");
const auditDir = path.join(auditRoot, "audit");
const renderDir = path.join(auditDir, "workbook_renders");
await fs.mkdir(renderDir, { recursive: true });

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, "");
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < source.length; i += 1) {
    const char = source[i];
    if (quoted) {
      if (char === '"' && source[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += char;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const [headers, ...body] = rows.filter((values) => values.some((value) => value !== ""));
  return body.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

async function loadCsv(filename) {
  return parseCsv(await fs.readFile(path.join(dataDir, filename), "utf8"));
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

const navy = "#172554";
const blue = "#0369A1";
const cyan = "#0E7490";
const slate = "#334155";
const paleBlue = "#E0F2FE";
const paleYellow = "#FEF3C7";
const paleGreen = "#DCFCE7";
const white = "#FFFFFF";
const grid = "#CBD5E1";
const font = "Microsoft YaHei";

const workbook = Workbook.create();
workbook.setColorScheme({
  name: "QTS Blind Audit",
  themeColors: { accent1: blue, accent2: cyan, bg1: white, tx1: navy },
});

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: navy,
    font: { name: font, bold: true, color: white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    rowHeightPx: 44,
    borders: { preset: "all", style: "thin", color: grid },
  };
}

function writeTableSheet(name, headers, rows, tableName, widths = {}) {
  const sheet = workbook.worksheets.add(name);
  const endColumn = columnLetter(headers.length);
  sheet.getRange(`A1:${endColumn}${rows.length + 1}`).values = [headers, ...rows];
  styleHeader(sheet, `A1:${endColumn}1`);
  const body = sheet.getRange(`A2:${endColumn}${rows.length + 1}`);
  body.format = {
    font: { name: font, size: 9, color: slate },
    verticalAlignment: "top",
    wrapText: true,
    rowHeightPx: 38,
    borders: { preset: "all", style: "thin", color: "#E2E8F0" },
  };
  for (let index = 0; index < headers.length; index += 1) {
    const width = widths[headers[index]] ?? 115;
    sheet.getRange(`${columnLetter(index + 1)}1:${columnLetter(index + 1)}${rows.length + 1}`).format.columnWidthPx = width;
  }
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add(`A1:${endColumn}${rows.length + 1}`, true, tableName);
  return sheet;
}

const topicReview = await loadCsv("blind_topic_ai_review.csv");
const cardIndex = await loadCsv(path.join("..", "audit", "blind_card_index.csv"));
const cardOrder = new Map(cardIndex.map((row) => [row.blind_topic_id, Number(row.card_order)]));
topicReview.sort((left, right) => cardOrder.get(left.blind_topic_id) - cardOrder.get(right.blind_topic_id));
const aiHeaders = [
  "blind_topic_id", "AI_provisional_label", "AI_word_coherence", "AI_poem_coherence",
  "AI_word_poem_agreement", "AI_labelability", "AI_literary_interpretability",
  "AI_semantic_specificity", "AI_overbreadth", "AI_internal_mixedness",
  "AI_topic_type", "AI_under_split_flag", "AI_notes",
];
const researcherHeaders = [
  "Researcher_Label", "Researcher_Word_Coherence", "Researcher_Poem_Coherence",
  "Researcher_Word_Poem_Agreement", "Researcher_Labelability", "Researcher_Interpretability",
  "Researcher_Specificity", "Researcher_Overbreadth", "Researcher_Mixedness",
  "Researcher_Topic_Type", "Researcher_UnderSplit", "Researcher_Notes",
];
const topicHeaders = [...aiHeaders, ...researcherHeaders];
const numericAi = new Set(aiHeaders.filter((header) => header.startsWith("AI_") && ![
  "AI_provisional_label", "AI_topic_type", "AI_under_split_flag", "AI_notes",
].includes(header)));
const topicRows = topicReview.map((row) => topicHeaders.map((header) => {
  if (researcherHeaders.includes(header)) return null;
  if (numericAi.has(header)) return Number(row[header]);
  return row[header];
}));

const overview = workbook.worksheets.add("Overview");
overview.getRange("A1:J1").merge();
overview.getRange("A1").values = [["QTS Four-Model Blind Researcher Review"]];
overview.getRange("A1:J1").format = {
  fill: navy,
  font: { name: font, bold: true, color: white, size: 18 },
  verticalAlignment: "middle",
  rowHeightPx: 48,
};
overview.getRange("A2:J2").merge();
overview.getRange("A2").values = [["AI-assisted blind audit｜AI 字段均为 AI_PROVISIONAL｜Researcher_* 字段全部留空，等待研究者复核"]];
overview.getRange("A2:J2").format = { fill: paleBlue, font: { name: font, color: navy, size: 10 }, rowHeightPx: 30 };
overview.getRange("A4:B8").values = [
  ["Review KPI", "Value"],
  ["Blind topic cards", 61],
  ["AI provisional reviews", 61],
  ["Researcher labels completed", null],
  ["Priority 1 items", null],
];
overview.getRange("B7").formulas = [["=COUNTA('Topic Review'!N2:N62)"]];
overview.getRange("B8").formulas = [["=COUNTIF('Priority Queue'!A2:A62,1)"]];
styleHeader(overview, "A4:B4");
overview.getRange("A5:B8").format = {
  font: { name: font, color: slate, size: 11 }, fill: "#F8FAFC",
  borders: { preset: "all", style: "thin", color: grid }, rowHeightPx: 28,
};
overview.getRange("B5:B8").format = { font: { name: font, bold: true, color: blue, size: 14 }, horizontalAlignment: "center" };
overview.getRange("D4:H8").values = [
  ["Blind model", "CLEAR", "BROAD", "MIXED", "NOISE"],
  ["MODEL_A", 6, 4, 0, 2],
  ["MODEL_B", 7, 6, 4, 3],
  ["MODEL_C", 5, 6, 3, 1],
  ["MODEL_D", 7, 2, 3, 2],
];
styleHeader(overview, "D4:H4");
overview.getRange("D5:H8").format = {
  font: { name: font, color: slate, size: 10 }, horizontalAlignment: "center",
  borders: { preset: "all", style: "thin", color: grid }, rowHeightPx: 26,
};
const qualityChart = overview.charts.add("column", overview.getRange("D4:H8"));
qualityChart.title = "Blind topic quality counts";
qualityChart.titleTextStyle.typeface = font;
qualityChart.xAxis = { axisType: "textAxis", textStyle: { typeface: font } };
qualityChart.yAxis = { numberFormatCode: "0", numberFormatSourceLinked: false, textStyle: { typeface: font } };
qualityChart.setPosition("A11", "J27");
overview.getRange("A29:J31").merge();
overview.getRange("A29").values = [["使用顺序：先按 Priority Queue 打开随机编号卡片，再在 Topic Review 的黄色区域填写研究者判断。不要依据模型 Topic 数推断真实身份。"]];
overview.getRange("A29:J31").format = { fill: paleYellow, font: { name: font, color: navy, size: 10 }, wrapText: true, verticalAlignment: "middle" };
overview.getRange("A1:J31").format.font.name = font;
overview.getRange("A:A").format.columnWidthPx = 235;
overview.getRange("B:B").format.columnWidthPx = 120;
overview.getRange("C:C").format.columnWidthPx = 25;
overview.getRange("D:H").format.columnWidthPx = 105;
overview.freezePanes.freezeRows(2);

const topicWidths = Object.fromEntries(topicHeaders.map((header) => [header, 96]));
topicWidths.blind_topic_id = 100;
topicWidths.AI_provisional_label = 230;
topicWidths.AI_notes = 430;
topicWidths.Researcher_Label = 230;
topicWidths.Researcher_Notes = 430;
topicWidths.AI_topic_type = 105;
topicWidths.Researcher_Topic_Type = 125;
const topicSheet = writeTableSheet("Topic Review", topicHeaders, topicRows, "TopicReviewTable", topicWidths);
topicSheet.getRange("C2:J62").format.horizontalAlignment = "center";
topicSheet.getRange("K2:L62").format.horizontalAlignment = "center";
topicSheet.getRange("N2:Y62").format.fill = paleYellow;
topicSheet.getRange("N1:Y1").format.fill = cyan;
topicSheet.getRange("O2:V62").dataValidation = {
  rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  allowBlank: true,
};
topicSheet.getRange("W2:W62").dataValidation = {
  rule: { type: "list", values: ["CLEAR", "BROAD", "MIXED", "NOISE"] },
  allowBlank: true,
};
topicSheet.getRange("X2:X62").dataValidation = {
  rule: { type: "list", values: ["TRUE", "FALSE", "UNCERTAIN"] },
  allowBlank: true,
};
topicSheet.getRange("N2:N62").conditionalFormats.add("Custom", {
  formula: "=LEN(N2)>0",
  format: { fill: paleGreen },
});

const genericSheets = [
  ["Under-split", "blind_under_split_audit.csv", "UnderSplitTable"],
  ["Fragmentation", "blind_fragmentation_audit.csv", "FragmentationTable"],
  ["Alignment", "blind_cross_model_alignment.csv", "AlignmentTable"],
  ["Family Matrix", "blind_semantic_family_matrix.csv", "FamilyMatrixTable"],
  ["Priority Queue", "blind_researcher_priority_queue.csv", "PriorityQueueTable"],
];
for (const [sheetName, filename, tableName] of genericSheets) {
  const records = await loadCsv(filename);
  const headers = Object.keys(records[0]);
  const numericColumns = new Set([
    "priority", "card_order", "topic_word_cosine_candidate", "top20_jaccard_candidate",
    "candidate_rank", "topic_word_cosine", "top20_jaccard",
  ]);
  const rows = records.map((record) => headers.map((header) => {
    if (numericColumns.has(header) && record[header] !== "") return Number(record[header]);
    return record[header];
  }));
  const widths = Object.fromEntries(headers.map((header) => [header, 125]));
  for (const header of headers) {
    if (/evidence|rationale|reason|label/i.test(header)) widths[header] = 340;
    if (/topic_ids|topic_id/.test(header)) widths[header] = 135;
  }
  const sheet = writeTableSheet(sheetName, headers, rows, tableName, widths);
  if (sheetName === "Priority Queue") {
    sheet.getRange(`A2:A${rows.length + 1}`).format.horizontalAlignment = "center";
    sheet.getRange(`A2:A${rows.length + 1}`).conditionalFormats.add("cellIs", {
      operator: "equalTo", formula: 1, format: { fill: "#FECACA", font: { bold: true, color: "#991B1B" } },
    });
  }
}

const guide = workbook.worksheets.add("Guide");
guide.getRange("A1:F1").merge();
guide.getRange("A1").values = [["Researcher Review Guide"]];
guide.getRange("A1:F1").format = { fill: navy, font: { name: font, bold: true, color: white, size: 17 }, rowHeightPx: 46 };
guide.getRange("A3:F12").values = [
  ["Field", "1", "2", "3", "4", "5"],
  ["Coherence", "无一致性", "明显分裂", "两个相邻方向", "核心明确", "高度集中"],
  ["Labelability", "难以命名", "需列举多方向", "复合标题", "主标签+限定", "简洁准确"],
  ["Interpretability", "意义不明", "统计共现", "可解释但宽泛", "研究价值较强", "明确解释单位"],
  ["Specificity", "残余桶", "高度泛化", "较宽", "较明确", "边界清楚"],
  ["Overbreadth", "范围适中", "略宽", "相邻子主题", "多个可分语义", "严重欠分"],
  ["Mixedness", "单一中心", "轻微混入", "两个方向", "多个语义场", "残余/混合桶"],
  ["Topic Type", "CLEAR", "BROAD", "MIXED", "NOISE", "—"],
  ["Priority 1", "MIXED", "欠分", "碎片化对", "—", "—"],
  ["Status", "所有 AI 判断均为 AI_PROVISIONAL", "", "", "", ""],
];
styleHeader(guide, "A3:F3");
guide.getRange("A4:F12").format = {
  font: { name: font, color: slate, size: 10 }, wrapText: true, verticalAlignment: "middle",
  borders: { preset: "all", style: "thin", color: grid }, rowHeightPx: 42,
};
guide.getRange("A:A").format.columnWidthPx = 145;
guide.getRange("B:F").format.columnWidthPx = 175;
guide.freezePanes.freezeRows(3);

const previewRanges = {
  "Overview": "A1:J31",
  "Topic Review": "A1:Y18",
  "Under-split": "A1:F11",
  "Fragmentation": "A1:I5",
  "Alignment": "A1:K18",
  "Family Matrix": "A1:M10",
  "Priority Queue": "A1:H18",
  "Guide": "A1:F12",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 0.8, format: "png" });
  const safeName = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(path.join(renderDir, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 5,
  tableMaxCols: 8,
  tableMaxCellChars: 100,
});
await fs.writeFile(path.join(auditDir, "workbook_inspection.json"), JSON.stringify(inspection, null, 2), "utf8");

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(path.join(auditRoot, "QTS_Four_Model_Blind_Researcher_Review.xlsx"));
console.log(JSON.stringify({
  status: "WORKBOOK_CREATED",
  topic_rows: topicRows.length,
  researcher_fields_blank: topicRows.every((row) => row.slice(aiHeaders.length).every((value) => value === null)),
  rendered_sheets: Object.keys(previewRanges).length,
  output: path.join(auditRoot, "QTS_Four_Model_Blind_Researcher_Review.xlsx"),
}, null, 2));
