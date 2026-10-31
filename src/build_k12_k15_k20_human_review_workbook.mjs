import fs from "node:fs/promises";
import path from "node:path";
import JSZip from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/jszip/lib/index.js";
import { SpreadsheetFile, Workbook } from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = process.cwd();
const taskRoot = path.join(root, "model_selection", "K12_K15_K20_human_adjudication");
const dataDir = path.join(taskRoot, "data");
const privateDir = path.join(taskRoot, "_private");
const candidateDir = path.join(privateDir, "stage2_candidates");
const auditDir = path.join(taskRoot, "audit");
const renderDir = path.join(auditDir, "workbook_renders");
const outputPath = path.join(taskRoot, "QTS_K12_K15_K20_Human_Review.xlsx");
await fs.mkdir(renderDir, { recursive: true });

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
  const [headers, ...body] = nonEmpty;
  return body.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
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

function numberOrBlank(value) {
  if (value === "" || value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : value;
}

const colors = {
  navy: "#17324D",
  teal: "#0F6B78",
  slate: "#334155",
  white: "#FFFFFF",
  paleBlue: "#E8F1F5",
  paleYellow: "#FFF4CC",
  paleGreen: "#E5F4EA",
  paleGray: "#F5F7F9",
  grid: "#D9E1E7",
  red: "#991B1B",
};
const font = "Microsoft YaHei";
const workbook = Workbook.create();
workbook.setColorScheme({
  name: "QTS Human Adjudication",
  themeColors: { accent1: colors.teal, accent2: colors.navy, bg1: colors.white, tx1: colors.navy },
});

function styleHeader(sheet, range) {
  sheet.getRange(range).format = {
    fill: colors.navy,
    font: { name: font, bold: true, color: colors.white, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "middle",
    wrapText: true,
    rowHeightPx: 46,
    borders: { preset: "all", style: "thin", color: colors.grid },
  };
}

function writeTableSheet(name, headers, rows, tableName, widths = {}, rowHeightPx = 40) {
  const sheet = workbook.worksheets.add(name);
  const lastColumn = columnLetter(headers.length);
  const lastRow = rows.length + 1;
  sheet.getRange(`A1:${lastColumn}${lastRow}`).values = [headers, ...rows];
  styleHeader(sheet, `A1:${lastColumn}1`);
  if (rows.length) {
    sheet.getRange(`A2:${lastColumn}${lastRow}`).format = {
      font: { name: font, size: 9, color: colors.slate },
      verticalAlignment: "top",
      wrapText: true,
      rowHeightPx,
      borders: { preset: "all", style: "thin", color: colors.grid },
    };
  }
  headers.forEach((header, index) => {
    sheet.getRange(`${columnLetter(index + 1)}1:${columnLetter(index + 1)}${lastRow}`).format.columnWidthPx = widths[header] ?? 125;
  });
  sheet.freezePanes.freezeRows(1);
  sheet.tables.add(`A1:${lastColumn}${lastRow}`, true, tableName);
  return sheet;
}

const topicReview = await loadCsv(path.join(dataDir, "researcher_topic_review_template.csv"));
const poemEvidence = await loadCsv(path.join(dataDir, "blind_poem_evidence.csv"));
const split1215 = await loadCsv(path.join(candidateDir, "K12_to_K15_split_merge_candidates.csv"));
const split1520 = await loadCsv(path.join(candidateDir, "K15_to_K20_split_merge_candidates.csv"));
const fragmentation = await loadCsv(path.join(candidateDir, "K20_fragmentation_candidates.csv"));
const hiddenAi = await loadCsv(path.join(privateDir, "hidden_ai_reference.csv"));

const researcherTopicFields = [
  "Researcher_Label",
  "Researcher_Word_Coherence",
  "Researcher_Poem_Coherence",
  "Researcher_Word_Poem_Agreement",
  "Researcher_Labelability",
  "Researcher_Literary_Interpretability",
  "Researcher_Semantic_Specificity",
  "Researcher_Overbreadth",
  "Researcher_Internal_Mixedness",
  "Researcher_Topic_Type",
  "Researcher_Needs_Splitting",
  "Researcher_Suspected_Subthemes",
  "Researcher_Confidence",
  "Researcher_Notes",
];

const instructions = workbook.worksheets.add("Instructions");
instructions.getRange("A1:F1").merge();
instructions.getRange("A1").values = [["《全唐诗》主题模型研究者盲审工作簿"]];
instructions.getRange("A1:F1").format = {
  fill: colors.navy,
  font: { name: font, bold: true, color: colors.white, size: 18 },
  verticalAlignment: "middle",
  rowHeightPx: 50,
};
instructions.getRange("A3:B7").values = [
  ["审核状态", "数值"],
  ["应审主题", 47],
  ["已填主题标签", null],
  ["审核进度", null],
  ["当前状态", null],
];
instructions.getRange("B5").formulas = [["=COUNTA('Blind_Topic_Review'!D2:D48)"]];
instructions.getRange("B6").formulas = [["=B5/B4"]];
instructions.getRange("B7").formulas = [["=IF(B5=B4,\"TOPIC_REVIEW_COMPLETE\",\"TOPIC_REVIEW_INCOMPLETE\")"]];
instructions.getRange("B6").format.numberFormat = "0%";
styleHeader(instructions, "A3:B3");
instructions.getRange("A4:B7").format = {
  font: { name: font, size: 10, color: colors.slate },
  fill: colors.paleGray,
  borders: { preset: "all", style: "thin", color: colors.grid },
  rowHeightPx: 29,
};
instructions.getRange("B4:B7").format.font = { name: font, size: 11, bold: true, color: colors.teal };
instructions.getRange("A9:F16").values = [
  ["步骤", "操作", "说明", "限制", "完成标志", "备注"],
  ["1", "逐一查看盲号", "先读 Top30 词，再读诗证据；不得猜测候选身份。", "独立判断", "47项均填", "保存工作簿"],
  ["2", "填写14个研究者字段", "黄色单元格均由研究者本人填写。", "不得委托自动生成", "校验无缺项", "评分均为1–5整数"],
  ["3", "处理拆分判断", "若 Needs_Splitting=YES，必须写明疑似子主题。", "条件必填", "无条件错误", "可写多个子主题"],
  ["4", "运行验证脚本", "验证只检查完整性与枚举值，不评价语义。", "未完成不得冻结", "状态为 COMPLETE", "保留原工作簿"],
  ["5", "冻结 Topic 级审核", "冻结后生成只读式 CSV 与 SHA256；此后不得改写。", "冻结后方可解盲", "哈希生成", "后续页面才可启用"],
  ["证据规则", "原始材料优先", "诗题、作者、正文与概率仅用于主题语义判断。", "不使用历史判断", "全程可追溯", "盲号全局随机"],
  ["停止点", "完成第一阶段后停止", "当前文件只供 Topic 级盲审；不要进行模型级裁决。", "不提前解盲", "等待下一阶段", "不作最终选择"],
];
styleHeader(instructions, "A9:F9");
instructions.getRange("A10:F16").format = {
  font: { name: font, size: 10, color: colors.slate },
  verticalAlignment: "middle",
  wrapText: true,
  rowHeightPx: 48,
  borders: { preset: "all", style: "thin", color: colors.grid },
};
instructions.getRange("A18:F27").values = [
  ["评分项", "1", "2", "3", "4", "5",],
  ["Word / Poem Coherence", "无中心", "明显分裂", "部分一致", "中心清楚", "高度集中"],
  ["Word–Poem Agreement", "严重不符", "多处不符", "大体对应", "对应良好", "高度吻合"],
  ["Labelability", "无法命名", "勉强列举", "可作复合标题", "主标签明确", "简洁准确"],
  ["Literary Interpretability", "意义不明", "仅统计共现", "宽泛可释", "具研究价值", "明确解释单位"],
  ["Semantic Specificity", "残余语汇", "高度泛化", "边界偏宽", "语义明确", "边界清晰"],
  ["Overbreadth", "范围适中", "略宽", "含相邻方向", "多个可分方向", "严重欠分"],
  ["Internal Mixedness", "单一中心", "轻微混入", "双重方向", "多重语义场", "混合/噪声"],
  ["Topic Type", "CLEAR", "BROAD", "MIXED", "NOISE", "—"],
  ["Confidence", "LOW", "MEDIUM", "HIGH", "—", "—"],
];
styleHeader(instructions, "A18:F18");
instructions.getRange("A19:F27").format = {
  font: { name: font, size: 10, color: colors.slate },
  verticalAlignment: "middle",
  wrapText: true,
  rowHeightPx: 40,
  borders: { preset: "all", style: "thin", color: colors.grid },
};
instructions.getRange("A:A").format.columnWidthPx = 175;
instructions.getRange("B:B").format.columnWidthPx = 180;
instructions.getRange("C:F").format.columnWidthPx = 165;
instructions.freezePanes.freezeRows(1);

const topicHeaders = ["blind_topic_id", "Top30_words", "representative_poem_summary", ...researcherTopicFields];
const topicRows = topicReview.map((record) => topicHeaders.map((header) => {
  if (researcherTopicFields.includes(header)) return null;
  return record[header];
}));
const topicWidths = Object.fromEntries(topicHeaders.map((header) => [header, 105]));
topicWidths.blind_topic_id = 95;
topicWidths.Top30_words = 420;
topicWidths.representative_poem_summary = 420;
topicWidths.Researcher_Label = 250;
topicWidths.Researcher_Suspected_Subthemes = 300;
topicWidths.Researcher_Notes = 430;
const topicSheet = writeTableSheet("Blind_Topic_Review", topicHeaders, topicRows, "BlindTopicReviewTable", topicWidths, 70);
topicSheet.getRange("B2:C48").format.fill = colors.paleGray;
topicSheet.getRange("D2:Q48").format.fill = colors.paleYellow;
topicSheet.getRange("D1:Q1").format.fill = colors.teal;
topicSheet.getRange("E2:L48").dataValidation = {
  rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
  allowBlank: true,
};
topicSheet.getRange("M2:M48").dataValidation = {
  rule: { type: "list", values: ["CLEAR", "BROAD", "MIXED", "NOISE"] },
  allowBlank: true,
};
topicSheet.getRange("N2:N48").dataValidation = {
  rule: { type: "list", values: ["YES", "NO", "UNCERTAIN"] },
  allowBlank: true,
};
topicSheet.getRange("P2:P48").dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] },
  allowBlank: true,
};
topicSheet.getRange("D2:D48").conditionalFormats.add("Custom", {
  formula: "=LEN(D2)>0",
  format: { fill: colors.paleGreen },
});

const poemHeaders = ["blind_topic_id", "sample_type", "rank", "poem_id", "author", "title", "topic_probability", "original_text"];
const poemRows = poemEvidence.map((record) => poemHeaders.map((header) => {
  if (["rank", "poem_id", "topic_probability"].includes(header)) return numberOrBlank(record[header]);
  return record[header];
}));
const poemSheet = writeTableSheet("Blind_Poem_Evidence", poemHeaders, poemRows, "BlindPoemEvidenceTable", {
  blind_topic_id: 95,
  sample_type: 175,
  rank: 60,
  poem_id: 85,
  author: 110,
  title: 280,
  topic_probability: 120,
  original_text: 760,
}, 80);
poemSheet.getRange(`G2:G${poemRows.length + 1}`).format.numberFormat = "0.000000";
poemSheet.getRange(`A2:G${poemRows.length + 1}`).format.fill = colors.paleGray;

const modelHeaders = [
  "model_identity", "topic_count", "Researcher_Model_Strengths", "Researcher_Model_Weaknesses",
  "Researcher_Semantic_Resolution", "Researcher_Overbreadth_Assessment",
  "Researcher_Fragmentation_Assessment", "Researcher_Confidence", "Researcher_Notes",
];
const modelRows = [12, 15, 20].map((value) => [`K${value}`, value, null, null, null, null, null, null, null]);
const modelSheet = writeTableSheet("Model_Level_Review", modelHeaders, modelRows, "ModelLevelReviewTable", {
  model_identity: 110,
  topic_count: 90,
  Researcher_Model_Strengths: 360,
  Researcher_Model_Weaknesses: 360,
  Researcher_Semantic_Resolution: 230,
  Researcher_Overbreadth_Assessment: 270,
  Researcher_Fragmentation_Assessment: 290,
  Researcher_Confidence: 130,
  Researcher_Notes: 430,
}, 72);
modelSheet.getRange("C2:I4").format.fill = colors.paleYellow;
modelSheet.getRange("H2:H4").dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
};

function buildRelationSheet(sheetName, records, tableName) {
  const headers = [
    "candidate_event_id", "relation_candidate", "lower_topic_ids", "lower_blind_topic_ids",
    "higher_topic_ids", "higher_blind_topic_ids", "source_top20_words", "target_top20_words",
    "source_top10_poems", "target_top10_poems", "source_author_balanced_top10",
    "target_author_balanced_top10", "maximum_topic_word_cosine", "minimum_selected_topic_word_cosine",
    "Researcher_Judgment", "Researcher_Rationale", "Researcher_Confidence",
  ];
  const rows = records.map((record) => headers.map((header) => {
    if (["maximum_topic_word_cosine", "minimum_selected_topic_word_cosine"].includes(header)) return numberOrBlank(record[header]);
    if (header.startsWith("Researcher_")) return null;
    return record[header];
  }));
  const widths = Object.fromEntries(headers.map((header) => [header, 125]));
  for (const header of ["source_top20_words", "target_top20_words"]) widths[header] = 380;
  for (const header of ["source_top10_poems", "target_top10_poems", "source_author_balanced_top10", "target_author_balanced_top10"]) widths[header] = 520;
  widths.Researcher_Judgment = 230;
  widths.Researcher_Rationale = 420;
  widths.Researcher_Confidence = 130;
  const sheet = writeTableSheet(sheetName, headers, rows, tableName, widths, 88);
  const lastRow = rows.length + 1;
  sheet.getRange(`O2:Q${lastRow}`).format.fill = colors.paleYellow;
  sheet.getRange(`O2:O${lastRow}`).dataValidation = {
    rule: {
      type: "list",
      values: [
        "MEANINGFUL_SPLIT", "MINOR_REFINEMENT", "FRAGMENTATION",
        "MEANINGFUL_AGGREGATION", "ACCEPTABLE_BROADENING", "PROBLEMATIC_MERGE", "UNCERTAIN",
      ],
    },
    allowBlank: true,
  };
  sheet.getRange(`Q2:Q${lastRow}`).dataValidation = {
    rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
  };
  sheet.getRange(`M2:N${lastRow}`).format.numberFormat = "0.000000";
  return { sheet, rows };
}

const relation1215 = buildRelationSheet("K12_to_K15_Splits", split1215, "K12K15SplitMergeTable");
const relation1520 = buildRelationSheet("K15_to_K20_Splits", split1520, "K15K20SplitMergeTable");

const fragmentationHeaders = [
  "candidate_pair_id", "left_topic_id", "left_blind_topic_id", "right_topic_id", "right_blind_topic_id",
  "topic_word_cosine", "top20_word_jaccard", "top20_representative_poem_jaccard",
  "left_top20_words", "right_top20_words", "left_top10_poems", "right_top10_poems",
  "Researcher_Fragmentation_Judgment", "Researcher_Fragmentation_Rationale", "Researcher_Confidence",
];
const fragmentationRows = fragmentation.map((record) => fragmentationHeaders.map((header) => {
  if (["left_topic_id", "right_topic_id", "topic_word_cosine", "top20_word_jaccard", "top20_representative_poem_jaccard"].includes(header)) return numberOrBlank(record[header]);
  if (header.startsWith("Researcher_")) return null;
  return record[header];
}));
const fragmentationSheet = writeTableSheet("Fragmentation_Review", fragmentationHeaders, fragmentationRows, "FragmentationReviewTable", {
  candidate_pair_id: 110,
  left_topic_id: 90,
  left_blind_topic_id: 110,
  right_topic_id: 90,
  right_blind_topic_id: 110,
  topic_word_cosine: 115,
  top20_word_jaccard: 120,
  top20_representative_poem_jaccard: 165,
  left_top20_words: 380,
  right_top20_words: 380,
  left_top10_poems: 520,
  right_top10_poems: 520,
  Researcher_Fragmentation_Judgment: 260,
  Researcher_Fragmentation_Rationale: 430,
  Researcher_Confidence: 130,
}, 88);
fragmentationSheet.getRange(`M2:O${fragmentationRows.length + 1}`).format.fill = colors.paleYellow;
fragmentationSheet.getRange(`M2:M${fragmentationRows.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["TRUE_FRAGMENTATION", "MEANINGFUL_SEPARATION", "UNCERTAIN"] }, allowBlank: true,
};
fragmentationSheet.getRange(`O2:O${fragmentationRows.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
};
fragmentationSheet.getRange(`F2:H${fragmentationRows.length + 1}`).format.numberFormat = "0.000000";

const familyHeaders = [
  "family_id", "Researcher_Family_Label", "Researcher_Semantic_Importance",
  "K12_Coverage", "K15_Coverage", "K20_Coverage", "Researcher_Comparative_Rationale",
];
const familyRows = Array.from({ length: 20 }, (_, index) => [`RF${String(index + 1).padStart(2, "0")}`, null, null, null, null, null, null]);
const familySheet = writeTableSheet("Semantic_Family_Coverage", familyHeaders, familyRows, "SemanticFamilyCoverageTable", {
  family_id: 90,
  Researcher_Family_Label: 260,
  Researcher_Semantic_Importance: 180,
  K12_Coverage: 150,
  K15_Coverage: 150,
  K20_Coverage: 150,
  Researcher_Comparative_Rationale: 480,
}, 60);
familySheet.getRange("B2:G21").format.fill = colors.paleYellow;
familySheet.getRange("C2:C21").dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
};
for (const column of ["D", "E", "F"]) {
  familySheet.getRange(`${column}2:${column}21`).dataValidation = {
    rule: { type: "list", values: ["CLEAR", "PARTIAL", "ABSENT", "FRAGMENTED", "OVERBROAD", "UNCERTAIN"] }, allowBlank: true,
  };
}

const evidenceItems = [
  "Quantitative fit and stability", "Topic-level semantic coherence", "Poem-level coherence",
  "Word–poem agreement", "Labelability", "Literary interpretability", "Semantic specificity",
  "Overbreadth burden", "Internal mixedness burden", "Split/merge quality", "Fragmentation burden",
  "Semantic-family coverage", "Researcher overall assessment",
];
const matrixHeaders = ["evidence_dimension", "K12", "K15", "K20", "Researcher_Evidence_Weight", "Researcher_Notes"];
const matrixRows = evidenceItems.map((item) => [item, null, null, null, null, null]);
const matrixSheet = writeTableSheet("Final_Evidence_Matrix", matrixHeaders, matrixRows, "FinalEvidenceMatrixTable", {
  evidence_dimension: 260,
  K12: 250,
  K15: 250,
  K20: 250,
  Researcher_Evidence_Weight: 190,
  Researcher_Notes: 470,
}, 62);
matrixSheet.getRange(`B2:F${matrixRows.length + 1}`).format.fill = colors.paleYellow;
matrixSheet.getRange(`E2:E${matrixRows.length + 1}`).dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
};

const finalHeaders = ["decision_item", "Researcher_Final_Response"];
const finalRows = [
  ["Researcher_Final_K", null],
  ["Researcher_Final_Rationale", null],
  ["Researcher_Final_Confidence", null],
  ["Researcher_Residual_Risks", null],
];
const finalSheet = writeTableSheet("Final_Decision", finalHeaders, finalRows, "FinalDecisionTable", {
  decision_item: 240,
  Researcher_Final_Response: 760,
}, 72);
finalSheet.getRange("B2:B5").format.fill = colors.paleYellow;
finalSheet.getRange("B2").dataValidation = {
  rule: { type: "list", values: [12, 15, 20] }, allowBlank: true,
};
finalSheet.getRange("B4").dataValidation = {
  rule: { type: "list", values: ["HIGH", "MEDIUM", "LOW"] }, allowBlank: true,
};

const hiddenAiHeaders = Object.keys(hiddenAi[0]);
const hiddenAiRows = hiddenAi.map((record) => hiddenAiHeaders.map((header) => {
  if (/^AI_(word|poem|label|literary|semantic|overbreadth|internal)/i.test(header)) return numberOrBlank(record[header]);
  return record[header];
}));
writeTableSheet("Hidden_AI_Reference", hiddenAiHeaders, hiddenAiRows, "HiddenAIReferenceTable", Object.fromEntries(hiddenAiHeaders.map((header) => [header, /notes|label/i.test(header) ? 410 : 120])), 66);

const previewRanges = {
  Instructions: "A1:F27",
  Blind_Topic_Review: "A1:Q12",
  Blind_Poem_Evidence: "A1:H12",
  Model_Level_Review: "A1:I4",
  K12_to_K15_Splits: `A1:Q${Math.min(relation1215.rows.length + 1, 8)}`,
  K15_to_K20_Splits: `A1:Q${Math.min(relation1520.rows.length + 1, 8)}`,
  Fragmentation_Review: `A1:O${Math.min(fragmentationRows.length + 1, 8)}`,
  Semantic_Family_Coverage: "A1:G10",
  Final_Evidence_Matrix: `A1:F${matrixRows.length + 1}`,
  Final_Decision: "A1:B5",
  Hidden_AI_Reference: "A1:N8",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const rendered = await workbook.render({ sheetName, range, scale: 0.72, format: "png" });
  const safeName = sheetName.toLowerCase();
  await fs.writeFile(path.join(renderDir, `${safeName}.png`), new Uint8Array(await rendered.arrayBuffer()));
}

const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 20000,
  tableMaxRows: 4,
  tableMaxCols: 8,
  tableMaxCellChars: 120,
});
await fs.writeFile(path.join(auditDir, "workbook_inspection.json"), `${JSON.stringify(inspection, null, 2)}\n`, "utf8");
const formulaTokens = ["#REF!", "#DIV/0!", "#VALUE!", "#NAME?", "#N/A"];
const inspectionText = JSON.stringify(inspection);
const formulaErrors = formulaTokens.filter((token) => inspectionText.includes(token));
await fs.writeFile(path.join(auditDir, "workbook_formula_errors.json"), `${JSON.stringify({ error_count: formulaErrors.length, errors: formulaErrors }, null, 2)}\n`, "utf8");

// artifact-tool 负责内容、公式、样式、验证与渲染；以下仅补写其当前未公开的工作表可见性属性。
const rawXlsx = await SpreadsheetFile.exportXlsx(workbook);
const rawPath = path.join(auditDir, "_workbook_before_visibility_hardening.xlsx");
await rawXlsx.save(rawPath);
const archive = await JSZip.loadAsync(await fs.readFile(rawPath));
const workbookXmlPath = "xl/workbook.xml";
const workbookXml = await archive.file(workbookXmlPath).async("string");
const veryHiddenSheets = new Set([
  "Model_Level_Review",
  "K12_to_K15_Splits",
  "K15_to_K20_Splits",
  "Fragmentation_Review",
  "Semantic_Family_Coverage",
  "Final_Evidence_Matrix",
  "Final_Decision",
  "Hidden_AI_Reference",
]);
const hardenedXml = workbookXml.replace(/<(?:[A-Za-z0-9_]+:)?sheet\b[^>]*\/>/g, (tag) => {
  const match = tag.match(/\bname="([^"]+)"/);
  if (!match || !veryHiddenSheets.has(match[1])) return tag;
  const withoutState = tag.replace(/\sstate="[^"]*"/g, "");
  return withoutState.replace(/\s*\/>$/, ' state="veryHidden"/>');
});
archive.file(workbookXmlPath, hardenedXml);
const finalBytes = await archive.generateAsync({ type: "uint8array", compression: "DEFLATE", compressionOptions: { level: 6 } });
await fs.writeFile(outputPath, finalBytes);
await fs.unlink(rawPath);
try {
  await fs.unlink(`${rawPath}.inspect.ndjson`);
} catch (error) {
  if (error.code !== "ENOENT") throw error;
}

console.log(JSON.stringify({
  status: "WORKBOOK_CREATED",
  topic_rows: topicRows.length,
  poem_evidence_rows: poemRows.length,
  K12_to_K15_candidate_events: relation1215.rows.length,
  K15_to_K20_candidate_events: relation1520.rows.length,
  K20_fragmentation_candidate_pairs: fragmentationRows.length,
  researcher_topic_fields_blank: topicRows.every((row) => row.slice(3).every((value) => value === null)),
  visible_sheets: ["Instructions", "Blind_Topic_Review", "Blind_Poem_Evidence"],
  very_hidden_sheets: [...veryHiddenSheets],
  rendered_sheets: Object.keys(previewRanges).length,
  formula_error_count: formulaErrors.length,
  output: outputPath,
}, null, 2));
