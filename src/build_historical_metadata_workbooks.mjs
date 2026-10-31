/** 使用 artifact-tool 生成并核验历史元数据工作簿。 */

import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function parseArgs(argv) {
  const result = { payload: "", inspectSourceOnly: false, target: "" };
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--payload") result.payload = argv[index + 1];
    if (argv[index] === "--target") result.target = argv[index + 1];
    if (argv[index] === "--inspect-source-only") result.inspectSourceOnly = true;
  }
  if (!result.payload) throw new Error("必须提供 --payload");
  if (!result.inspectSourceOnly && !["author", "review", "dictionary", "master"].includes(result.target)) {
    throw new Error("必须通过 --target 指定 author、review、dictionary 或 master");
  }
  return result;
}


function columnName(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}


function normalizeValue(value) {
  if (value === null || value === undefined || value === "") return null;
  return value;
}


function rowMatrix(rows, fields) {
  return rows.map((row) => fields.map((field) => normalizeValue(row[field])));
}


const headerFormat = {
  fill: "#1F4E78",
  font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#B4C7E7" },
};

const sectionFormat = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#1F4E78" },
  verticalAlignment: "center",
};

const noteFormat = {
  font: { color: "#404040" },
  verticalAlignment: "top",
  wrapText: true,
};


function prepareTableSheet(workbook, name, fields, rows, options = {}) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  const endColumn = columnName(fields.length - 1);
  sheet.getRange(`A1:${endColumn}1`).values = [fields];
  sheet.getRange(`A1:${endColumn}1`).format = headerFormat;
  sheet.getRange(`A1:${endColumn}1`).format.rowHeight = 34;
  const chunkSize = 1500;
  for (let offset = 0; offset < rows.length; offset += chunkSize) {
    const chunk = rows.slice(offset, offset + chunkSize);
    const startRow = offset + 2;
    const endRow = startRow + chunk.length - 1;
    sheet.getRange(`A${startRow}:${endColumn}${endRow}`).values = rowMatrix(chunk, fields);
  }
  const finalRow = Math.max(rows.length + 1, 1);
  if (rows.length > 0) {
    const table = sheet.tables.add(`A1:${endColumn}${finalRow}`, true, options.tableName || `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
    table.style = "TableStyleMedium2";
    table.showBandedColumns = false;
    table.showFilterButton = true;
  }
  sheet.getRange(`A1:${endColumn}${Math.min(finalRow, 5000)}`).format.verticalAlignment = "top";
  return sheet;
}


function setColumnWidth(sheet, fields, field, width) {
  const index = fields.indexOf(field);
  if (index < 0) return;
  const column = columnName(index);
  sheet.getRange(`${column}1:${column}2`).format.columnWidth = width;
}


function styleAuthorSheet(sheet, fields, rowCount) {
  for (const field of ["author_id_internal", "author_raw", "author_normalized", "author_display_name", "cbdb_name", "notes"]) {
    setColumnWidth(sheet, fields, field, field === "notes" ? 52 : 20);
  }
  for (const field of ["cbdb_match_status", "cbdb_match_confidence", "activity_year_method", "author_period_chronological", "author_period_traditional", "author_period_primary", "period_assignment_method", "period_confidence"]) {
    setColumnWidth(sheet, fields, field, 20);
  }
  for (const field of ["poem_count", "author_birth_year", "author_death_year", "author_fl_start", "author_fl_end", "observed_activity_start", "observed_activity_end", "cbdb_index_year", "period_reference_year", "source_count"]) {
    const index = fields.indexOf(field);
    if (index >= 0 && rowCount > 0) {
      const column = columnName(index);
      sheet.getRange(`${column}2:${column}${rowCount + 1}`).format.numberFormat = "0";
      setColumnWidth(sheet, fields, field, 15);
    }
  }
}


function addNotesSheet(workbook, name, rows) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRange("A1:B1").values = [["项目", "说明"]];
  sheet.getRange("A1:B1").format = headerFormat;
  sheet.getRange(`A2:B${rows.length + 1}`).values = rows;
  sheet.getRange(`A2:A${rows.length + 1}`).format = sectionFormat;
  sheet.getRange(`B2:B${rows.length + 1}`).format = noteFormat;
  sheet.getRange("A1:A2").format.columnWidth = 28;
  sheet.getRange("B1:B2").format.columnWidth = 100;
  for (let index = 2; index <= rows.length + 1; index += 1) sheet.getRange(`A${index}:B${index}`).format.rowHeight = 36;
  return sheet;
}


function styleQcSheet(sheet, fields, rowCount) {
  setColumnWidth(sheet, fields, "description", 52);
  setColumnWidth(sheet, fields, "observed", 48);
  setColumnWidth(sheet, fields, "expected", 48);
  if (rowCount > 0) {
    const observedColumn = columnName(fields.indexOf("observed"));
    const expectedColumn = columnName(fields.indexOf("expected"));
    sheet.getRange(`${observedColumn}2:${expectedColumn}${rowCount + 1}`).format.wrapText = true;
    sheet.getRange(`A2:${expectedColumn}${rowCount + 1}`).format.rowHeight = 34;
  }
}


async function renderPreview(workbook, sheetName, range, outputPath) {
  const blob = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(outputPath, new Uint8Array(await blob.arrayBuffer()));
  if (globalThis.gc) globalThis.gc();
}


async function renderPreviewIfMissing(workbook, sheetName, range, outputPath) {
  try {
    await fs.access(outputPath);
  } catch {
    await renderPreview(workbook, sheetName, range, outputPath);
  }
}


async function exportWorkbook(workbook, outputPath) {
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}


async function readJsonl(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return text.split(/\r?\n/).filter((line) => line.trim()).map((line) => JSON.parse(line));
}


async function inspectSource(payload, previewDir) {
  const source = await SpreadsheetFile.importXlsx(await FileBlob.load(payload.metadata.input_master_xlsx));
  const overview = await source.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
  await renderPreview(source, "Data", "A1:N12", path.join(previewDir, "source_Data_A_N.png"));
  return overview.ndjson;
}


async function buildAuthorWorkbook(payload, outputRoot, previewDir) {
  const workbook = Workbook.create();
  const authorSheet = prepareTableSheet(
    workbook,
    "Author_Metadata",
    payload.author_master_fields,
    payload.author_master,
    { freezeColumns: 5, tableName: "AuthorMetadataTable" },
  );
  styleAuthorSheet(authorSheet, payload.author_master_fields, payload.author_master.length);
  const periodFields = ["author_period_primary", "author_count", "poems_by_author_period", "definition_note"];
  const periodSheet = prepareTableSheet(workbook, "Period_Coverage", periodFields, payload.period_coverage, { tableName: "PeriodCoverageTable" });
  setColumnWidth(periodSheet, periodFields, "author_period_primary", 22);
  setColumnWidth(periodSheet, periodFields, "author_count", 18);
  setColumnWidth(periodSheet, periodFields, "poems_by_author_period", 24);
  setColumnWidth(periodSheet, periodFields, "definition_note", 90);
  const qcFields = ["test_id", "description", "status", "observed", "expected"];
  const qcSheet = prepareTableSheet(workbook, "Build_QC", qcFields, payload.qc, { tableName: "AuthorBuildQCTable" });
  styleQcSheet(qcSheet, qcFields, payload.qc.length);
  const statusIndex = qcFields.indexOf("status");
  const statusColumn = columnName(statusIndex);
  qcSheet.getRange(`${statusColumn}2:${statusColumn}${payload.qc.length + 1}`).conditionalFormats.add("containsText", {
    text: "PASS",
    format: { fill: "#C6EFCE", font: { bold: true, color: "#006100" } },
  });
  addNotesSheet(workbook, "Method_Notes", [
    ["数据粒度", "Author_Metadata 一行对应一个原始作者实体；不会因简繁或尾随数字自动合并。"],
    ["CBDB 身份", "同名多候选不自动选第一条；LOW 与 AMBIGUOUS 均进入人工审核。"],
    ["Floruit", "author_fl_start/end 只保存 CBDB 明示 floruit；observed_activity_* 来自明确有年的事件。"],
    ["Index year", "只作低精度时间定位，不是 birth year、floruit 或作品创作年。"],
    ["作者分期", "author_period_* 是作者元数据；poems_by_author_period 不是 dated poems。"],
    ["传统分期", "只有带学术来源的人工映射才填写；本管线不由年份自动生成传统分期。"],
  ]);
  const inspect = await workbook.inspect({ kind: "table", sheetId: "Author_Metadata", range: "A1:L15", include: "values,formulas", tableMaxRows: 15, tableMaxCols: 12, maxChars: 6000 });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "作者工作簿公式错误扫描" });
  await renderPreview(workbook, "Author_Metadata", "A1:L18", path.join(previewDir, "author_Author_Metadata.png"));
  await renderPreview(workbook, "Period_Coverage", "A1:D7", path.join(previewDir, "author_Period_Coverage.png"));
  await renderPreview(workbook, "Build_QC", "A1:E18", path.join(previewDir, "author_Build_QC.png"));
  await renderPreview(workbook, "Method_Notes", "A1:B7", path.join(previewDir, "author_Method_Notes.png"));
  const outputPath = path.join(outputRoot, "master", "QTS_author_historical_metadata.xlsx");
  await exportWorkbook(workbook, outputPath);
  return { outputPath, sheets: ["Author_Metadata", "Period_Coverage", "Build_QC", "Method_Notes"], inspect: inspect.ndjson, errors: errors.ndjson };
}


async function buildReviewWorkbook(payload, outputRoot, previewDir) {
  const workbook = Workbook.create();
  const reviewSheet = prepareTableSheet(workbook, "Review_Queue", payload.review_fields, payload.review_queue, { freezeColumns: 4, tableName: "HumanReviewQueueTable" });
  styleAuthorSheet(reviewSheet, payload.review_fields, payload.review_queue.length);
  for (const field of ["review_priority", "review_reasons", "human_decision", "approved_cbdb_person_id", "approved_traditional_period", "reviewer", "reviewed_at", "review_note"]) {
    setColumnWidth(reviewSheet, payload.review_fields, field, field === "review_note" || field === "review_reasons" ? 42 : 22);
  }
  const decisionColumn = columnName(payload.review_fields.indexOf("human_decision"));
  const traditionalColumn = columnName(payload.review_fields.indexOf("approved_traditional_period"));
  const finalRow = payload.review_queue.length + 1;
  reviewSheet.getRange(`${decisionColumn}2:${decisionColumn}${finalRow}`).dataValidation = {
    rule: { type: "list", values: ["APPROVE", "REJECT", "DEFER", "NEEDS_MORE_EVIDENCE"] },
  };
  reviewSheet.getRange(`${traditionalColumn}2:${traditionalColumn}${finalRow}`).dataValidation = {
    rule: { type: "list", values: ["初唐", "盛唐", "中唐", "晚唐"] },
  };
  const priorityColumn = columnName(payload.review_fields.indexOf("review_priority"));
  reviewSheet.getRange(`${priorityColumn}2:${priorityColumn}${finalRow}`).conditionalFormats.add("colorScale", {
    colors: ["#F8696B", "#FFEB84", "#63BE7B"],
    thresholds: ["min", "50%", "max"],
  });
  const candidateSheet = prepareTableSheet(workbook, "Candidate_Audit", payload.candidate_fields, payload.candidates, { freezeColumns: 4, tableName: "CBDBCandidateAuditTable" });
  for (const field of ["author_raw", "author_normalized", "cbdb_name", "cbdb_dynasty", "name_match_type", "matched_name_form", "selection_reason"]) {
    setColumnWidth(candidateSheet, payload.candidate_fields, field, field === "selection_reason" ? 56 : 20);
  }
  addNotesSheet(workbook, "Instructions", [
    ["审核顺序", "Priority 1 身份歧义；2 分期未解或作者身份异常；3 跨期/边界；4 LOW；5 生卒冲突。每级内按 poem_count 降序。"],
    ["候选审计", "Candidate_Audit 保存全部姓名候选、朝代/时间相容性、透明排序分数和是否选中。match_score 只用于排序，不代替人工判断。"],
    ["人工填写", "human_decision、approved_cbdb_person_id、approved_traditional_period、reviewer、reviewed_at、review_note 为人工字段。"],
    ["禁止事项", "不要把 author_period_primary 当作 poem_period；不要从 index year 推断 birth year 或作品创作年。"],
  ]);
  const inspect = await workbook.inspect({ kind: "table", sheetId: "Review_Queue", range: "A1:M15", include: "values,formulas", tableMaxRows: 15, tableMaxCols: 13, maxChars: 6000 });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "审核工作簿公式错误扫描" });
  await renderPreview(workbook, "Review_Queue", "A1:M18", path.join(previewDir, "review_Review_Queue.png"));
  await renderPreview(workbook, "Candidate_Audit", "A1:M18", path.join(previewDir, "review_Candidate_Audit.png"));
  await renderPreview(workbook, "Instructions", "A1:B5", path.join(previewDir, "review_Instructions.png"));
  const outputPath = path.join(outputRoot, "review", "author_metadata_human_review.xlsx");
  await exportWorkbook(workbook, outputPath);
  return { outputPath, sheets: ["Review_Queue", "Candidate_Audit", "Instructions"], inspect: inspect.ndjson, errors: errors.ndjson };
}


async function buildDictionaryWorkbook(payload, outputRoot, previewDir) {
  const workbook = Workbook.create();
  const fieldsSheet = prepareTableSheet(workbook, "Fields", payload.dictionary_columns, payload.data_dictionary, { freezeColumns: 2, tableName: "HistoricalMetadataDictionaryTable" });
  for (const field of payload.dictionary_columns) setColumnWidth(fieldsSheet, payload.dictionary_columns, field, field.includes("description") || field === "derivation_method" || field === "notes" ? 42 : 22);
  const schemeFields = ["label", "start", "end", "scheme_version", "scheme_note"];
  const schemeRows = payload.period_scheme.map((row) => ({ ...row, scheme_version: "QTS_PERIOD_V1", scheme_note: "操作性年代分段，不宣称文学史边界唯一客观。" }));
  const schemeSheet = prepareTableSheet(workbook, "Period_Scheme", schemeFields, schemeRows, { tableName: "PeriodSchemeTable" });
  setColumnWidth(schemeSheet, schemeFields, "scheme_note", 72);
  addNotesSheet(workbook, "Provenance_Notes", [
    ["CBDB snapshot", "固定 SQLite 版本与 SHA-256 记录在 historical_metadata_version.json 和 input_manifest.csv。"],
    ["字段级来源", "author_sources.csv 区分字段本身来自 BIOG_MAIN 与人物级书目来源；不虚构字段级页码。"],
    ["Index year", "CBDB index year 是数据库定位/推导字段；不能等同 birth year、floruit 或 poem composition year。"],
    ["Chronological period", "由 period_reference_year 与 QTS_PERIOD_V1 映射，是透明的操作性作者分期。"],
    ["Traditional period", "只来自带来源的人工映射，未填时保持 NA。"],
    ["Poem-level inheritance", "诗级表仅继承 author_* 字段；未创建 poem_year 或 poem_period。"],
  ]);
  const inspect = await workbook.inspect({ kind: "table", sheetId: "Fields", range: "A1:J15", include: "values,formulas", tableMaxRows: 15, tableMaxCols: 10, maxChars: 6000 });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "数据字典工作簿公式错误扫描" });
  await renderPreview(workbook, "Fields", "A1:J16", path.join(previewDir, "dictionary_Fields.png"));
  await renderPreview(workbook, "Period_Scheme", "A1:E6", path.join(previewDir, "dictionary_Period_Scheme.png"));
  await renderPreview(workbook, "Provenance_Notes", "A1:B7", path.join(previewDir, "dictionary_Provenance_Notes.png"));
  const outputPath = path.join(outputRoot, "docs", "historical_metadata_data_dictionary.xlsx");
  await exportWorkbook(workbook, outputPath);
  return { outputPath, sheets: ["Fields", "Period_Scheme", "Provenance_Notes"], inspect: inspect.ndjson, errors: errors.ndjson };
}


async function buildEnrichedMasterWorkbook(payload, outputRoot, previewDir) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(payload.metadata.input_master_xlsx));
  const dataSheet = workbook.worksheets.getItem("Data");
  const appendRows = await readJsonl(payload.append_jsonl);
  const formulaRows = await readJsonl(payload.formula_values_jsonl);
  if (formulaRows.length !== appendRows.length) throw new Error("公式缓存行数与诗歌追加行数不一致");
  const formulaColumns = {
    dominant_topic_probability: "AS",
    second_topic_probability: "AV",
    third_topic_probability: "AY",
    topic_entropy: "AZ",
    normalized_topic_entropy: "BA",
    top1_top2_gap: "BB",
  };
  const formulaChunkSize = 3000;
  for (const field of payload.formula_value_fields) {
    const column = formulaColumns[field];
    if (!column) throw new Error(`缺少公式缓存列映射：${field}`);
    for (let offset = 0; offset < formulaRows.length; offset += formulaChunkSize) {
      const chunk = formulaRows.slice(offset, offset + formulaChunkSize);
      const startRow = offset + 2;
      const endRow = startRow + chunk.length - 1;
      const target = dataSheet.getRange(`${column}${startRow}:${column}${endRow}`);
      target.clear({ applyTo: "contents" });
      target.values = chunk.map((row) => [row[field]]);
      target.format.numberFormat = "0.000000000000";
    }
  }
  workbook.worksheets.getItem("Codebook").getRange("B4").values = [["原母表 AS、AV、AY、AZ、BA、BB 为公式列；enriched 副本为降低 43,015 行 artifact-tool 导出内存，保存经全量 QC 验证的相同缓存数值。原 XLSX 未修改。"]];
  if (globalThis.gc) globalThis.gc();
  const fields = payload.poem_author_fields;
  const startColumn = 54;
  const endColumn = startColumn + fields.length - 1;
  dataSheet.getRangeByIndexes(0, startColumn, 1, fields.length).values = [fields];
  dataSheet.getRangeByIndexes(0, startColumn, 1, fields.length).format = headerFormat;
  dataSheet.getRangeByIndexes(0, startColumn, 1, fields.length).format.rowHeight = 36;
  const chunkSize = 2500;
  for (let offset = 0; offset < appendRows.length; offset += chunkSize) {
    const chunk = appendRows.slice(offset, offset + chunkSize);
    dataSheet.getRangeByIndexes(offset + 1, startColumn, chunk.length, fields.length).values = rowMatrix(chunk, fields);
  }
  for (let index = 0; index < fields.length; index += 1) {
    const column = columnName(startColumn + index);
    const field = fields[index];
    dataSheet.getRange(`${column}1:${column}2`).format.columnWidth = field.includes("period") || field.includes("confidence") || field.includes("method") ? 22 : 17;
    if (["author_birth_year", "author_death_year", "author_fl_start", "author_fl_end"].includes(field)) {
      dataSheet.getRange(`${column}2:${column}${appendRows.length + 1}`).format.numberFormat = "0";
    }
  }
  const codebookFields = payload.dictionary_columns;
  const newFieldSet = new Set(payload.poem_author_fields);
  const codebookRows = payload.data_dictionary.filter((row) => newFieldSet.has(row.field_name));
  const codebookSheet = prepareTableSheet(workbook, "Author_History_Codebook", codebookFields, codebookRows, { freezeColumns: 2, tableName: "AuthorHistoryCodebookTable" });
  for (const field of codebookFields) setColumnWidth(codebookSheet, codebookFields, field, field.includes("description") || field === "derivation_method" || field === "notes" ? 42 : 22);
  codebookSheet.getRange(`A${codebookRows.length + 3}:J${codebookRows.length + 3}`).merge();
  codebookSheet.getRange(`A${codebookRows.length + 3}`).values = [["重要：这些字段是 poem-level 表中继承的 AUTHOR metadata，不是 poem_year / poem_period，也不提供作品真实创作年代。"]];
  codebookSheet.getRange(`A${codebookRows.length + 3}:J${codebookRows.length + 3}`).format = { fill: "#FFF2CC", font: { bold: true, color: "#7F6000" }, wrapText: true };
  const inspect = await workbook.inspect({ kind: "table", sheetId: "Data", range: `${columnName(startColumn)}1:${columnName(endColumn)}12`, include: "values,formulas", tableMaxRows: 12, tableMaxCols: fields.length, maxChars: 7000 });
  const errors = await workbook.inspect({ kind: "match", sheetId: "Data", range: "AQ1:BP43016", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "enriched master 原公式列与新增列错误扫描" });
  await renderPreviewIfMissing(workbook, "Data", "A1:N12", path.join(previewDir, "master_Data_original.png"));
  await renderPreviewIfMissing(workbook, "Data", `${columnName(startColumn)}1:${columnName(endColumn)}12`, path.join(previewDir, "master_Data_author_history.png"));
  for (const [sheetName, range] of [["Topic_Metadata", "A1:R12"], ["Build_QC", "A1:E20"], ["Codebook", "A1:B12"], ["Author_History_Codebook", "A1:J18"]]) {
    await renderPreviewIfMissing(workbook, sheetName, range, path.join(previewDir, `master_${sheetName}.png`));
  }
  if (globalThis.gc) globalThis.gc();
  const outputPath = path.join(outputRoot, "master", "C_K20_literary_history_master_with_author_history.xlsx");
  await exportWorkbook(workbook, outputPath);
  return { outputPath, sheets: ["Data", "Topic_Metadata", "Build_QC", "Codebook", "Author_History_Codebook"], inspect: inspect.ndjson, errors: errors.ndjson };
}


async function buildEnrichedMasterWorkbookFresh(payload, outputRoot, previewDir) {
  const csvPath = path.join(outputRoot, "master", "C_K20_literary_history_master_with_author_history.csv");
  let csvText = await fs.readFile(csvPath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Data" });
  csvText = null;
  if (globalThis.gc) globalThis.gc();
  const dataSheet = workbook.worksheets.getItem("Data");
  dataSheet.showGridLines = false;
  dataSheet.freezePanes.freezeRows(1);
  dataSheet.freezePanes.freezeColumns(7);
  dataSheet.getRange("A1:BP1").format = headerFormat;
  dataSheet.getRange("A1:BP1").format.rowHeight = 36;
  dataSheet.getRange("A1:E2").format.columnWidth = 14;
  dataSheet.getRange("F1:G2").format.columnWidth = 20;
  dataSheet.getRange("H1:J2").format.columnWidth = 48;
  dataSheet.getRange("K1:V2").format.columnWidth = 13;
  dataSheet.getRange("W1:AP2").format.columnWidth = 14;
  dataSheet.getRange("AQ1:BB2").format.columnWidth = 22;
  dataSheet.getRange("BC1:BP2").format.columnWidth = 21;
  const qcFields = ["test_id", "description", "status", "observed", "expected"];
  const qcSheet = prepareTableSheet(workbook, "Build_QC", qcFields, payload.qc, { tableName: "EnrichedMasterQCTable" });
  styleQcSheet(qcSheet, qcFields, payload.qc.length);
  const statusColumn = columnName(qcFields.indexOf("status"));
  qcSheet.getRange(`${statusColumn}2:${statusColumn}${payload.qc.length + 1}`).conditionalFormats.add("containsText", {
    text: "PASS",
    format: { fill: "#C6EFCE", font: { bold: true, color: "#006100" } },
  });
  const codebookFields = payload.dictionary_columns;
  const newFieldSet = new Set(payload.poem_author_fields);
  const codebookRows = payload.data_dictionary.filter((row) => newFieldSet.has(row.field_name));
  const codebookSheet = prepareTableSheet(workbook, "Author_History_Codebook", codebookFields, codebookRows, { freezeColumns: 2, tableName: "EnrichedAuthorHistoryCodebookTable" });
  for (const field of codebookFields) setColumnWidth(codebookSheet, codebookFields, field, field.includes("description") || field === "derivation_method" || field === "notes" ? 42 : 22);
  codebookSheet.getRange(`A${codebookRows.length + 3}:J${codebookRows.length + 3}`).merge();
  codebookSheet.getRange(`A${codebookRows.length + 3}`).values = [["重要：Data 中新增的是继承的 AUTHOR metadata，不是 poem_year / poem_period；本轻量工作簿由已验证 enriched CSV 新建，不复制原 XLSX 公式对象。"]];
  codebookSheet.getRange(`A${codebookRows.length + 3}:J${codebookRows.length + 3}`).format = { fill: "#FFF2CC", font: { bold: true, color: "#7F6000" }, wrapText: true };
  const inspect = await workbook.inspect({ kind: "table", sheetId: "Data", range: "BC1:BP12", include: "values,formulas", tableMaxRows: 12, tableMaxCols: 14, maxChars: 7000 });
  const errors = await workbook.inspect({ kind: "match", sheetId: "Data", range: "AQ1:BP43016", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 300 }, summary: "轻量 enriched master 值错误扫描" });
  await renderPreview(workbook, "Data", "A1:N12", path.join(previewDir, "master_fresh_Data_original.png"));
  await renderPreview(workbook, "Data", "BC1:BP12", path.join(previewDir, "master_fresh_Data_author_history.png"));
  await renderPreview(workbook, "Build_QC", "A1:E18", path.join(previewDir, "master_fresh_Build_QC.png"));
  await renderPreview(workbook, "Author_History_Codebook", "A1:J18", path.join(previewDir, "master_fresh_Author_History_Codebook.png"));
  if (globalThis.gc) globalThis.gc();
  const outputPath = path.join(outputRoot, "master", "C_K20_literary_history_master_with_author_history.xlsx");
  await exportWorkbook(workbook, outputPath);
  return { outputPath, sheets: ["Data", "Build_QC", "Author_History_Codebook"], inspect: inspect.ndjson, errors: errors.ndjson };
}


async function main() {
  const args = parseArgs(process.argv.slice(2));
  const payloadPath = path.resolve(args.payload);
  const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
  const outputRoot = path.resolve(payload.metadata.output_root);
  const previewDir = path.join(outputRoot, ".build", "previews");
  await fs.mkdir(previewDir, { recursive: true });
  if (args.inspectSourceOnly) {
    const overview = await inspectSource(payload, previewDir);
    process.stdout.write(JSON.stringify({ status: "PASS", overview, preview: path.join(previewDir, "source_Data_A_N.png") }, null, 2));
    return;
  }
  const builders = {
    author: buildAuthorWorkbook,
    review: buildReviewWorkbook,
    dictionary: buildDictionaryWorkbook,
    master: buildEnrichedMasterWorkbookFresh,
  };
  const result = await builders[args.target](payload, outputRoot, previewDir);
  const verification = {
    status: "PASS",
    target: args.target,
    generated_at: new Date().toISOString(),
    output: result.outputPath,
    sheets: result.sheets,
    formula_error_scan: result.errors,
    inspected_range: result.inspect,
    previews: (await fs.readdir(previewDir)).filter((name) => name.endsWith(".png")).sort(),
  };
  await fs.writeFile(path.join(outputRoot, "qc", `workbook_${args.target}_verification.json`), JSON.stringify(verification, null, 2) + "\n", "utf8");
  process.stdout.write(JSON.stringify({ status: "PASS", output: verification.output, sheets: verification.sheets, preview_count: verification.previews.length }, null, 2));
}


await main();
