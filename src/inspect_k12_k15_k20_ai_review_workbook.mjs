import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { FileBlob, SpreadsheetFile } from "file:///C:/Users/ASUS%20E23688/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const root = process.cwd();
const taskRoot = path.join(root, "model_selection", "K12_K15_K20_human_adjudication");
const workbookPath = path.join(taskRoot, "QTS_K12_K15_K20_Human_Review.xlsx");
const auditDir = path.join(taskRoot, "AI_review", "audit", "pre_edit");
await fs.mkdir(auditDir, { recursive: true });

const bytes = await fs.readFile(workbookPath);
const originalSha256 = crypto.createHash("sha256").update(bytes).digest("hex");
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const inspection = await workbook.inspect({
  kind: "workbook,sheet,table,formula",
  maxChars: 12000,
  tableMaxRows: 4,
  tableMaxCols: 8,
  tableMaxCellChars: 100,
});
await fs.writeFile(path.join(auditDir, "workbook_inspection.json"), `${JSON.stringify(inspection, null, 2)}\n`, "utf8");

const previewRanges = {
  Instructions: "A1:F27",
  Blind_Topic_Review: "A1:Q12",
  Blind_Poem_Evidence: "A1:H12",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 0.8, format: "png" });
  await fs.writeFile(path.join(auditDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

console.log(JSON.stringify({
  original_sha256: originalSha256,
  sheet_count: workbook.worksheets.items.length,
  rendered_sheets: Object.keys(previewRanges),
}, null, 2));
