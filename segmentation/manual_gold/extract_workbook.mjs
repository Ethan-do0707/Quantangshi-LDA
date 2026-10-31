import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import { createHash } from "node:crypto";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";


function parseArgs(argv) {
  const result = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error("参数必须使用 --名称 值 的形式");
    }
    result[key.slice(2)] = value;
  }
  return result;
}


async function sha256(filePath) {
  const bytes = await fs.readFile(filePath);
  return createHash("sha256").update(bytes).digest("hex");
}


const args = parseArgs(process.argv.slice(2));
for (const required of ["workbook", "output", "metadata"]) {
  if (!args[required]) {
    throw new Error(`缺少 --${required}`);
  }
}

const workbookPath = path.resolve(args.workbook);
const outputPath = path.resolve(args.output);
const metadataPath = path.resolve(args.metadata);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(workbookPath));
const sheetName = "Sheet1";
const sheet = workbook.worksheets.getItem(sheetName);
const values = sheet.getUsedRange().values;
const expectedHeader = ["诗人", "篇名", "分词"];

if (!Array.isArray(values) || values.length < 2) {
  throw new Error("人工金标准主表为空");
}
if (JSON.stringify(values[0]) !== JSON.stringify(expectedHeader)) {
  throw new Error(`主表字段不符合预期：${JSON.stringify(values[0])}`);
}
if (values.some((row) => row.length !== expectedHeader.length)) {
  throw new Error("主表存在列数异常的行");
}

const records = values.slice(1).map((row, index) => {
  const [poet, title, segmentation] = row.map((value) =>
    value === null || value === undefined ? "" : String(value).trim(),
  );
  if (!poet || !title || !segmentation) {
    throw new Error(`主表第 ${index + 2} 行存在空字段`);
  }
  return { excel_row: index + 2, poet, title, segmentation };
});

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const jsonl = `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
await fs.writeFile(outputPath, jsonl, "utf8");

const metadata = {
  source_workbook: path.basename(workbookPath),
  source_workbook_sha256: await sha256(workbookPath),
  sheet: sheetName,
  range: `A1:C${values.length}`,
  header: expectedHeader,
  record_count: records.length,
  annotation_output: path.basename(outputPath),
  annotation_output_sha256: await sha256(outputPath),
};
await fs.writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify(metadata, null, 2)}\n`);
