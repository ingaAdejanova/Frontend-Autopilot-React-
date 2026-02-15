#!/usr/bin/env node
const fs = require("fs");
const path = require("path");

function readInput() {
  try {
    const raw = fs.readFileSync(0, "utf8");
    return JSON.parse(raw || "{}");
  } catch (_e) {
    return {};
  }
}

function writeOutput(obj) {
  try {
    process.stdout.write(JSON.stringify(obj));
  } catch (_e) {
    process.stdout.write(JSON.stringify({ spans: [] }));
  }
}

let ts;
try {
  // eslint-disable-next-line
  ts = require("typescript");
} catch (e) {
  writeOutput({ spans: [], warning: "typescript_not_installed" });
  process.exit(0);
}

const input = readInput();
const repoRoot = path.resolve(input.repoRoot || process.cwd());
const focusFiles = (input.files || []).map((f) => path.resolve(repoRoot, f));
const maxFiles = Number(input.maxFiles || 4000);
const maxRefs = Number(input.maxRefs || 120);
const maxDefs = Number(input.maxDefs || 80);
const maxImports = Number(input.maxImports || 80);

function safeReadFile(f) {
  try {
    return ts.sys.readFile(f) || "";
  } catch (_e) {
    return "";
  }
}

let config = { options: { allowJs: true, checkJs: false, jsx: ts.JsxEmit.ReactJSX }, fileNames: [] };
const configPath = ts.findConfigFile(repoRoot, ts.sys.fileExists, "tsconfig.json");
if (configPath) {
  try {
    const configFile = ts.readConfigFile(configPath, ts.sys.readFile);
    const parsed = ts.parseJsonConfigFileContent(configFile.config || {}, ts.sys, path.dirname(configPath));
    if (parsed) {
      config = parsed;
    }
  } catch (_e) {
    // fall back to minimal config
  }
}

let fileNames = (config.fileNames || []).map((f) => path.resolve(f));
if (!fileNames.length) {
  fileNames = focusFiles.slice();
}

const useFullProgram = fileNames.length <= maxFiles;
if (!useFullProgram) {
  fileNames = Array.from(new Set(focusFiles));
}

const servicesHost = {
  getScriptFileNames: () => fileNames,
  getScriptVersion: () => "1",
  getScriptSnapshot: (fileName) => {
    if (!ts.sys.fileExists(fileName)) {
      return undefined;
    }
    return ts.ScriptSnapshot.fromString(safeReadFile(fileName));
  },
  getCurrentDirectory: () => repoRoot,
  getCompilationSettings: () => config.options || {},
  getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
  fileExists: ts.sys.fileExists,
  readFile: ts.sys.readFile,
  readDirectory: ts.sys.readDirectory,
};

const languageService = ts.createLanguageService(servicesHost, ts.createDocumentRegistry());
const program = languageService.getProgram();

function getSourceFile(fileName) {
  if (program) {
    const sf = program.getSourceFile(fileName);
    if (sf) {
      return sf;
    }
  }
  const text = safeReadFile(fileName);
  if (!text) {
    return null;
  }
  return ts.createSourceFile(fileName, text, ts.ScriptTarget.Latest, true);
}

function toLineRange(sf, span) {
  const start = ts.getLineAndCharacterOfPosition(sf, span.start);
  const end = ts.getLineAndCharacterOfPosition(sf, span.start + span.length);
  return { start_line: start.line + 1, end_line: Math.max(start.line + 1, end.line + 1) };
}

function relPath(fileName) {
  const rel = path.relative(repoRoot, fileName).replace(/\\/g, "/");
  if (!rel || rel.startsWith("..")) {
    return "";
  }
  return rel;
}

const spans = [];
const seen = new Set();
let refCount = 0;
let defCount = 0;
let importCount = 0;

function addSpan(fileName, span, reason) {
  if (!span) return;
  const sf = getSourceFile(fileName);
  if (!sf) return;
  const rel = relPath(fileName);
  if (!rel) return;
  const r = toLineRange(sf, span);
  const key = `${rel}:${r.start_line}:${r.end_line}:${reason}`;
  if (seen.has(key)) return;
  seen.add(key);
  spans.push({ path: rel, start_line: r.start_line, end_line: r.end_line, reason: reason || "" });
}

function addHeader(fileName, reason) {
  const sf = getSourceFile(fileName);
  if (!sf) return;
  const rel = relPath(fileName);
  if (!rel) return;
  const endLine = Math.min(160, sf.getLineAndCharacterOfPosition(sf.end).line + 1);
  const key = `${rel}:1:${endLine}:${reason}`;
  if (seen.has(key)) return;
  seen.add(key);
  spans.push({ path: rel, start_line: 1, end_line: endLine, reason: reason || "" });
}

function resolveModule(moduleText, containingFile) {
  try {
    const res = ts.resolveModuleName(moduleText, containingFile, config.options || {}, ts.sys);
    if (res && res.resolvedModule && res.resolvedModule.resolvedFileName) {
      return res.resolvedModule.resolvedFileName;
    }
  } catch (_e) {
    return "";
  }
  return "";
}

function collectImports(fileName, sf) {
  const imports = new Set();
  sf.forEachChild((node) => {
    if (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) {
      if (node.moduleSpecifier && ts.isStringLiteral(node.moduleSpecifier)) {
        const mod = node.moduleSpecifier.text;
        const resolved = resolveModule(mod, fileName);
        if (resolved) {
          imports.add(resolved);
        }
      }
      if (ts.isImportDeclaration(node) && node.importClause) {
        const clause = node.importClause;
        const toDefs = [];
        if (clause.name) {
          toDefs.push(clause.name);
        }
        if (clause.namedBindings) {
          if (ts.isNamespaceImport(clause.namedBindings)) {
            toDefs.push(clause.namedBindings.name);
          } else if (ts.isNamedImports(clause.namedBindings)) {
            for (const el of clause.namedBindings.elements || []) {
              toDefs.push(el.name);
            }
          }
        }
        for (const ident of toDefs) {
          if (defCount >= maxDefs) break;
          const defs = languageService.getDefinitionAtPosition(fileName, ident.getStart());
          if (defs && defs.length) {
            for (const d of defs) {
              if (defCount >= maxDefs) break;
              addSpan(d.fileName, d.textSpan, `def of ${ident.text}`);
              defCount += 1;
            }
          }
        }
      }
    }
  });
  return Array.from(imports);
}

function collectExports(fileName, sf) {
  const exported = [];
  sf.forEachChild((node) => {
    const hasExport = (ts.getCombinedModifierFlags(node) & ts.ModifierFlags.Export) !== 0;
    if (!hasExport) return;
    if (ts.isFunctionDeclaration(node) && node.name) exported.push(node.name);
    if (ts.isClassDeclaration(node) && node.name) exported.push(node.name);
    if (ts.isInterfaceDeclaration(node) && node.name) exported.push(node.name);
    if (ts.isTypeAliasDeclaration(node) && node.name) exported.push(node.name);
    if (ts.isEnumDeclaration(node) && node.name) exported.push(node.name);
    if (ts.isVariableStatement(node)) {
      for (const d of node.declarationList.declarations || []) {
        if (d.name && ts.isIdentifier(d.name)) {
          exported.push(d.name);
        }
      }
    }
  });
  return exported;
}

const allImports = new Set();

for (const f of focusFiles) {
  if (!ts.sys.fileExists(f)) continue;
  const sf = getSourceFile(f);
  if (!sf) continue;
  const imports = collectImports(f, sf);
  for (const imp of imports) {
    if (importCount >= maxImports) break;
    if (!allImports.has(imp)) {
      allImports.add(imp);
      addHeader(imp, `imported by ${relPath(f)}`);
      importCount += 1;
    }
  }

  if (refCount >= maxRefs) continue;
  const exported = collectExports(f, sf);
  for (const name of exported) {
    if (refCount >= maxRefs) break;
    const refs = languageService.findReferences(f, name.getStart());
    if (!refs) continue;
    for (const ref of refs) {
      if (!ref || !ref.references) continue;
      for (const r of ref.references) {
        if (refCount >= maxRefs) break;
        if (r.fileName === f && r.isDefinition) continue;
        addSpan(r.fileName, r.textSpan, `ref of ${name.text}`);
        refCount += 1;
      }
    }
  }
}

writeOutput({ spans });
