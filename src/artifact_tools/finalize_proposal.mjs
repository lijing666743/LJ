import fs from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
const workspaceDir='D:/My_project_li';
const skill='C:/Users/李静/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const finalPath=workspaceDir+'/docs/presentations/proposal_revision_20261015/exports/李静_开题答辩_原风格修订版.pptx';
const candidatePath=workspaceDir+'/docs/presentations/proposal_revision_20261015/analysis/original_style_candidate.pptx';
process.env.RUNTIME_NODE_MODULES='C:/Users/李静/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
await fs.copyFile(finalPath,candidatePath);
// The finalizer writes a new output and never overwrites the candidate.
await fs.rename(finalPath,workspaceDir+'/docs/presentations/proposal_revision_20261015/analysis/original_style_precheck_'+Date.now()+'.pptx');
try {
  const result=await finalizePresentation({
    workspaceDir,candidatePath,finalPath,explicitTotalSlideCount:19,
    pythonExecutable:'C:/Users/李静/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
    integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
    layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
    layoutArgs:['--expected-slide-size-emu','12192000,6858000','--expected-slide-count','19'],
    requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],
    verifyArtifactToolImport:true,
    receiptPath:workspaceDir+'/metadata/audits/proposal_revision_20261015/finalizer_receipt_'+Date.now()+'.json'
  });
  process.stdout.write(JSON.stringify({status:'finalized',finalPath,result})+'\n');
} catch(e) {
  await fs.copyFile(candidatePath,finalPath);
  throw e;
}
