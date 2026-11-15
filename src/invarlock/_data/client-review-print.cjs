// Local print exporter. Input is an extracted, validated client package.
const { chromium } = require(process.argv[2]);
const path = require('node:path');
const fs = require('node:fs');
const { pathToFileURL } = require('node:url');
(async () => {
  const root = path.resolve(process.argv[3]);
  const paper = process.argv[4] || 'A4';
  if (!['A4', 'Letter'].includes(paper)) throw new Error('Supported paper: A4 or Letter');
  const browser = await chromium.launch({headless:true, ...(process.env.INVARLOCK_REPORT_BROWSER_EXECUTABLE ? {executablePath:process.env.INVARLOCK_REPORT_BROWSER_EXECUTABLE} : {})});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:960}});
    const manifest = JSON.parse(fs.readFileSync(path.join(root,'package-manifest.json'),'utf8'));
    for (const document of ['review','brief']) {
      const output=path.join(root,document+'.pdf');
      if (fs.existsSync(output)) throw new Error('PDF destination already exists');
      await page.goto(pathToFileURL(path.join(root,document+'.html')).href);
      await page.setViewportSize({width:320,height:844});
      if (await page.evaluate(()=>document.documentElement.scrollWidth > innerWidth)) throw new Error(document+' overflows at 320 pixels');
      await page.setViewportSize({width:1280,height:960});
      await page.evaluate(({snapshot,evidence})=>{
        // A detached summary cannot resolve another computer's checkout paths.
        for (const anchor of document.querySelectorAll('a')) anchor.removeAttribute('href');
        const note=document.createElement('p');
        note.className='identity identity-block';
        note.style.cssText='font:7pt/1.4 ui-monospace,monospace;border-top:1px solid #d5ded8;padding-top:8px;margin-top:12px;overflow-wrap:anywhere';
        note.textContent='Forwardable summary. Companion package required for supporting records and replay. '+(document.querySelector('footer.identity-block')?'':'Evidence: '+evidence+'. Report snapshot: '+snapshot+'. ')+'Navigation labels refer to files in that package; this PDF contains no file links.';
        const identityBlock=document.querySelector('footer.identity-block');
        if (identityBlock) { note.style.cssText='font:7pt/1.3 system-ui,sans-serif;margin:6px 0 0;overflow-wrap:anywhere'; note.className='scope'; const navigation=document.querySelector('nav'); if(navigation) navigation.after(note); else document.querySelector('main').prepend(note); }
        else document.querySelector('main').append(note);
      },{snapshot:manifest.snapshot_sha256,evidence:manifest.evidence_statement_sha256});
      await page.pdf({path:output,format:paper,printBackground:paper==='A4'});
    }
    fs.writeFileSync(path.join(root,'print-runtime.json'),JSON.stringify({browser:browser.version(),node:process.version,paper,mode:'summary requiring companion package',pdf_reproducibility:'Exact hashes identify delivered bytes; browser metadata and timestamps may vary.'},null,2)+'\n',{flag:'wx'});
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});
