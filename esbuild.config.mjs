import esbuild from 'esbuild'
import cssModulesPlugin from "esbuild-css-modules-plugin"
import * as importMap from "esbuild-plugin-import-map";

const args = process.argv.slice(2);
const watch = args.includes('--watch');

const options = {
  entryPoints: ['src/genstudio/js/widget.jsx'],
  bundle: true,
  format: 'esm',
  outfile: 'src/genstudio/js/widget_build.js',
  plugins: [cssModulesPlugin()],
  minify: !watch,
  sourcemap: watch,
};

const USE_CDN_IMPORTS = false //!watch
if (USE_CDN_IMPORTS) {
  importMap.load('src/genstudio/js/import-map.cdn.json');
  options.plugins.push(importMap.plugin());
}

const build = watch
  ? esbuild.context(options).then((r) => r.watch())
  : esbuild.build(options).then(() => {
      console.log('Build completed successfully');
      console.log('Output file:', options.outfile);
    });

build.catch((error) => {
  console.error('Build failed:', error);
  process.exit(1);
})
