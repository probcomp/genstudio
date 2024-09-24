import esbuild from 'esbuild'
import cssModulesPlugin from "esbuild-css-modules-plugin"
import * as importMap from "esbuild-plugin-import-map";

const args = process.argv.slice(2);
const watch = args.includes('--watch');

const options = {
  entryPoints: ['src/genstudio/js/widget.js'],
  bundle: true,
  format: 'esm',
  outfile: 'src/genstudio/js/widget_build.js',
  plugins: [cssModulesPlugin()],
};

// currently threejs cannot load from esm.sh due to
// incompatible nested dependencies (@react-three/drei => zustand => react 18.2)
if (watch && false) {
  importMap.load('src/genstudio/js/import-map.cdn.json');
  options.plugins.push(importMap.plugin());
}

const build = watch ? esbuild.context(options).then((r) => r.watch()) : esbuild.build(options)

build.catch(() => process.exit(1))
