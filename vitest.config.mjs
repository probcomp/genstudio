import { defineConfig } from 'vitest/config'
import * as importMap from "esbuild-plugin-import-map"

// Load the npm import map (which is empty in this case)
importMap.load('src/genstudio/js/import-map.npm.json')

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}']
  },
  plugins: [
    importMap.plugin()
  ],
})
