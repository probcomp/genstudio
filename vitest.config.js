import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    include: ['tests/**/*.test.{js,mjs,cjs,ts,mts,cts,jsx,tsx}']
  },
  resolve: {
    alias: {
      './imports': path.resolve(__dirname, 'src/genstudio/js/imports.npm.js'),
    },
  },
})
