/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "docs/**/*.{md,py}",
    "src/*.py",
    "src/js/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {},
  },
  plugins: [
    require('@tailwindcss/typography'),],
}
