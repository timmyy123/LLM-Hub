/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0D0F12',
          800: '#16191E',
          700: '#1E232B',
          600: '#2A303C',
        },
        grok: {
          amber: '#D97706',
          orange: '#F59E0B',
          light: '#FBBF24',
        }
      }
    },
  },
  plugins: [],
}
