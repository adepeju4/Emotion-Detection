/** @type {import('tailwindcss').Config} */
import typography from '@tailwindcss/typography'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        textSmall: "#202A36",
        textLarge: "#242424",
        blue: "#2663E9"
      },
    },
  },
  plugins: [typography],
} 