/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    './pages/**/*.{js,jsx}',
    './component/**/*.{js,jsx}',
    './app/**/*.{js,jsx}',
    './src/**/*.{js,jsx}',
	],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
      colors: {
        'muted-teal': '#88B8A1',
        'dark-teal': '#78A08B',
        'soft-gray': '#F0F0F0',
        'light-gray': '#E4E7EB',
        'light-beige': '#F7F5F2',
        'dark-gray': '#333333',
        'soft-red': '#D95F5F',
        'dark-red': '#C04A4A',
      }
    },
  },
  plugins: [require("tailwindcss-animate")],
}