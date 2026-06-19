/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#fff9f0',
        'bg-2': '#f5eee0',
        surface: '#fffcf5',
        'surface-2': '#f5eee0',
        'surface-3': '#ebe2ce',
        border: '#d8cdb6',
        'border-strong': '#a89c82',
        text: '#1a1a1a',
        'text-muted': '#5e564a',
        'text-dim': '#8a8170',
        accent: '#990f3d',
        'accent-soft': 'rgba(153,15,61,0.08)',
        'accent-line': 'rgba(153,15,61,0.30)',
        profit: '#1a7c4f',
        loss: '#c3352f',
        warn: '#a86b00',
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
      },
      fontFeatureSettings: {
        tnum: '"tnum"',
      },
    },
  },
  plugins: [],
}
