module.exports = {
  content: [
    "./src/**/*.{html,ts}",
  ],
  theme: {
    extend: {
      colors: {
        // Hashbrown-inspired color palette
        'cream': {
          50: '#FDFCFB',
          100: '#FBF6F0',
          200: '#F5EDE3',
          300: '#EFE4D6',
        },
        'brown': {
          400: '#C9A882',
          500: '#A67C52',
          600: '#8B6B47',
          700: '#6F563A',
        },
        'sky-blue': {
          100: '#E8F4F8',
          200: '#C9E4ED',
          300: '#B8D8E8',
        },
      },
    },
  },
  plugins: [],
}