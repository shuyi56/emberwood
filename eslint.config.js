import js from '@eslint/js';
import globals from 'globals';

export default [
  js.configs.recommended,
  { rules: { 'no-unused-vars': 'warn', 'no-constant-condition': 'off' } },
  { files: ['src/**/*.js'], languageOptions: { globals: globals.browser } },
  { files: ['*.js', 'scripts/**/*.mjs'], languageOptions: { globals: globals.node } }
];
