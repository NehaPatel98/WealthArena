module.exports = {
  root: true,
  extends: '@react-native',
  ignorePatterns: [
    'node_modules/**/*',
    'android/**/*',
    'ios/**/*',
    '*.config.js',
    '**/node_modules/**',
    '**/Pods/**',
  ],
  env: {
    'react-native/react-native': true,
  },
  rules: {
    // Disable TypeScript-specific rules for JS files
    '@typescript-eslint/no-unused-vars': 'off',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/no-explicit-any': 'off',
    'react-native/no-unused-styles': 'warn',
    'react-native/split-platform-components': 'warn',
    'react-native/no-inline-styles': 'warn',
    'react-native/no-color-literals': 'warn',
  },
  overrides: [
    {
      files: ['**/*.js'],
      rules: {
        '@typescript-eslint/no-var-requires': 'off',
        '@typescript-eslint/explicit-module-boundary-types': 'off',
      },
    },
  ],
};
