const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');

/**
 * Metro configuration
 * https://reactnative.dev/docs/metro
 *
 * @type {import('@react-native/metro-config').MetroConfig}
 */
const config = {
  resolver: {
    ...getDefaultConfig(__dirname).resolver,
    alias: {
      '@react-native/normalize-colors': require.resolve('@react-native/normalize-colors'),
    },
    // Ignore TypeScript errors in node_modules
    platforms: ['ios', 'android', 'native', 'web'],
  },
  // Exclude problematic files from bundling
  watchFolders: [],
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
