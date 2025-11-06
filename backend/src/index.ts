/**
 * Backend Entrypoint - selects real or mock server based on USE_MOCK
 */

if (String(process.env.USE_MOCK || '').toLowerCase() === 'true') {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('./server-mock');
} else {
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  require('./server');
}


