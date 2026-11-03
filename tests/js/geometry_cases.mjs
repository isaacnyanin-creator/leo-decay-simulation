// Evaluates app/components/orbit3d/geometry.js on cases from stdin (JSON), prints JSON results.
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

const [, , modulePath] = process.argv;
const G = await import(pathToFileURL(modulePath).href);
const cases = JSON.parse(readFileSync(0, 'utf8'));
const out = cases.map(([fn, args]) => G[fn](...args));
process.stdout.write(JSON.stringify(out));
