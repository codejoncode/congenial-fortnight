import { parsePowerballData } from './src/utils/parseData.js';
import fs from 'fs';

const draws = parsePowerballData();
fs.writeFileSync('./src/data/draws.json', JSON.stringify(draws, null, 2));
