// src/utils/parseData.js
import fs from 'fs';
import path from 'path';

export const parsePowerballData = () => {
  const filePath = path.join(process.cwd(), 'powerballdata.md');
  const content = fs.readFileSync(filePath, 'utf-8');

  const lines = content.split('\n').slice(1); // Skip header
  const draws = [];

  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split(',');
    if (parts.length >= 2) {
      const whiteBalls = parts[1].split('|').map(num => parseInt(num.trim(), 10));
      if (whiteBalls.length === 5 && whiteBalls.every(num => !isNaN(num))) {
        draws.push(whiteBalls.sort((a, b) => a - b));
      }
    }
  }

  return draws;
};
