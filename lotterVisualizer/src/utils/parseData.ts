// src/utils/parseData.ts
import fs from 'fs';
import path from 'path';

export function parsePowerballData(): number[][] {
  const filePath = path.join(path.dirname(new URL(import.meta.url).pathname), '../data/powerballdata.md');
  const data = fs.readFileSync(filePath, 'utf-8');
  const lines = data.split('\n').slice(1); // Skip header
  const draws: number[][] = [];

  for (const line of lines) {
    if (!line.trim()) continue;
    const parts = line.split(',');
    if (parts.length < 2) continue;
    const whiteBalls = parts[1].split('|').map((n: string) => parseInt(n.trim())).sort((a: number, b: number) => a - b);
    if (whiteBalls.length === 5) {
      draws.push(whiteBalls);
    }
  }

  return draws;
}
