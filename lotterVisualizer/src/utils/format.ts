// src/utils/format.ts
export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const plus = (v: number, digits = 0) => `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`;
