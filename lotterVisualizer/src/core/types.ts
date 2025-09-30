// src/core/types.ts
export interface ComboExplanation {
  combo: number[];
  reasons: string[];
  sum: number;
  parity: { evens: number; odds: number };
  regimeMatch?: number; // center used
  keysMatched: number[];

  history: {
    testedDraws: number;
    hits: number;
    hitRate: number;       // hits / testedDraws
    avgResidue: number;    // average pool size when combo was “eligible”
    lastHitIndex?: number; // index in history; 0 = most recent
    drawsOut?: number;     // same as lastHitIndex for convenience
  };

  baseline?: {
    hitRate: number;
    avgResidue: number;
  };

  lift?: number; // % over baseline hit rate

  recurrence?: {
    typeId: string;
    pRepeat: number;   // P(hit next | hit now) for the type
    pRebound: number;  // P(hit next | miss now) for the type
    passed: boolean;
    reason: string;
  }[];

  // Scoring overlays
  confidence?: number;        // base confidence (lift + recurrence + regime)
  baseConfidence?: number;    // stored before momentum/freshness
  totalPoints?: number;       // sum of per-number points post-Q4
  heatScore?: number;         // momentum-adjusted, freshness-weighted confidence
  momentumDelta?: number;     // change in confidence over recent draws
}

export interface TypeStats {
  typeId: string;
  hitPct: number;
  repeatPct: number;
  avgGap: number;
  gapDist: Record<number, number>;
  maxGap: number;
  streakPct: number;
  pRepeat: number;
  pRebound: number;
}

export interface FilterConfig {
  keys?: number[];
  minSum: number;
  maxSum: number;
  evenCount?: number;
  oddCount?: number;
  regime?: { center: number; tol: number };
  recurrenceRules?: {
    types: { id: string; fn: (draw: number[]) => boolean }[];
    minRepeatProb?: number;
    minReboundProb?: number;
    mode?: 'excludeLowRepeat' | 'includeHighRebound' | 'both';
  };
  minHeat?: number;
  minPoints?: number;
  regimeMatch?: boolean;
  positiveMomentum?: boolean;
  passesRecurrence?: boolean;
  sortBy?: string;
}

export interface ComboMomentumPoint {
  drawIndex: number;
  confidence: number;
  totalPoints: number;
}

export interface ComboMomentumTrail {
  comboKey: string;
  trail: ComboMomentumPoint[];
}
