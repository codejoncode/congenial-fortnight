import { create } from 'zustand';

interface LotteryState {
  draws: number[][];
  setDraws: (draws: number[][]) => void;
}

export const useStore = create<LotteryState>((set) => ({
  draws: [],
  setDraws: (draws) => set({ draws }),
}));
