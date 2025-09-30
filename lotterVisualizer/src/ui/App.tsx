import React, { useEffect } from 'react';
import Dashboard from './Dashboard';
import { useStore } from '../store';
import drawsData from '../data/draws.json';

const App: React.FC = () => {
  const setDraws = useStore((state) => state.setDraws);

  useEffect(() => {
    setDraws(drawsData);
  }, [setDraws]);

  return (
    <div className="app">
      <header>
        <h1>ApexScoop Picks</h1>
      </header>
      <main>
        <Dashboard />
      </main>
    </div>
  );
};

export default App;
