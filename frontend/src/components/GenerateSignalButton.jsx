import React from 'react';
import axios from 'axios';

export default function GenerateSignalButton({ onSignalGenerated }) {
  const handleClick = async () => {
    const res = await axios.post('/api/generate-signal/', { pair: 'all' });
    if (onSignalGenerated) onSignalGenerated(res.data.signals);
    alert('Signals generated!');
  };
  return <button onClick={handleClick}>🎯 Generate Daily Signal</button>;
}
