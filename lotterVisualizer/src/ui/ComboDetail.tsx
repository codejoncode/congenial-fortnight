import React from 'react';
import { useStore } from '../store';

interface ComboDetailProps {
  combo: number[];
  onClose: () => void;
}

const ComboDetail: React.FC<ComboDetailProps> = ({ combo, onClose }) => {
  const combos = useStore((state) => state.combos);
  const comboData = combos.find(c => c.combo.join(',') === combo.join(','));

  return (
    <div className="combo-detail">
      <h3>Combo Detail: {combo.join('-')}</h3>
      <button onClick={onClose}>Close</button>
      {comboData && (
        <div>
          <p>Sum: {comboData.sum}</p>
          <p>Parity: {comboData.parity.evens}E-{comboData.parity.odds}O</p>
          <p>Lift: {comboData.lift ? `${comboData.lift.toFixed(1)}%` : 'N/A'}</p>
          <p>Heat Score: {comboData.heatScore || 0}</p>
          <p>Reasons: {comboData.reasons.join(', ')}</p>
        </div>
      )}
    </div>
  );
};

export default ComboDetail;
