import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useStore } from '../store';
import type { ComboExplanation } from '../core/types';

interface HeatmapProps {
  onSelectCombo: (combo: number[]) => void;
}

const Heatmap: React.FC<HeatmapProps> = ({ onSelectCombo }) => {
  const combos = useStore((state) => state.combos);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || combos.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove(); // Clear previous

    const width = 400;
    const height = 300;
    const margin = { top: 20, right: 20, bottom: 40, left: 40 };

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(combos, (d: ComboExplanation) => d.totalPoints || 0) || 50])
      .range([margin.left, width - margin.right]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(combos, (d: ComboExplanation) => d.heatScore || 0) || 100])
      .range([height - margin.bottom, margin.top]);

    const colorScale = d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([d3.min(combos, (d: ComboExplanation) => d.lift || 0) || -10, d3.max(combos, (d: ComboExplanation) => d.lift || 0) || 10]);

    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(xScale));

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(yScale));

    svg.selectAll('circle')
      .data(combos.slice(0, 50))
      .enter()
      .append('circle')
      .attr('cx', (d: ComboExplanation) => xScale(d.totalPoints || 0))
      .attr('cy', (d: ComboExplanation) => yScale(d.heatScore || 0))
      .attr('r', (d: ComboExplanation) => Math.sqrt(d.history.avgResidue || 10) / 2)
      .attr('fill', (d: ComboExplanation) => colorScale(d.lift || 0))
      .attr('stroke', 'black')
      .attr('stroke-width', 1)
      .on('click', (_event: MouseEvent, d: ComboExplanation) => onSelectCombo(d.combo));

  }, [combos, onSelectCombo]);

  return (
    <div className="heatmap">
      <h3>Confidence Heatmap</h3>
      <svg ref={svgRef} width="400" height="300"></svg>
    </div>
  );
};

export default Heatmap;
