/* Additional chart views for the existing PaddyLiteX benchmark interface. */
(function () {
  'use strict';

  const metricNames = {
    precision: 'Precision',
    recall: 'Recall',
    map50: 'mAP@0.5',
    map5095: 'mAP@0.5:0.95',
  };
  const hiddenSeries = new Map();
  const $ = (id) => document.getElementById(id);
  const escape = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[character]));
  const hasScore = (value) => typeof value === 'number' && Number.isFinite(value);
  const percent = (value) => hasScore(value) ? `${value.toFixed(2)}%` : 'N/A';
  const classLabel = (name) => name.replace(/^Rice__/, '').replace(/([a-z])([A-Z])/g, '$1 $2');
  const modelName = (model) => model.label.replace(/\bDenseNet121\b/g, 'Yolov8n_DenseNet121');

  function legend(models, key, redraw) {
    if (!hiddenSeries.has(key)) hiddenSeries.set(key, new Set());
    const hidden = hiddenSeries.get(key);
    const markup = models.map((model) => `
      <button type="button" class="chart-series-toggle" data-series="${escape(model.id)}"
        aria-pressed="${!hidden.has(model.id)}" title="Show or hide ${escape(modelName(model))}">
        <i style="background:${model.color}"></i>${escape(modelName(model))}
      </button>`).join('');
    return { hidden, markup, bind(container) {
      container.querySelectorAll('[data-series]').forEach((button) => {
        button.addEventListener('click', () => {
          const id = button.dataset.series;
          if (hidden.has(id)) hidden.delete(id);
          else hidden.add(id);
          redraw();
        });
      });
    } };
  }

  function radar(axes, models, getScore, key, redraw) {
    const width = 900;
    const height = 590;
    const center = { x: 450, y: 285 };
    const radius = 195;
    const point = (index, score, padding = 0) => {
      const angle = -Math.PI / 2 + index * 2 * Math.PI / axes.length;
      const distance = radius * score / 100 + padding;
      return { x: center.x + Math.cos(angle) * distance,
        y: center.y + Math.sin(angle) * distance };
    };
    const points = (score) => axes.map((_, index) => {
      const p = point(index, score);
      return `${p.x},${p.y}`;
    }).join(' ');
    const controls = legend(models, key, redraw);
    const rings = [20, 40, 60, 80, 100].map((score) => `
      <polygon points="${points(score)}" fill="none" stroke="#d6e3ea" />
      <text x="${center.x + 7}" y="${center.y - radius * score / 100 + 14}"
        fill="#617587" font-size="12">${score}%</text>`).join('');
    const spokes = axes.map((axis, index) => {
      const edge = point(index, 100);
      const label = point(index, 100, 29);
      const anchor = Math.abs(label.x - center.x) < 15 ? 'middle'
        : label.x < center.x ? 'end' : 'start';
      const words = axis.label.split(' ');
      const lines = words.length > 3
        ? [words.slice(0, Math.ceil(words.length / 2)).join(' '), words.slice(Math.ceil(words.length / 2)).join(' ')]
        : [axis.label];
      return `
        <line x1="${center.x}" y1="${center.y}" x2="${edge.x}" y2="${edge.y}" stroke="#d6e3ea" />
        <text x="${label.x}" y="${label.y}" text-anchor="${anchor}" fill="#344d61" font-size="14">
          ${lines.map((line, i) => `<tspan x="${label.x}" dy="${i ? 17 : 0}">${i ? ' ' : ''}${escape(line)}</tspan>`).join('')}
        </text>`;
    }).join('');
    let incomplete = 0;
    const polygons = models.map((model, modelIndex) => {
      if (controls.hidden.has(model.id)) return '';
      const scores = axes.map((axis) => getScore(model, axis));
      if (!scores.every(hasScore)) { incomplete += 1; return ''; }
      const vertices = scores.map((score, i) => point(i, score));
      const dash = ['', '7 3', '2 3', '10 3 2 3'][modelIndex % 4];
      const description = `${modelName(model)}: ${axes.map((axis, i) => `${axis.label} ${percent(scores[i])}`).join('; ')}`;
      return `
        <polygon data-model-id="${escape(model.id)}" points="${vertices.map((p) => `${p.x},${p.y}`).join(' ')}"
          fill="${model.color}" fill-opacity="${models.length > 4 ? 0.025 : 0.055}"
          stroke="${model.color}" stroke-width="2.5" stroke-dasharray="${dash}">
          <title>${escape(description)}</title>
        </polygon>
        ${vertices.map((p, i) => `<circle cx="${p.x}" cy="${p.y}" r="4" fill="${model.color}" stroke="white" stroke-width="1.5"><title>${escape(modelName(model))} · ${escape(axes[i].label)}: ${percent(scores[i])}</title></circle>`).join('')}`;
    }).join('');
    return { controls, markup: `
      <div class="chart-radar">
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Radar chart, percentage scores on a shared zero to one hundred scale">
          ${rings}${spokes}${polygons}
        </svg>
      </div>
      <div class="chart-series-controls">${controls.markup}</div>
      <p class="chart-tip">Select model names to show or hide their series. All axes use 0–100%.${incomplete ? ' Models with missing scores are omitted from the radar.' : ''}</p>` };
  }

  function scoreColor(value) {
    if (!hasScore(value)) return '#edf1f5';
    // A fixed 0–100% scale keeps heatmaps comparable across metrics and studies.
    const ratio = Math.max(0, Math.min(100, value)) / 100;
    const start = [245, 248, 252];
    const end = [27, 113, 143];
    return `rgb(${start.map((channel, i) => Math.round(channel + (end[i] - channel) * ratio)).join(',')})`;
  }

  function heatmap(rowLabels, columnLabels, getScore, caption) {
    return `
      <div class="chart-table-scroll">
        <table class="score-heatmap">
          <caption>${escape(caption)}</caption>
          <thead><tr><th scope="col">${escape(rowLabels.heading)}</th>
            ${columnLabels.map((column) => `<th scope="col">${escape(column.label).replace(/_/g, '_<wbr>')}</th>`).join('')}
          </tr></thead>
          <tbody>${rowLabels.items.map((row) => `
            <tr><th scope="row">${escape(row.label)}</th>
              ${columnLabels.map((column) => {
                const value = getScore(row, column);
                return `<td style="background:${scoreColor(value)};color:${hasScore(value) && value >= 65 ? '#fff' : '#304859'}"
                  title="${escape(row.label)} · ${escape(column.label)}: ${percent(value)}">${percent(value)}</td>`;
              }).join('')}
            </tr>`).join('')}
          </tbody>
        </table>
      </div>
      <div class="heatmap-key"><span>0%</span><i></i><span>100%</span></div>`;
  }

  function dotRows(models, getScore, format, limit) {
    return models.map((model) => {
      const value = getScore(model);
      const label = hasScore(value) ? format(value) : 'N/A';
      const position = hasScore(value) ? Math.max(0, Math.min(100, value / limit * 100)) : 0;
      return `<div class="chart-dot-row" data-model-id="${escape(model.id)}">
        <div class="bar-label"><span>${escape(modelName(model))}</span><strong>${label}</strong></div>
        <div class="chart-dot-track" role="img" aria-label="${escape(modelName(model))}: ${label}">
          ${hasScore(value) ? `<i class="chart-dot" style="left:${position}%;background:${model.color}"></i>` : ''}
        </div>
      </div>`;
    }).join('');
  }

  function renderComparison(containerId, models, metric, options) {
    const container = $(containerId);
    const redraw = () => renderComparison(containerId, models, metric, options);
    if (options.type === 'radar') {
      const axes = Object.entries(metricNames).map(([id, label]) => ({ id, label }));
      const chart = radar(axes, models, (model, axis) => model[axis.id], `quality-${options.study}`, redraw);
      container.innerHTML = chart.markup;
      chart.controls.bind(container);
    } else if (options.type === 'heatmap') {
      container.innerHTML = heatmap(
        { heading: 'Model', items: models.map((model) => ({ id: model.id, label: modelName(model), model })) },
        Object.entries(metricNames).map(([id, label]) => ({ id, label })),
        (row, column) => row.model[column.id], 'Detection performance across all four metrics (%)',
      );
    } else {
      container.innerHTML = dotRows(models, (model) => model[metric], options.format, options.limit)
        + `<div class="axis"><span>0</span><span>${options.format(options.limit)}</span></div>`;
    }
  }

  function renderClasses(models, rows, metric, options) {
    const container = $('class-chart');
    const getScore = (row, model) => row.models[model.id]?.[metric];
    if (options.type === 'radar') {
      // Keep class axes in dataset order, even if bar/heatmap sorting was changed.
      const axes = [...rows].sort((a, b) => a.id - b.id)
        .map((row) => ({ id: row.id, label: `${row.id} · ${classLabel(row.name)}`, row }));
      const redraw = () => renderClasses(models, rows, metric, options);
      const chart = radar(axes, models, (model, axis) => getScore(axis.row, model), `classes-${options.study}`, redraw);
      container.innerHTML = chart.markup;
      chart.controls.bind(container);
    } else if (options.type === 'heatmap') {
      container.innerHTML = heatmap(
        { heading: 'Disease class', items: rows.map((row) => ({ label: `${row.id} · ${row.name}`, row })) },
        models.map((model) => ({ label: modelName(model), model })),
        (row, column) => getScore(row.row, column.model), `${options.label} by disease class (%)`,
      );
    } else {
      container.innerHTML = rows.map((row) => `<section class="class-dot-group" data-class-id="${row.id}">
        <h3>${row.id} · ${escape(row.name)}</h3>
        ${dotRows(models, (model) => getScore(row, model), percent, 100)}
      </section>`).join('') + '<div class="axis"><span>0%</span><span>100%</span></div>';
    }
  }

  window.PaddyLiteXCharts = { renderComparison, renderClasses };
})();
