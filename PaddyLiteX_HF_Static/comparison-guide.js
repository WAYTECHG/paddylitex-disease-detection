/* Keep the workflow's settings chips aligned with the real upload controls. */
(function () {
  'use strict';

  function initializeComparisonGuide() {
    const inputSize = document.getElementById('image-size');
    const confidence = document.getElementById('confidence');
    const sizeCaption = document.getElementById('guide-input-size');
    const confidenceCaption = document.getElementById('guide-confidence');

    if (!inputSize || !confidence || !sizeCaption || !confidenceCaption) return;

    function updateSettings() {
      const size = Number(inputSize.value);
      const threshold = Number(confidence.value);
      sizeCaption.textContent = `${size} × ${size}`;
      confidenceCaption.textContent = `${Math.round(threshold * 100)}%`;
    }

    inputSize.addEventListener('change', updateSettings);
    confidence.addEventListener('input', updateSettings);
    confidence.addEventListener('change', updateSettings);
    updateSettings();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeComparisonGuide, { once: true });
  } else {
    initializeComparisonGuide();
  }
})();
