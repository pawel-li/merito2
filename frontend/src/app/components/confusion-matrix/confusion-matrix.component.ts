import { Component, input, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { ConfusionMatrixData } from '../../store/project.store';

@Component({
  selector: 'app-confusion-matrix',
  imports: [CommonModule],
  template: `
    <div class="mt-4 p-6 bg-cream-50 rounded-xl border border-cream-300">
      @if (isBinaryMatrix(); as binaryData) {
        <div class="grid grid-cols-3 gap-4 max-w-md mx-auto">
          <div></div>
          <div class="text-center font-semibold text-gray-700 text-sm">Predicted Negative</div>
          <div class="text-center font-semibold text-gray-700 text-sm">Predicted Positive</div>

          <div class="text-right font-semibold text-gray-700 text-sm flex items-center justify-end">Actual Negative</div>
          <div class="bg-green-100 p-8 text-center rounded-xl border border-green-200">
            <div class="text-3xl font-bold text-green-700">{{ binaryData.true_negatives }}</div>
            <div class="text-xs text-gray-600 mt-1">True Negatives</div>
          </div>
          <div class="bg-red-100 p-8 text-center rounded-xl border border-red-200">
            <div class="text-3xl font-bold text-red-700">{{ binaryData.false_positives }}</div>
            <div class="text-xs text-gray-600 mt-1">False Positives</div>
          </div>

          <div class="text-right font-semibold text-gray-700 text-sm flex items-center justify-end">Actual Positive</div>
          <div class="bg-orange-100 p-8 text-center rounded-xl border border-orange-200">
            <div class="text-3xl font-bold text-orange-700">{{ binaryData.false_negatives }}</div>
            <div class="text-xs text-gray-600 mt-1">False Negatives</div>
          </div>
          <div class="bg-green-200 p-8 text-center rounded-xl border border-green-300">
            <div class="text-3xl font-bold text-green-800">{{ binaryData.true_positives }}</div>
            <div class="text-xs text-gray-600 mt-1">True Positives</div>
          </div>
        </div>

        <div class="mt-6 grid grid-cols-4 gap-4 text-center">
          <div class="bg-white p-4 rounded-xl border border-cream-200">
            <div class="font-semibold text-gray-700 text-sm">Accuracy</div>
            <div class="text-lg font-bold text-brown-600 mt-1">{{ (binaryData.accuracy * 100).toFixed(1) }}%</div>
          </div>
          <div class="bg-white p-4 rounded-xl border border-cream-200">
            <div class="font-semibold text-gray-700 text-sm">Precision</div>
            <div class="text-lg font-bold text-brown-600 mt-1">{{ ((binaryData.precision ?? 0) * 100).toFixed(1) }}%</div>
          </div>
          <div class="bg-white p-4 rounded-xl border border-cream-200">
            <div class="font-semibold text-gray-700 text-sm">Recall</div>
            <div class="text-lg font-bold text-brown-600 mt-1">{{ ((binaryData.recall ?? 0) * 100).toFixed(1) }}%</div>
          </div>
          <div class="bg-white p-4 rounded-xl border border-cream-200">
            <div class="font-semibold text-gray-700 text-sm">F1-Score</div>
            <div class="text-lg font-bold text-brown-600 mt-1">{{ ((binaryData.f1_score ?? 0) * 100).toFixed(1) }}%</div>
          </div>
        </div>
      } @else {
        <div class="text-sm text-gray-700 mb-3 font-medium">Multi-class confusion matrix</div>
        <div class="overflow-x-auto">
          <table class="min-w-full bg-white border border-cream-300 rounded-xl overflow-hidden">
            <tbody>
              @for (row of data().matrix; track $index) {
                <tr class="border-t border-cream-200">
                  @for (value of row; track $index) {
                    <td class="px-4 py-3 text-center text-gray-700 font-medium">{{ value }}</td>
                  }
                </tr>
              }
            </tbody>
          </table>
        </div>
        <div class="mt-4 text-center bg-white p-4 rounded-xl border border-cream-200">
          <span class="font-semibold text-gray-700">Accuracy:</span>
          <span class="text-lg font-bold text-brown-600 ml-2">{{ (data().accuracy * 100).toFixed(1) }}%</span>
        </div>
      }
    </div>
  `
})
export class ConfusionMatrixComponent {
  readonly data = input<ConfusionMatrixData>({
    true_positives: 0,
    true_negatives: 0,
    false_positives: 0,
    false_negatives: 0,
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1_score: 0,
    matrix: []
  });

  readonly isBinaryMatrix = computed(() => {
    const d = this.data();
    return d && d.true_positives !== undefined ? d : null;
  });
}