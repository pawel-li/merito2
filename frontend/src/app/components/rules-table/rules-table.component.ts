import { Component, input, output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-rules-table',
  imports: [CommonModule, FormsModule],
  template: `
    <!-- Toolbar above table -->
    <div class="flex items-center justify-between mb-4 gap-3 flex-wrap">
      <button
        (click)="showAddForm = !showAddForm"
        class="bg-brown-500 text-white px-4 py-2 rounded-full hover:bg-brown-600 transition-colors font-medium text-sm shadow-sm flex items-center gap-1"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clip-rule="evenodd"/>
        </svg>
        Add Rule
      </button>
      <button
        (click)="recalculate.emit()"
        class="bg-green-600 text-white px-4 py-2 rounded-full hover:bg-green-700 transition-colors font-medium text-sm shadow-sm flex items-center gap-1"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
          <path fill-rule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clip-rule="evenodd"/>
        </svg>
        Recalculate
      </button>
    </div>

    <!-- Add Rule Form -->
    @if (showAddForm) {
      <div class="bg-cream-50 border border-cream-300 rounded-xl p-4 mb-4">
        <h4 class="text-sm font-semibold text-gray-700 mb-3">New Rule</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label class="block text-xs text-gray-600 mb-1">Conditions</label>
            <input
              [(ngModel)]="newConditions"
              placeholder="e.g. feature1 <= 2.50 AND feature2 > 1.75"
              class="w-full border border-cream-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-brown-400"
            />
          </div>
          <div>
            <label class="block text-xs text-gray-600 mb-1">Predicted Class</label>
            <input
              [(ngModel)]="newClass"
              placeholder="e.g. Iris-setosa"
              class="w-full border border-cream-300 rounded-lg px-3 py-2 text-sm bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-brown-400"
            />
          </div>
        </div>
        <div class="flex gap-2 mt-3">
          <button
            (click)="onAddRule()"
            [disabled]="!newConditions.trim() || !newClass.trim()"
            class="bg-brown-500 text-white px-4 py-1.5 rounded-full hover:bg-brown-600 disabled:bg-gray-300 disabled:text-gray-500 transition-colors text-sm font-medium"
          >
            Add
          </button>
          <button
            (click)="showAddForm = false; newConditions = ''; newClass = ''"
            class="bg-gray-200 text-gray-700 px-4 py-1.5 rounded-full hover:bg-gray-300 transition-colors text-sm"
          >
            Cancel
          </button>
        </div>
      </div>
    }

    <!-- Table -->
    <div class="overflow-x-auto">
      <table class="min-w-full bg-white border border-cream-300 rounded-xl overflow-hidden shadow-sm">
        <thead class="bg-brown-500 text-white">
          <tr>
            <th class="px-4 py-3 text-left text-sm font-semibold">#</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">Conditions</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">Class</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">Coverage</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">Precision</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">Quality</th>
            <th class="px-4 py-3 text-left text-sm font-semibold">P-Value</th>
            <th class="px-4 py-3 text-center text-sm font-semibold">Actions</th>
          </tr>
        </thead>
        <tbody>
          @for (rule of rules(); track rule.rule_id) {
            <tr class="border-t border-cream-200 hover:bg-cream-50 transition-colors">
              <td class="px-4 py-3 text-gray-700">{{ rule.rule_id }}</td>
              <td class="px-4 py-3 text-sm font-mono text-gray-600">{{ rule.conditions }}</td>
              <td class="px-4 py-3">
                <span class="bg-brown-100 text-brown-700 px-3 py-1 rounded-full text-sm font-medium">
                  {{ rule.predicted_class }}
                </span>
              </td>
              <td class="px-4 py-3 text-gray-700">{{ (rule.coverage * 100).toFixed(1) }}%</td>
              <td class="px-4 py-3 text-gray-700">{{ (rule.precision * 100).toFixed(1) }}%</td>
              <td class="px-4 py-3 text-gray-700">{{ rule.quality.toFixed(3) }}</td>
              <td class="px-4 py-3 text-gray-700" [class.text-green-600]="rule.p_value < 0.05" [class.font-semibold]="rule.p_value < 0.05">
                {{ rule.p_value.toFixed(4) }}
              </td>
              <td class="px-4 py-3 text-center">
                <button
                  (click)="deleteRule.emit(rule.rule_id)"
                  class="text-red-500 hover:text-red-700 hover:bg-red-50 p-1.5 rounded-lg transition-colors"
                  title="Delete rule"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd"/>
                  </svg>
                </button>
              </td>
            </tr>
          }
        </tbody>
      </table>
    </div>
  `
})
export class RulesTableComponent {
  readonly rules = input<any[]>([]);
  readonly deleteRule = output<number>();
  readonly recalculate = output<void>();
  readonly addRuleEvent = output<{ conditions: string; predicted_class: string }>();

  showAddForm = false;
  newConditions = '';
  newClass = '';

  onAddRule() {
    if (!this.newConditions.trim() || !this.newClass.trim()) return;
    this.addRuleEvent.emit({
      conditions: this.newConditions.trim(),
      predicted_class: this.newClass.trim(),
    });
    this.newConditions = '';
    this.newClass = '';
    this.showAddForm = false;
  }
}