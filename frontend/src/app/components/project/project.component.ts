import { Component, inject, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ProjectStore } from '../../store/project.store';
import { RulesTableComponent } from '../rules-table/rules-table.component';
import { ConfusionMatrixComponent } from '../confusion-matrix/confusion-matrix.component';
import { ChatComponent } from '../chat/chat.component';

@Component({
  selector: 'app-project',
  imports: [CommonModule, FormsModule, RulesTableComponent, ConfusionMatrixComponent, ChatComponent],
  template: `
    <div class="flex h-[calc(100vh-73px)]">
      <!-- LEFT: Rules table + project info -->
      <div class="flex-1 overflow-y-auto p-6 border-r border-cream-200">
        @if (store.project(); as project) {
          <!-- Project Header -->
          <div class="mb-6">
            <h2 class="text-xl font-bold text-gray-800">{{ project.name }}</h2>
            <p class="text-sm text-gray-500 mt-1">
              {{ project.filename }} · {{ project.rows }} rows · {{ project.columns.length }} columns
            </p>
          </div>

          <!-- Training controls (if not yet trained) -->
          @if (!store.isTrained()) {
            <div class="bg-white rounded-xl border border-cream-200 p-6 mb-6">
              <h3 class="text-lg font-semibold text-gray-800 mb-4">Configure Model</h3>
              <div class="flex items-end gap-4 flex-wrap">
                <div>
                  <label class="block mb-1 text-gray-700 font-medium text-sm">Target Column</label>
                  <select
                    [ngModel]="store.selectedColumn()"
                    (ngModelChange)="store.setSelectedColumn($event)"
                    class="border border-cream-300 rounded-lg px-4 py-2 bg-white text-gray-700 focus:outline-none focus:ring-2 focus:ring-brown-400"
                  >
                    @for (col of store.columns(); track col) {
                      <option [value]="col">{{ col }}</option>
                    }
                  </select>
                </div>
                <button
                  (click)="store.trainModel(projectId)"
                  [disabled]="!store.selectedColumn() || store.training()"
                  class="bg-brown-500 text-white px-6 py-2 rounded-full hover:bg-brown-600 disabled:bg-gray-300 disabled:text-gray-500 transition-colors font-medium shadow-sm"
                >
                  {{ store.training() ? 'Training…' : 'Train Model' }}
                </button>
              </div>
            </div>
          }

          <!-- Rules Table -->
          @if (store.hasRules()) {
            <div class="bg-white rounded-xl border border-cream-200 p-6 mb-6">
              <h3 class="text-lg font-semibold text-gray-800 mb-4">Classification Rules</h3>
              <app-rules-table
                [rules]="store.rules()"
                (deleteRule)="onDeleteRule($event)"
                (addRuleEvent)="onAddRule($event)"
                (recalculate)="onRecalculate()"
              ></app-rules-table>
            </div>
          }

          <!-- Confusion Matrix -->
          @if (store.hasConfusionMatrix()) {
            <div class="bg-white rounded-xl border border-cream-200 p-6">
              <h3 class="text-lg font-semibold text-gray-800 mb-4">Confusion Matrix</h3>
              <app-confusion-matrix [data]="store.confusionMatrix()!"></app-confusion-matrix>
            </div>
          }
        } @else {
          <div class="flex items-center justify-center h-full text-gray-400">Loading project…</div>
        }
      </div>

      <!-- RIGHT: Chat panel -->
      <div class="w-[420px] flex-shrink-0 bg-cream-50 flex flex-col">
        <app-chat [projectId]="projectId" class="flex flex-col h-full"></app-chat>
      </div>
    </div>
  `,
})
export class ProjectComponent implements OnInit, OnDestroy {
  readonly store = inject(ProjectStore);
  projectId = '';

  private readonly route = inject(ActivatedRoute);

  ngOnInit() {
    this.projectId = this.route.snapshot.paramMap.get('id') ?? '';
    this.store.loadProject(this.projectId);
  }

  ngOnDestroy() {
    this.store.reset();
  }

  onDeleteRule(ruleId: number) {
    this.store.deleteRule(this.projectId, ruleId);
  }

  onAddRule(rule: { conditions: string; predicted_class: string }) {
    this.store.addRule(this.projectId, rule);
  }

  onRecalculate() {
    this.store.recalculate(this.projectId);
  }
}