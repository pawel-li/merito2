import { computed, inject } from '@angular/core';
import {
  signalStore,
  withState,
  withComputed,
  withMethods,
  patchState,
} from '@ngrx/signals';
import { ApiService } from '../services/api.service';
import type { Project } from './projects.store';

export interface Rule {
  rule_id: number;
  conditions: string;
  predicted_class: string;
  coverage: number;
  precision: number;
  quality: number;
  p_value: number;
}

export interface ConfusionMatrixData {
  true_positives?: number;
  true_negatives?: number;
  false_positives?: number;
  false_negatives?: number;
  accuracy: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  matrix?: number[][];
  classes?: string[];
}

interface ProjectState {
  project: Project | null;
  selectedColumn: string;
  training: boolean;
  rules: Rule[];
  confusionMatrix: ConfusionMatrixData | null;
  loading: boolean;
  error: string | null;
}

const initialState: ProjectState = {
  project: null,
  selectedColumn: '',
  training: false,
  rules: [],
  confusionMatrix: null,
  loading: false,
  error: null,
};

export const ProjectStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withComputed((store) => ({
    isTrained: computed(() => !!store.project()?.model_id),
    hasRules: computed(() => store.rules().length > 0),
    hasConfusionMatrix: computed(() => store.confusionMatrix() !== null),
    columns: computed(() => store.project()?.columns ?? []),
  })),
  withMethods((store, api = inject(ApiService)) => ({
    reset() {
      patchState(store, initialState);
    },
    setSelectedColumn(column: string) {
      patchState(store, { selectedColumn: column });
    },
    loadProject(projectId: string) {
      patchState(store, { loading: true, error: null });
      api.getProject(projectId).subscribe({
        next: (project) => {
          const selectedColumn =
            project.columns[project.columns.length - 1] ?? '';
          patchState(store, { project, selectedColumn, loading: false });

          if (project.model_id) {
            this.loadRules(projectId);
            this.loadConfusionMatrix(projectId);
          }
        },
        error: (err) =>
          patchState(store, { loading: false, error: err.message }),
      });
    },
    trainModel(projectId: string) {
      const selectedColumn = store.selectedColumn();
      if (!selectedColumn) return;

      patchState(store, { training: true, error: null });
      api.trainModel(projectId, selectedColumn).subscribe({
        next: () => {
          patchState(store, { training: false });
          this.loadProject(projectId);
        },
        error: (err) =>
          patchState(store, { training: false, error: err.message }),
      });
    },
    loadRules(projectId: string) {
      api.getRules(projectId).subscribe({
        next: (resp) => patchState(store, { rules: resp.rules || [] }),
        error: () => patchState(store, { rules: [] }),
      });
    },
    loadConfusionMatrix(projectId: string) {
      api.getConfusionMatrix(projectId).subscribe({
        next: (data) => patchState(store, { confusionMatrix: data }),
        error: () => {},
      });
    },
    addRule(projectId: string, rule: { conditions: string; predicted_class: string }) {
      api.addRule(projectId, rule).subscribe({
        next: () => this.recalculate(projectId),
        error: (err) => patchState(store, { error: err.message }),
      });
    },
    deleteRule(projectId: string, ruleId: number) {
      api.deleteRule(projectId, ruleId).subscribe({
        next: () => this.recalculate(projectId),
        error: (err) => patchState(store, { error: err.message }),
      });
    },
    recalculate(projectId: string) {
      patchState(store, { training: true, error: null });
      api.recalculate(projectId).subscribe({
        next: () => {
          patchState(store, { training: false });
          this.loadRules(projectId);
          this.loadConfusionMatrix(projectId);
        },
        error: (err) => patchState(store, { training: false, error: err.message }),
      });
    },
  })),
);
