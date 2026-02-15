import { computed, inject } from '@angular/core';
import { Router } from '@angular/router';
import {
  signalStore,
  withState,
  withComputed,
  withMethods,
  patchState,
} from '@ngrx/signals';
import { ApiService } from '../services/api.service';

export interface Project {
  id: string;
  name: string;
  filename: string;
  rows: number;
  columns: string[];
  model_id: string | null;
}

interface ProjectsState {
  projects: Project[];
  loading: boolean;
  uploading: boolean;
  error: string | null;
}

const initialState: ProjectsState = {
  projects: [],
  loading: false,
  uploading: false,
  error: null,
};

export const ProjectsStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withComputed((store) => ({
    hasProjects: computed(() => store.projects().length > 0),
  })),
  withMethods((store, api = inject(ApiService), router = inject(Router)) => ({
    loadProjects() {
      patchState(store, { loading: true, error: null });
      api.getProjects().subscribe({
        next: (projects) => patchState(store, { projects, loading: false }),
        error: (err) =>
          patchState(store, { loading: false, error: err.message }),
      });
    },
    uploadFile(file: File) {
      patchState(store, { uploading: true, error: null });
      api.uploadFile(file).subscribe({
        next: (response) => {
          patchState(store, { uploading: false });
          router.navigate(['/projects', response.project_id]);
        },
        error: (err) =>
          patchState(store, { uploading: false, error: err.message }),
      });
    },
  })),
);
