import { Component, inject, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { ProjectsStore } from '../../store/projects.store';

@Component({
  selector: 'app-home',
  imports: [CommonModule, FormsModule, RouterLink],
  template: `
    <div class="container mx-auto px-6 py-8 max-w-3xl">
      <!-- Upload Card -->
      <div class="bg-white rounded-2xl shadow-sm border border-cream-200 p-8 mb-8">
        <h2 class="text-xl font-bold text-gray-800 mb-6">Upload CSV Dataset</h2>

        <div class="border-2 border-dashed border-cream-300 rounded-xl p-8 text-center bg-cream-50">
          <input
            type="file"
            accept=".csv"
            (change)="onFileSelected($event)"
            class="hidden"
            #fileInput
          />
          <button
            (click)="fileInput.click()"
            [disabled]="store.uploading()"
            class="bg-sky-blue-300 text-gray-700 px-6 py-3 rounded-full hover:bg-sky-blue-200 transition-colors font-medium shadow-sm disabled:opacity-50"
          >
            {{ store.uploading() ? 'Uploading…' : 'Choose CSV File' }}
          </button>
        </div>
      </div>

      <!-- Projects List -->
      @if (store.hasProjects()) {
        <div class="bg-white rounded-2xl shadow-sm border border-cream-200 p-8">
          <h2 class="text-xl font-bold text-gray-800 mb-6">Projects</h2>
          <div class="space-y-3">
            @for (p of store.projects(); track p.id) {
              <a
                [routerLink]="['/projects', p.id]"
                class="flex items-center justify-between p-4 bg-cream-50 border border-cream-200 rounded-xl hover:border-brown-400 hover:shadow-sm transition-all no-underline"
              >
                <div>
                  <div class="font-semibold text-gray-800">{{ p.name }}</div>
                  <div class="text-sm text-gray-500">{{ p.filename }} · {{ p.rows }} rows · {{ p.columns.length }} columns</div>
                </div>
                <div class="text-brown-500 text-sm font-medium">
                  @if (p.model_id) {
                    <span class="bg-green-100 text-green-700 px-3 py-1 rounded-full text-xs">Trained</span>
                  } @else {
                    <span class="bg-cream-200 text-gray-600 px-3 py-1 rounded-full text-xs">Not trained</span>
                  }
                </div>
              </a>
            }
          </div>
        </div>
      }
    </div>
  `,
})
export class HomeComponent implements OnInit {
  readonly store = inject(ProjectsStore);

  ngOnInit() {
    this.store.loadProjects();
  }

  onFileSelected(event: Event) {
    const file = (event?.target as HTMLInputElement)?.files?.[0];
    if (!file) return;
    this.store.uploadFile(file);
  }
}
