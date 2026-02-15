import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet, RouterLink } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [CommonModule, RouterOutlet, RouterLink],
  template: `
    <div class="min-h-screen bg-cream-100 flex flex-col">
      <!-- Header -->
      <header class="bg-white border-b border-cream-200">
        <div class="container mx-auto px-6 py-4 flex items-center justify-between">
          <a routerLink="/" class="flex items-center gap-2 no-underline">
            <div class="w-10 h-10 bg-brown-500 rounded-full flex items-center justify-center text-white font-bold">
              ML
            </div>
            <h1 class="text-2xl font-bold text-brown-700">ML Classification Tool</h1>
          </a>
          <nav class="flex gap-6 text-sm">
            <a routerLink="/" class="text-gray-600 hover:text-brown-600">Home</a>
          </nav>
        </div>
      </header>

      <div class="flex-1">
        <router-outlet></router-outlet>
      </div>
    </div>
  `,
})
export class AppComponent {}