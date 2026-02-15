import { Component, inject, input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ChatStore } from '../../store/chat.store';

@Component({
  selector: 'app-chat',
  imports: [CommonModule, FormsModule],
  template: `
    <div class="flex flex-col h-full border-l border-cream-200 bg-cream-50">
      <!-- Messages area -->
      <div class="flex-1 overflow-y-auto p-5 space-y-4" #scrollContainer>
        @if (store.messages().length <= 1) {
          <!-- Suggested prompts -->
          <div class="mt-auto space-y-3">
            @for (prompt of suggestedPrompts; track prompt) {
              <button
                (click)="useSuggestedPrompt(prompt)"
                class="block w-full text-left p-3 bg-white border border-cream-200 rounded-xl hover:border-brown-400 hover:shadow-sm transition-all text-gray-700 text-sm"
              >
                "{{ prompt }}"
              </button>
            }
          </div>
        }

        @for (msg of store.messages(); track $index) {
          <div [class]="msg.role === 'user' ? 'text-right' : 'text-left'">
            <div
              [class]="msg.role === 'user'
                ? 'inline-block bg-brown-500 text-white rounded-2xl px-4 py-2 shadow-sm max-w-[90%] text-sm'
                : 'inline-block bg-white text-gray-800 rounded-2xl px-4 py-2 border border-cream-200 shadow-sm max-w-[90%] text-sm whitespace-pre-wrap'"
            >
              {{ msg.content }}
            </div>
          </div>
        }

        @if (store.loading()) {
          <div class="text-left">
            <div class="inline-block bg-white text-gray-400 rounded-2xl px-4 py-2 border border-cream-200 shadow-sm text-sm">
              <span class="animate-pulse">Thinking…</span>
            </div>
          </div>
        }
      </div>

      <!-- Input area -->
      <div class="border-t border-cream-300 p-4 bg-white">
        <div class="flex gap-2">
          <input
            [(ngModel)]="userInput"
            (keyup.enter)="sendMessage()"
            [disabled]="store.loading()"
            placeholder="Ask about rules, add/delete rules, recalculate…"
            class="flex-1 border border-cream-300 rounded-full px-4 py-2 bg-white text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brown-400 text-sm disabled:opacity-50"
          />
          <button
            (click)="sendMessage()"
            [disabled]="store.loading()"
            class="bg-brown-500 text-white w-9 h-9 rounded-full hover:bg-brown-600 transition-colors flex items-center justify-center shadow-sm disabled:opacity-50"
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="w-4 h-4" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  `,
})
export class ChatComponent {
  readonly projectId = input('');
  readonly store = inject(ChatStore);

  userInput = '';

  suggestedPrompts = [
    'Show me the classification rules',
    'Add a rule: IF petal_length <= 2.5 THEN setosa',
    'Delete rule 1',
    'Recalculate the confusion matrix',
    'What is the current accuracy?',
  ];

  useSuggestedPrompt(prompt: string) {
    this.userInput = prompt;
    this.sendMessage();
  }

  sendMessage() {
    if (!this.userInput.trim()) return;
    this.store.sendMessage(this.projectId(), this.userInput);
    this.userInput = '';
  }
}