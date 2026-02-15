import { inject } from '@angular/core';
import {
  signalStore,
  withState,
  withMethods,
  patchState,
} from '@ngrx/signals';
import { ApiService } from '../services/api.service';
import { ProjectStore } from './project.store';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatState {
  messages: ChatMessage[];
  loading: boolean;
}

const initialState: ChatState = {
  messages: [
    {
      role: 'assistant',
      content:
        'Hi! I can help manage your classification rules. Try asking me to show, add, or delete rules, or recalculate the confusion matrix.',
    },
  ],
  loading: false,
};

export const ChatStore = signalStore(
  { providedIn: 'root' },
  withState(initialState),
  withMethods((store, api = inject(ApiService), projectStore = inject(ProjectStore)) => ({
    reset() {
      patchState(store, initialState);
    },
    addMessage(msg: ChatMessage) {
      patchState(store, { messages: [...store.messages(), msg] });
    },
    sendMessage(projectId: string, userInput: string) {
      if (!userInput.trim()) return;

      const userMsg: ChatMessage = { role: 'user', content: userInput };
      patchState(store, { messages: [...store.messages(), userMsg], loading: true });

      // Build conversation history for the API (only user/assistant messages)
      const history = store
        .messages()
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m) => ({ role: m.role, content: m.content }));

      api.chat(projectId, history).subscribe({
        next: (resp) => {
          const assistantMsg: ChatMessage = {
            role: 'assistant',
            content: resp.content || 'Done.',
          };
          patchState(store, {
            messages: [...store.messages(), assistantMsg],
            loading: false,
          });

          // Refresh UI if the AI executed tool calls
          const actions: string[] = resp.actions || [];
          if (
            actions.includes('add_rule') ||
            actions.includes('delete_rule')
          ) {
            projectStore.recalculate(projectId);
          } else if (
            actions.includes('recalculate') ||
            actions.includes('get_confusion_matrix') ||
            actions.includes('get_rules')
          ) {
            projectStore.loadRules(projectId);
            projectStore.loadConfusionMatrix(projectId);
          }
        },
        error: (err) => {
          const errorContent =
            err?.error?.detail || err?.message || 'Something went wrong.';
          patchState(store, {
            messages: [
              ...store.messages(),
              { role: 'assistant', content: `Error: ${errorContent}` },
            ],
            loading: false,
          });
        },
      });
    },
  })),
);
