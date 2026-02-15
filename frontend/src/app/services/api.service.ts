import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  private readonly baseUrl = 'http://localhost:8000/api';

  constructor(private readonly http: HttpClient) {}

  // ---- Upload ----
  uploadFile(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post(`${this.baseUrl}/upload`, formData);
  }

  // ---- Projects ----
  getProjects(): Observable<any[]> {
    return this.http.get<any[]>(`${this.baseUrl}/projects`);
  }

  getProject(projectId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/projects/${projectId}`);
  }

  // ---- Classify (project-based) ----
  trainModel(projectId: string, targetColumn: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/projects/${projectId}/classify`, null, {
      params: { target_column: targetColumn },
    });
  }

  // ---- Rules (project-based) ----
  getRules(projectId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/projects/${projectId}/rules`);
  }

  // ---- Add / Delete Rules ----
  addRule(projectId: string, rule: { conditions: string; predicted_class: string }): Observable<any> {
    return this.http.post(`${this.baseUrl}/projects/${projectId}/rules`, rule);
  }

  deleteRule(projectId: string, ruleId: number): Observable<any> {
    return this.http.delete(`${this.baseUrl}/projects/${projectId}/rules/${ruleId}`);
  }

  // ---- Recalculate ----
  recalculate(projectId: string): Observable<any> {
    return this.http.post(`${this.baseUrl}/projects/${projectId}/recalculate`, null);
  }

  // ---- Confusion Matrix (project-based) ----
  getConfusionMatrix(projectId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/projects/${projectId}/confusion-matrix`);
  }

  // ---- Chat (AI) ----
  chat(projectId: string, messages: { role: string; content: string }[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/projects/${projectId}/chat`, {
      project_id: projectId,
      messages,
    });
  }
}