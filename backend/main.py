from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uuid
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from database import (
    init_db, save_project, get_project, list_projects, update_project,
    save_csv_rows, get_csv_data, save_rules, get_rules,
    save_confusion_matrix, get_confusion_matrix as db_get_confusion_matrix,
    add_rule as db_add_rule, delete_rule as db_delete_rule,
    update_rule_metrics,
)
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="ML Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory cache for trained sklearn objects (can't serialise to SQLite)
_model_cache: dict = {}


class AddRuleRequest(BaseModel):
    conditions: str
    predicted_class: str
    coverage: float = 0.0
    precision: float = 0.0
    support: int = 0
    quality: float = 0.0
    p_value: float = 1.0
    class_distribution: Optional[dict] = None


@app.on_event("startup")
def startup():
    init_db()


def extract_rules_from_tree(tree, feature_names, class_names):
    """Extract human-readable rules from a decision tree"""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            left_conditions = conditions + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], left_conditions)
            right_conditions = conditions + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], right_conditions)
        else:
            values = tree_.value[node][0]
            total = np.sum(values)
            class_idx = np.argmax(values)
            predicted_class = class_names[class_idx]
            coverage = total / np.sum(tree_.value)
            precision = values[class_idx] / total if total > 0 else 0
            rules.append({
                'conditions': conditions,
                'predicted_class': predicted_class,
                'coverage': float(coverage),
                'precision': float(precision),
                'support': int(total),
                'class_distribution': {
                    class_names[i]: int(values[i]) for i in range(len(values))
                },
            })

    recurse(0, [])
    return rules


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@app.get("/api/projects")
def api_list_projects():
    return list_projects()


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    return project


# ---------------------------------------------------------------------------
# Upload CSV  →  create project
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files accepted")

    project_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{project_id}.csv"

    content = await file.read()
    with open(file_path, 'wb') as saved_file:
        saved_file.write(content)

    df = pd.read_csv(file_path)

    # Persist to SQLite
    project_name = file.filename.rsplit('.', 1)[0]
    save_project(project_id, project_name, file.filename, len(df), df.columns.tolist())
    save_csv_rows(project_id, df)

    return {
        "project_id": project_id,
        "file_id": project_id,
        "filename": file.filename,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "preview": df.head(5).to_dict('records'),
    }


# ---------------------------------------------------------------------------
# Train  (classify)
# ---------------------------------------------------------------------------

@app.post("/api/projects/{project_id}/classify")
def train_classifier(
    project_id: str,
    target_column: str,
    max_depth: int = 5,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
):
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    df = get_csv_data(project_id)
    if df is None:
        raise HTTPException(404, "CSV data not found")

    if target_column not in df.columns:
        raise HTTPException(400, f"Column {target_column} not found")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Coerce numeric columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna(axis=1, how='all')

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    raw_rules = extract_rules_from_tree(
        clf, X_encoded.columns.tolist(), clf.classes_.tolist()
    )

    rules_data = []
    for idx, rule in enumerate(raw_rules):
        conditions_str = (
            " AND ".join(rule['conditions']) if rule['conditions'] else "TRUE"
        )
        rules_data.append({
            "rule_id": idx + 1,
            "conditions": conditions_str,
            "predicted_class": rule['predicted_class'],
            "coverage": rule['coverage'],
            "precision": rule['precision'],
            "support": rule['support'],
            "quality": rule['coverage'] * rule['precision'],
            "p_value": 0.05,
            "class_distribution": rule['class_distribution'],
        })

    save_rules(project_id, rules_data)

    acc = float(accuracy_score(y_test, y_pred))
    classes = clf.classes_.tolist()
    extra = None
    if len(cm) == 2:
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        extra = {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }
    save_confusion_matrix(project_id, cm.tolist(), acc, classes, extra)

    model_id = str(uuid.uuid4())
    update_project(project_id, target_column=target_column, model_id=model_id)

    _model_cache[project_id] = {
        'classifier': clf,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
    }

    return {
        "model_id": model_id,
        "status": "success",
        "message": f"Model trained with {len(rules_data)} rules (leaf nodes)",
        "tree_depth": int(clf.get_depth()),
        "n_leaves": int(clf.get_n_leaves()),
    }


# Legacy endpoint
@app.post("/api/classify")
def train_classifier_legacy(
    file_id: str,
    target_column: str,
    max_depth: int = 5,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
):
    return train_classifier(file_id, target_column, max_depth, min_samples_split, min_samples_leaf)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/rules")
def api_get_rules(project_id: str):
    rules = get_rules(project_id)
    return {"rules": rules or []}


@app.post("/api/projects/{project_id}/rules")
def api_add_rule(project_id: str, rule: AddRuleRequest):
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    rule_data = rule.model_dump()
    if rule_data.get("class_distribution") is None:
        rule_data["class_distribution"] = {}
    new_id = db_add_rule(project_id, rule_data)
    return {"status": "success", "rule_id": new_id}


@app.delete("/api/projects/{project_id}/rules/{rule_id}")
def api_delete_rule(project_id: str, rule_id: int):
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    deleted = db_delete_rule(project_id, rule_id)
    if not deleted:
        raise HTTPException(404, "Rule not found")
    return {"status": "success", "message": f"Rule {rule_id} deleted"}


@app.post("/api/projects/{project_id}/recalculate")
def api_recalculate(project_id: str):
    """Recalculate confusion matrix AND per-rule metrics based on current rules applied to data."""
    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    target_column = project.get("target_column")
    if not target_column:
        raise HTTPException(400, "No target column set - train the model first")

    df = get_csv_data(project_id)
    if df is None:
        raise HTTPException(404, "CSV data not found")

    rules = get_rules(project_id)
    if not rules:
        raise HTTPException(400, "No rules to apply")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.dropna(axis=1, how='all')

    total_rows = len(X)
    classes = sorted(y.unique().tolist(), key=str)
    default_class = y.mode().iloc[0] if len(y) > 0 else classes[0]

    def _matches_rule(row, conditions_str):
        """Check if a data row matches a rule's conditions."""
        if conditions_str == "TRUE":
            return True
        try:
            parts = conditions_str.split(" AND ")
            for part in parts:
                if "<=" in part:
                    feature, val = part.split("<=")
                    if row.get(feature.strip(), float('inf')) > float(val.strip()):
                        return False
                elif ">=" in part:
                    feature, val = part.split(">=")
                    if row.get(feature.strip(), float('-inf')) < float(val.strip()):
                        return False
                elif ">" in part:
                    feature, val = part.split(">")
                    if row.get(feature.strip(), float('-inf')) <= float(val.strip()):
                        return False
                elif "<" in part:
                    feature, val = part.split("<")
                    if row.get(feature.strip(), float('inf')) >= float(val.strip()):
                        return False
                elif "=" in part:
                    feature, val = part.split("=")
                    if str(row.get(feature.strip(), '')) != val.strip():
                        return False
            return True
        except Exception:
            return False

    # --- Compute per-rule metrics ---
    for rule in rules:
        matched_indices = []
        for idx, (_, row) in enumerate(X.iterrows()):
            if _matches_rule(row, rule["conditions"]):
                matched_indices.append(idx)

        support = len(matched_indices)
        coverage = support / total_rows if total_rows > 0 else 0.0

        # Of the matched rows, how many have the predicted class?
        correct = 0
        class_dist = {str(c): 0 for c in classes}
        y_list_all = y.tolist()
        for idx in matched_indices:
            actual = str(y_list_all[idx])
            if actual in class_dist:
                class_dist[actual] += 1
            if actual == str(rule["predicted_class"]):
                correct += 1

        precision_val = correct / support if support > 0 else 0.0
        quality = coverage * precision_val

        # Simple p-value heuristic: Fisher-style based on support
        from scipy.stats import binomtest
        try:
            # Probability of seeing this many correct by chance
            n_classes = len(classes)
            expected_p = 1.0 / n_classes if n_classes > 0 else 0.5
            if support > 0:
                result = binomtest(correct, support, expected_p, alternative='greater')
                p_val = float(result.pvalue)
            else:
                p_val = 1.0
        except Exception:
            p_val = 0.05 if precision_val > 0.5 and support > 5 else 1.0

        update_rule_metrics(
            project_id, rule["rule_id"],
            coverage=float(coverage),
            precision_val=float(precision_val),
            support=support,
            quality=float(quality),
            p_value=float(p_val),
            class_distribution=class_dist,
        )

    # --- Compute confusion matrix using first-matching-rule approach ---
    predictions = []
    for _, row in X.iterrows():
        predicted = None
        for rule in rules:
            if _matches_rule(row, rule["conditions"]):
                predicted = rule["predicted_class"]
                break
        predictions.append(predicted if predicted is not None else default_class)

    y_list = y.tolist()
    cm = confusion_matrix(y_list, predictions, labels=classes)
    acc = float(accuracy_score(y_list, predictions))

    extra = None
    if len(cm) == 2:
        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        extra = {
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
        }

    str_classes = [str(c) for c in classes]
    save_confusion_matrix(project_id, cm.tolist(), acc, str_classes, extra)

    return {
        "status": "success",
        "accuracy": acc,
        "matrix_size": len(cm),
    }


@app.get("/api/rules/{model_id}")
def api_get_rules_legacy(model_id: str):
    from database import get_db
    conn = get_db()
    row = conn.execute("SELECT id FROM projects WHERE model_id = ?", (model_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Model not found")
    return api_get_rules(row["id"])


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

@app.get("/api/projects/{project_id}/confusion-matrix")
def api_get_confusion_matrix(project_id: str):
    data = db_get_confusion_matrix(project_id)
    if not data:
        raise HTTPException(404, "Confusion matrix not found - train the model first")
    return data


@app.get("/api/confusion-matrix/{model_id}")
def api_get_confusion_matrix_legacy(model_id: str):
    from database import get_db
    conn = get_db()
    row = conn.execute("SELECT id FROM projects WHERE model_id = ?", (model_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Model not found")
    return api_get_confusion_matrix(row["id"])


# ---------------------------------------------------------------------------
# Chat  (OpenRouter AI with tool-calling)
# ---------------------------------------------------------------------------
import os
import httpx
import json as _json

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

# Tools the LLM can call
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_rules",
            "description": "Retrieve the current classification rules for this project",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_rule",
            "description": "Add a new classification rule. The conditions use feature comparisons joined by AND.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "string",
                        "description": "Rule conditions, e.g. 'petal_length <= 2.50 AND sepal_width > 3.00'",
                    },
                    "predicted_class": {
                        "type": "string",
                        "description": "The class the rule predicts",
                    },
                },
                "required": ["conditions", "predicted_class"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_rule",
            "description": "Delete a classification rule by its rule_id number",
            "parameters": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "integer", "description": "The ID of the rule to delete"},
                },
                "required": ["rule_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recalculate",
            "description": "Recalculate the confusion matrix using the current set of rules",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_confusion_matrix",
            "description": "Get the confusion matrix and accuracy metrics",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


class ChatRequest(BaseModel):
    messages: list
    project_id: str


def _execute_tool(tool_name: str, arguments: dict, project_id: str) -> str:
    """Execute a tool call and return result as string."""
    try:
        if tool_name == "get_rules":
            rules = get_rules(project_id)
            return _json.dumps({"rules": rules or []}, default=str)
        elif tool_name == "add_rule":
            rule_data = {
                "conditions": arguments["conditions"],
                "predicted_class": arguments["predicted_class"],
                "coverage": 0.0, "precision": 0.0, "support": 0,
                "quality": 0.0, "p_value": 1.0, "class_distribution": {},
            }
            new_id = db_add_rule(project_id, rule_data)
            return _json.dumps({"status": "success", "rule_id": new_id})
        elif tool_name == "delete_rule":
            deleted = db_delete_rule(project_id, arguments["rule_id"])
            if deleted:
                return _json.dumps({"status": "success", "message": f"Rule {arguments['rule_id']} deleted"})
            return _json.dumps({"status": "error", "message": "Rule not found"})
        elif tool_name == "recalculate":
            result = api_recalculate(project_id)
            if "accuracy" in result:
                result["accuracy_pct"] = f"{result['accuracy'] * 100:.1f}%"
            return _json.dumps(result, default=str)
        elif tool_name == "get_confusion_matrix":
            data = db_get_confusion_matrix(project_id)
            if data and "accuracy" in data:
                data["accuracy_pct"] = f"{data['accuracy'] * 100:.1f}%"
            return _json.dumps(data or {}, default=str)
        else:
            return _json.dumps({"error": f"Unknown tool: {tool_name}"})
    except Exception as e:
        return _json.dumps({"error": str(e)})


@app.post("/api/projects/{project_id}/chat")
async def chat_endpoint(project_id: str, req: ChatRequest):
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not configured")

    project = get_project(project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    rules = get_rules(project_id)
    cm = db_get_confusion_matrix(project_id)

    # Build rich context about the dataset
    target_col = project.get('target_column', 'not set')
    feature_cols = [c for c in project['columns'] if c != target_col]

    # Get unique classes from dataset
    classes_list = []
    if cm and cm.get('classes'):
        classes_list = cm['classes']
    else:
        df = get_csv_data(project_id)
        if df is not None and target_col in df.columns:
            classes_list = sorted(df[target_col].unique().tolist(), key=str)

    # Format existing rules for context
    rules_summary = "None"
    if rules:
        rules_summary = "\n".join(
            f"  Rule {r['rule_id']}: IF {r['conditions']} THEN {r['predicted_class']}"
            for r in rules
        )

    accuracy_str = f"{cm['accuracy']*100:.1f}%" if cm else "N/A"
    system_msg = {
        "role": "system",
        "content": (
            f"You are an AI assistant for an ML classification project.\n"
            f"Project: {project['name']} ({project['rows']} rows).\n"
            f"Target column: {target_col}\n"
            f"Feature columns: {feature_cols}\n"
            f"Valid classes: {classes_list}\n"
            f"Current accuracy: {accuracy_str}\n\n"
            f"Current rules:\n{rules_summary}\n\n"
            f"IMPORTANT INSTRUCTIONS:\n"
            f"- When the user asks to add a rule, IMMEDIATELY call the add_rule tool. Do NOT validate or reject the input yourself.\n"
            f"- The user may write conditions informally like 'petal.length <= 2.45 Virginica'. Parse it as: conditions='petal.length <= 2.45', predicted_class='Virginica'.\n"
            f"- If the user gives feature name and comparison but no explicit 'THEN', treat the LAST word as the predicted class and everything before it as conditions.\n"
            f"- When asked to delete a rule, call the delete_rule tool with the rule_id.\n"
            f"- When asked to recalculate, call the recalculate tool.\n"
            f"- When asked to show rules, call the get_rules tool.\n"
            f"- Be concise. After tool calls succeed, briefly confirm what happened.\n"
            f"- Always format accuracy as a percentage (e.g. '6.7%'), never as a raw decimal.\n"
            f"- NEVER refuse to add or delete a rule. Always use the tool and let the system handle validation.\n"
        ),
    }

    messages = [system_msg] + req.messages

    # Allow up to 5 rounds of tool calls
    for _ in range(5):
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "tools": _TOOLS,
                    "tool_choice": "auto",
                },
            )

        if resp.status_code != 200:
            raise HTTPException(502, f"OpenRouter error: {resp.text}")

        data = resp.json()
        choice = data["choices"][0]
        assistant_msg = choice["message"]
        messages.append(assistant_msg)

        # If the model wants to call tools
        if assistant_msg.get("tool_calls"):
            for tc in assistant_msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = _json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
                result = _execute_tool(fn_name, fn_args, project_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })
            continue  # Let the model see tool results

        # No tool calls → return the final message
        # Gather which tools were called so the frontend can refresh
        tool_calls_made = []
        for m in messages:
            if m.get("role") == "tool":
                tool_calls_made.append(m.get("tool_call_id", ""))
        # Detect which actions happened by scanning assistant messages
        actions = set()
        for m in messages:
            if isinstance(m, dict) and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    actions.add(tc["function"]["name"])

        return {
            "content": assistant_msg.get("content", ""),
            "actions": list(actions),
        }

    # Fallback after max iterations
    return {"content": "I've completed the requested operations.", "actions": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)