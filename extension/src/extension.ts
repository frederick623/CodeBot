import * as vscode from "vscode";

function daemonUrl(): string {
  return vscode.workspace
    .getConfiguration("codebot")
    .get<string>("daemonUrl", "http://127.0.0.1:9099");
}

async function openWorkspace(root: string): Promise<void> {
  await fetch(`${daemonUrl()}/workspace/open`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ workspace_path: root }),
  });
}

interface ChatResp {
  answer: string;
  patches: { file: string; diff: string }[];
  context_used: string[];
  trace_id: string;
}

export function activate(context: vscode.ExtensionContext) {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (folder && vscode.workspace.isTrusted) {
    void openWorkspace(folder.uri.fsPath);
  }

  const ask = vscode.commands.registerCommand("codebot.ask", async () => {
    if (!vscode.workspace.isTrusted) {
      vscode.window.showWarningMessage(
        "Code Assistant is limited in untrusted workspaces."
      );
      return;
    }

    const prompt = await vscode.window.showInputBox({
      prompt: "What do you want to do?",
      placeHolder: "e.g. Add pagination to this endpoint",
    });
    if (!prompt) return;

    const editor = vscode.window.activeTextEditor;
    const root = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? "";

    // The extension gathers editor STATE only. The user's typed prompt is the
    // sole instruction channel; file content is sent to the daemon as data.
    const payload = {
      workspace_path: root,
      active_file: editor
        ? vscode.workspace.asRelativePath(editor.document.uri)
        : null,
      selection: editor
        ? {
            start: editor.document.offsetAt(editor.selection.start),
            end: editor.document.offsetAt(editor.selection.end),
          }
        : null,
      open_files: vscode.workspace.textDocuments.map((d) =>
        vscode.workspace.asRelativePath(d.uri)
      ),
      user_prompt: prompt,
    };

    const resp = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: "Thinking..." },
      async () => {
        const r = await fetch(`${daemonUrl()}/chat`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        return (await r.json()) as ChatResp;
      }
    );

    await showResult(resp);
  });

  context.subscriptions.push(ask);
}

async function showResult(resp: ChatResp) {
  const doc = await vscode.workspace.openTextDocument({
    content:
      resp.answer +
      "\n\n---\nContext used:\n" +
      resp.context_used.map((f) => "- " + f).join("\n"),
    language: "markdown",
  });
  await vscode.window.showTextDocument(doc, { preview: true });

  // User-approved patch application: show each diff, require explicit confirm.
  for (const patch of resp.patches) {
    const choice = await vscode.window.showInformationMessage(
      `Apply changes to ${patch.file}?`,
      { modal: true, detail: patch.diff.slice(0, 2000) },
      "Apply",
      "Skip"
    );
    if (choice === "Apply") {
      // Send approval back to the daemon's /edit endpoint, which performs the
      // sandboxed write via ToolExecutor.apply_full. (Endpoint omitted for brevity.)
      vscode.window.showInformationMessage(`Approved ${patch.file}`);
    }
  }
}

export function deactivate() {}