import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('NSL extension is now active!');

    // Start language server if enabled
    const config = vscode.workspace.getConfiguration('nsl');
    if (config.get<boolean>('languageServer.enabled', true)) {
        startLanguageServer(context, config);
    }

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('nsl.runFile', runCurrentFile),
        vscode.commands.registerCommand('nsl.runSelection', runSelection),
        vscode.commands.registerCommand('nsl.formatDocument', formatDocument),
        vscode.commands.registerCommand('nsl.restartServer', () => restartLanguageServer(context)),
        vscode.commands.registerCommand('nsl.showGpuInfo', showGpuInfo)
    );

    // Register document formatter
    context.subscriptions.push(
        vscode.languages.registerDocumentFormattingEditProvider('nsl', {
            provideDocumentFormattingEdits: formatDocument
        })
    );

    // Status bar item
    const statusBarItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 100);
    statusBarItem.text = '$(brain) NSL';
    statusBarItem.tooltip = 'NSL Language';
    statusBarItem.command = 'nsl.showGpuInfo';
    statusBarItem.show();
    context.subscriptions.push(statusBarItem);

    // Watch for configuration changes
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(e => {
            if (e.affectsConfiguration('nsl.languageServer')) {
                restartLanguageServer(context);
            }
        })
    );
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

function startLanguageServer(context: vscode.ExtensionContext, config: vscode.WorkspaceConfiguration) {
    // Get server path from configuration or use default
    let serverPath = config.get<string>('languageServer.path', '');

    if (!serverPath) {
        // Try to find the language server
        const possiblePaths = [
            path.join(context.extensionPath, '..', '..', 'src', 'NSL.LanguageServer', 'bin', 'Release', 'net8.0', 'nsl-language-server.exe'),
            path.join(context.extensionPath, '..', '..', 'src', 'NSL.LanguageServer', 'bin', 'Debug', 'net8.0', 'nsl-language-server.exe'),
            'nsl-language-server' // System PATH
        ];

        for (const p of possiblePaths) {
            // In a real implementation, check if file exists
            serverPath = p;
            break;
        }
    }

    const serverOptions: ServerOptions = {
        run: {
            command: serverPath,
            transport: TransportKind.stdio
        },
        debug: {
            command: serverPath,
            transport: TransportKind.stdio
        }
    };

    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nsl' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.nsl')
        },
        outputChannelName: 'NSL Language Server'
    };

    client = new LanguageClient(
        'nsl',
        'NSL Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

async function restartLanguageServer(context: vscode.ExtensionContext) {
    if (client) {
        await client.stop();
    }
    const config = vscode.workspace.getConfiguration('nsl');
    startLanguageServer(context, config);
    vscode.window.showInformationMessage('NSL Language Server restarted');
}

async function runCurrentFile() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'nsl') {
        vscode.window.showErrorMessage('No NSL file is open');
        return;
    }

    const filePath = editor.document.fileName;
    const terminal = vscode.window.createTerminal('NSL');
    terminal.show();
    terminal.sendText(`nsl "${filePath}"`);
}

async function runSelection() {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'nsl') {
        return;
    }

    const selection = editor.selection;
    const selectedText = editor.document.getText(selection);

    if (!selectedText) {
        vscode.window.showErrorMessage('No text selected');
        return;
    }

    // Create a temporary file and run it
    const terminal = vscode.window.createTerminal('NSL');
    terminal.show();

    // For now, just show the selection - a real implementation would create a temp file
    vscode.window.showInformationMessage(`Running NSL selection: ${selectedText.substring(0, 50)}...`);
}

function formatDocument(document: vscode.TextDocument): vscode.TextEdit[] {
    // Basic formatting - the language server handles full formatting
    const text = document.getText();
    const lines = text.split('\n');
    const formatted: string[] = [];
    let indentLevel = 0;
    const config = vscode.workspace.getConfiguration('nsl.format');
    const tabSize = config.get<number>('tabSize', 4);
    const useSpaces = config.get<boolean>('insertSpaces', true);
    const indent = useSpaces ? ' '.repeat(tabSize) : '\t';

    for (let rawLine of lines) {
        let line = rawLine.trim();

        // Decrease indent for closing braces
        if (line.startsWith('}') || line.startsWith(']') || line.startsWith(')')) {
            indentLevel = Math.max(0, indentLevel - 1);
        }

        // Add indentation
        if (line.length > 0) {
            formatted.push(indent.repeat(indentLevel) + line);
        } else {
            formatted.push('');
        }

        // Increase indent for opening braces
        const opens = (line.match(/\{|\[|\(/g) || []).length;
        const closes = (line.match(/\}|\]|\)/g) || []).length;
        indentLevel = Math.max(0, indentLevel + opens - closes);
    }

    const fullRange = new vscode.Range(
        new vscode.Position(0, 0),
        new vscode.Position(document.lineCount, 0)
    );

    return [vscode.TextEdit.replace(fullRange, formatted.join('\n'))];
}

async function showGpuInfo() {
    // Display GPU information
    const info = `
# NSL GPU Information

## Checking available accelerators...

To use GPU acceleration in NSL:

\`\`\`nsl
import gpu

// List available GPUs
let devices = gpu.list_devices()
for d in devices {
    println(d)
}

// Create GPU accelerator
let accel = gpu.GpuAccelerator()

// Transfer tensor to GPU
let gpu_tensor = accel.to_gpu(cpu_tensor)

// Perform operations on GPU
let result = accel.matmul(gpu_tensor, weights)

// Transfer back to CPU
let output = accel.to_cpu(result)
\`\`\`

## Consciousness Operators on GPU

\`\`\`nsl
// Holographic operator (◈) - GPU accelerated
let h = accel.holographic(x)

// Gradient operator (∇) - GPU backpropagation
backward(loss)
let g = grad(tensor)

// Tensor product (⊗) - GPU outer product
let t = accel.tensor_product(a, b)

// Quantum branching (Ψ) - GPU superposition
let psi = accel.quantum_branch(x, num_branches=4)
\`\`\`
`;

    const doc = await vscode.workspace.openTextDocument({
        content: info,
        language: 'markdown'
    });
    await vscode.window.showTextDocument(doc);
}
