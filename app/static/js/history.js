export class HistoryStack {
    constructor(maxDepth = 100, onChange = null) {
        this.maxDepth = maxDepth;
        this.undoStack = [];
        this.redoStack = [];
        this.onChange = onChange;
    }

    execute(command) {
        command.do();
        this.undoStack.push(command);
        if (this.undoStack.length > this.maxDepth) {
            this.undoStack.shift();
        }
        this.redoStack = [];
        this.#emitChange();
    }

    undo() {
        const command = this.undoStack.pop();
        if (!command) {
            return false;
        }
        command.undo();
        this.redoStack.push(command);
        this.#emitChange();
        return true;
    }

    redo() {
        const command = this.redoStack.pop();
        if (!command) {
            return false;
        }
        command.do();
        this.undoStack.push(command);
        this.#emitChange();
        return true;
    }

    canUndo() {
        return this.undoStack.length > 0;
    }

    canRedo() {
        return this.redoStack.length > 0;
    }

    #emitChange() {
        if (typeof this.onChange === "function") {
            this.onChange({
                canUndo: this.canUndo(),
                canRedo: this.canRedo(),
                undoCount: this.undoStack.length,
                redoCount: this.redoStack.length
            });
        }
    }
}

export class DeleteKeysCommand {
    constructor(context, keys) {
        this.context = context;
        this.keys = Array.from(new Set(keys));
        this.label = `Delete ${this.keys.length} objects`;
    }

    do() {
        this.context.setDeleted(this.keys, true);
    }

    undo() {
        this.context.setDeleted(this.keys, false);
    }
}

export class HideKeysCommand {
    constructor(context, keys) {
        this.context = context;
        this.keys = Array.from(new Set(keys));
        this.label = `Hide ${this.keys.length} objects`;
    }

    do() {
        this.context.setHidden(this.keys, true);
    }

    undo() {
        this.context.setHidden(this.keys, false);
    }
}

export class SetColorCommand {
    constructor(context, key, nextColor) {
        this.context = context;
        this.key = key;
        this.nextColor = nextColor;
        this.previousColor = context.getColor(key);
        this.label = "Set object color";
    }

    do() {
        this.context.setColor(this.key, this.nextColor);
    }

    undo() {
        this.context.setColor(this.key, this.previousColor);
    }
}

export class RestoreAllCommand {
    constructor(context) {
        this.context = context;
        this.previousHidden = context.getHiddenSnapshot();
        this.label = "Show all objects";
    }

    do() {
        this.context.restoreAllHidden();
    }

    undo() {
        this.context.replaceHidden(this.previousHidden);
    }
}
