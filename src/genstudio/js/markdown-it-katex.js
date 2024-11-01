// written by claude-sonnet-3.5 based on the MIT-licensed https://github.com/waylonflinn/markdown-it-katex
import katex from 'katex';

// Configuration options with defaults
const defaultOptions = {
    throwOnError: false,
    errorColor: '#cc0000',
    minRuleThickness: 0.05,
    strict: 'warn',
    delimiters: {
        inline: '$',
        block: '$$'
    },
    displayMode: true
};

// Enhanced delimiter validation with more robust character checking
const isValidDelim = (state, pos) => {
    const max = state.posMax;
    const src = state.src;

    // Check if we're at the end of the text
    if (pos >= max) return { can_open: false, can_close: false };

    const prevChar = pos > 0 ? src.charCodeAt(pos - 1) : -1;
    const nextChar = pos + 1 <= max ? src.charCodeAt(pos + 1) : -1;

    // Enhanced checks for opening/closing conditions
    const isWordChar = (char) => {
        return (char >= 0x30 && char <= 0x39) || // Numbers
               (char >= 0x41 && char <= 0x5A) || // Uppercase
               (char >= 0x61 && char <= 0x7A);   // Lowercase
    };

    const isPunctChar = (char) => {
        return (char >= 0x21 && char <= 0x2F) || // ! to /
               (char >= 0x3A && char <= 0x40) || // : to @
               (char >= 0x5B && char <= 0x60) || // [ to `
               (char >= 0x7B && char <= 0x7E);   // { to ~
    };

    // More specific rules for delimiter validation
    return {
        can_open: !isWordChar(prevChar) &&
                 nextChar !== 0x20 && // space
                 nextChar !== 0x09 && // tab
                 nextChar !== -1,     // end of line
        can_close: !isWordChar(nextChar) &&
                  prevChar !== 0x20 && // space
                  prevChar !== 0x09 && // tab
                  prevChar !== -1      // start of line
    };
};

// Enhanced inline math processing with better error handling
const mathInline = (state, silent) => {
    const delimiter = state.md.options.katex?.delimiters?.inline || '$';
    if (state.src[state.pos] !== delimiter) return false;

    const res = isValidDelim(state, state.pos);
    if (!res.can_open) {
        if (!silent) state.pending += delimiter;
        state.pos += 1;
        return true;
    }

    // Find closing delimiter with improved escape handling
    const start = state.pos + 1;
    let match = start;
    let escaped = false;

    while ((match = state.src.indexOf(delimiter, match)) !== -1) {
        escaped = false;
        let pos = match - 1;

        // Count consecutive escapes
        while (pos >= 0 && state.src[pos] === '\\') {
            escaped = !escaped;
            pos--;
        }

        if (!escaped) break;
        match++;
    }

    if (match === -1 || !isValidDelim(state, match).can_close) {
        if (!silent) state.pending += delimiter;
        state.pos = start;
        return true;
    }

    const content = state.src.slice(start, match);
    if (!content.trim()) {
        if (!silent) state.pending += delimiter + delimiter;
        state.pos = start + 1;
        return true;
    }

    if (!silent) {
        const token = state.push('math_inline', 'math', 0);
        token.markup = delimiter;
        token.content = content.trim();
        token.meta = { escaped: false };
    }

    state.pos = match + 1;
    return true;
};

// Enhanced block math with better multiline handling
const mathBlock = (state, start, end, silent) => {
    const delimiter = state.md.options.katex?.delimiters?.block || '$$';
    let pos = state.bMarks[start] + state.tShift[start];
    const max = state.eMarks[start];

    if (pos + delimiter.length > max) return false;
    if (state.src.slice(pos, pos + delimiter.length) !== delimiter) return false;

    let firstLine = state.src.slice(pos + delimiter.length, max);
    let next = start;
    let lastLine = '';
    let found = false;
    let lastPos = -1;

    // Handle single-line block math
    if (firstLine.trim().endsWith(delimiter)) {
        firstLine = firstLine.trim().slice(0, -delimiter.length);
        found = true;
    }

    // Enhanced multi-line block math handling
    while (!found && next < end) {
        next++;

        if (next >= end) break;

        pos = state.bMarks[next] + state.tShift[next];
        const currentMax = state.eMarks[next];

        // Check for proper indentation and empty lines
        if (pos < currentMax && state.tShift[next] < state.blkIndent) break;

        const currentLine = state.src.slice(pos, currentMax);
        if (currentLine.trim().endsWith(delimiter)) {
            lastPos = currentLine.trimEnd().lastIndexOf(delimiter);
            lastLine = currentLine.slice(0, lastPos).trim();
            found = true;
            break;
        }
    }

    if (!found) return false;
    if (silent) return true;

    const token = state.push('math_block', 'math', 0);
    token.block = true;
    token.content = [
        firstLine.trim(),
        state.getLines(start + 1, next, state.tShift[start], true),
        lastLine
    ].filter(Boolean).join('\n');
    token.map = [start, state.line];
    token.markup = delimiter;

    state.line = next + 1;
    return true;
};

// Enhanced rendering with better error handling and customization
const createPlugin = (md, userOptions = {}) => {
    const options = { ...defaultOptions, ...userOptions };

    const renderKatex = (latex, isBlock = false) => {
        const renderOptions = {
            ...options,
            displayMode: isBlock
        };

        try {
            return katex.renderToString(latex, renderOptions);
        } catch (error) {
            console.error(`KaTeX error in ${isBlock ? 'block' : 'inline'} mode:`, error);

            // Return formatted error in development, simple message in production
            return process.env.NODE_ENV === 'development'
                ? `<span style="color: ${options.errorColor}">KaTeX error: ${error.message}</span>`
                : `<span style="color: ${options.errorColor}">[Math Error]</span>`;
        }
    };

    const inlineRenderer = (tokens, idx) => {
        return renderKatex(tokens[idx].content, false);
    };

    const blockRenderer = (tokens, idx) => {
        return `<div class="math-block">${renderKatex(tokens[idx].content, true)}</div>\n`;
    };

    // Register rules and renderers
    md.inline.ruler.after('escape', 'math_inline', mathInline);
    md.block.ruler.after('blockquote', 'math_block', mathBlock, {
        alt: ['paragraph', 'reference', 'blockquote', 'list']
    });

    md.renderer.rules.math_inline = inlineRenderer;
    md.renderer.rules.math_block = blockRenderer;
};

export default createPlugin;
