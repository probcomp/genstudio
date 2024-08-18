/*
Pattern Matching Rules for Syntax Highlighting

This script implements a flexible pattern matching algorithm for syntax highlighting.
The following rules define how patterns are matched against text:

1. Match literal characters exactly.
   Example: `abc` matches `abc`, but not `abd` or `abcd`.

2. `...` acts as a wildcard, matching any sequence of characters (including none).
   Example: `a...e` matches `abcde`, `ae`, `a123e`, etc.

3. Wildcards can appear anywhere in the pattern.
   Example: `go...()` matches `goSomewhere()`, `goToMars()`, `go()`, etc.

4. Ensure proper bracket nesting. When a wildcard is encountered, maintain a bracket depth counter.
   Only accept a closing bracket when the counter returns to 0.
   Example: `func(...)` matches `func(a, (b, c))`, `func(x)`, `func(a, [b, {c: d}])`, etc.

5. Handle strings. Ignore them within wildcards (i.e., in a wildcard once in a string, skip to the end of the string before continuing)

Additional Examples:
- `f(..., x)` matches `f(a, b, c, x)`, `f(x)`, `f((a, b), x)`, but not `f(a, b, c, y)`.
- `a...c...e` matches `abcdcde`, `ace`, `a123c456e`, etc.
- `if (...) {...}` matches `if (x > 0) {doSomething();}`, `if (true) {}`, etc.
- `"..."` matches any string content, including empty strings: `""`, `"hello"`, `"a\"b"`, etc.
- `[...]` matches any array content: `[]`, `[1, 2, 3]`, `[[a], {b: c}]`, etc.

Note: The pattern matching is sensitive to brackets, quotes, and escapes to ensure correct matching in complex scenarios.

*/

console.log("Loading custom JavaScript for syntax highlighting...");

// Utility function
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Core functionality
function highlightText({ source, styles }) {
    const preElements = document.getElementsByTagName('pre');
    for (let preElement of preElements) {
        const text = preElement.textContent;
        const matches = findMatches(text, source);
        if (matches.length === 0) continue;
        const highlightedText = applyHighlights(text, matches, styles);
        preElement.innerHTML = `<code>${highlightedText}</code>`;
    }
}

// Function to handle wildcard matching
function matchWildcard(text, startIndex, nextLiteral) {
    let index = startIndex;
    let bracketDepth = 0;
    let inString = null;

    while (index < text.length) {
        if (inString) {
            if (text[index] === inString && text[index - 1] !== '\\') {
                inString = null;
            }
        } else if (text[index] === '"' || text[index] === "'") {
            inString = text[index];
        } else if (bracketDepth === 0 && text[index] === nextLiteral) {
            return index;
        } else if (text[index] === '(' || text[index] === '[' || text[index] === '{') {
            bracketDepth++;
        } else if (text[index] === ')' || text[index] === ']' || text[index] === '}') {
            if (bracketDepth === 0) {
                return index;
            }
            bracketDepth--;
        }
        index++;
    }
    return index;
}

// Pattern matching state machine
function findMatches(text, pattern, debug = false) {
    const matches = [];
    const debugLog = debug ? [] : null;
    let patternIndex = 0;
    let textIndex = 0;
    let matchStart = 0;

    const log = (msg) => debug && debugLog.push(msg);

    log(`Starting match: text="${text}", pattern="${pattern}"`);

    while (textIndex < text.length) {
        log(`Current state: textIndex=${textIndex}, patternIndex=${patternIndex}`);
        if (pattern[patternIndex] === '.' && pattern[patternIndex + 1] === '.' && pattern[patternIndex + 2] === '.') {
            const nextLiteral = pattern[patternIndex + 3] || '';
            const startIndex = textIndex;
            textIndex = matchWildcard(text, textIndex, nextLiteral);
            patternIndex += 3;
            log(`Wildcard matched: ${text.slice(startIndex, textIndex)}`);
        } else if (text[textIndex] === pattern[patternIndex]) {
            if (patternIndex === 0) matchStart = textIndex;
            textIndex++;
            patternIndex++;
            log(`Matched character at ${textIndex - 1}`);
        } else {
            log(`Mismatch at ${textIndex}, moving to next character`);
            textIndex++;
            patternIndex = 0;
            matchStart = textIndex;
        }

        if (patternIndex === pattern.length) {
            matches.push([matchStart, textIndex]);
            log(`Match found: ${text.slice(matchStart, textIndex)}`);
            patternIndex = 0;
            matchStart = textIndex;
        }
    }

    log(`Finished matching. Found ${matches.length} matches.`);
    return debug ? { matches, log: debugLog.join('\n') } : matches;
}

// Apply highlights to the text
function applyHighlights(text, matches) {
    let result = '';
    let lastIndex = 0;

    for (let [start, end, styleString, matchId] of matches) {
        result += text.slice(lastIndex, start);
        result += `<span style="${styleString}" data-match-id="${matchId}">`;
        result += text.slice(start, end);
        result += '</span>';
        lastIndex = end;
    }

    result += text.slice(lastIndex);
    return result;
}

// Add this at the top of the file
const Observable10Colors = [
    "#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
    "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
];

function processLinksAndHighlight() {
    console.log('Starting processLinksAndHighlight');

    const elements = document.querySelectorAll('pre, a[href^="uplight"]');
    console.log('Found elements:', elements.length);

    const preMap = new Map();
    let currentPre = null;

    elements.forEach(element => {
        if (element.tagName === 'PRE') {
            currentPre = element;
            preMap.set(currentPre, []);
        } else if (element.tagName === 'A' && currentPre) {
            preMap.get(currentPre).push(element);
        }
    });

    console.log('Number of pre elements with matches:', preMap.size);

    // Process each pre element
    preMap.forEach((links, preElement) => {
        console.log('Processing pre element:', preElement);
        let text = preElement.textContent;
        let allMatches = [];

        links.forEach((link, index) => {
            const url = new URL(link.href);
            let pattern = url.searchParams.get('match');
            if (!pattern) {
                pattern = link.textContent;
            }
            const colorIndex = index % Observable10Colors.length;
            const color = Observable10Colors[colorIndex];
            const style = `color: ${color}; font-weight: bold; background-color: ${color}20;`;
            console.log('Link pattern:', pattern, 'style:', style);

            const matches = findMatches(text, pattern);
            console.log('Matches found:', matches.length);
            const matchId = `match-${index}-${Math.random().toString(36).substr(2, 9)}`;
            allMatches.push(...matches.map(match => [...match, style, matchId]));

            // Replace link with span
            const span = document.createElement('span');
            span.textContent = link.textContent;
            span.style.cssText = style;
            span.dataset.matchId = matchId;
            link.parentNode.replaceChild(span, link);
        });

        // Only apply highlights if there are matches
        if (allMatches.length > 0) {
            console.log('Applying highlights to pre element');
            const highlightedText = applyHighlights(text, allMatches);
            preElement.innerHTML = `<code>${highlightedText}</code>`;
        } else {
            console.log('No matches found for this pre element');
        }
    });

    console.log('Finished processLinksAndHighlight');
}

// Add this function after processLinksAndHighlight
function addHoverEffect() {
    function setBackgroundColorWithOpacity(element, color, opacity) {
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
        if (rgbaMatch) {
            const [, r, g, b] = rgbaMatch;
            element.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${opacity})`;
        } else {
            console.warn('Color format not recognized:', color);
        }
    }

    document.addEventListener('mouseover', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const color = target.style.color;
            const elements = document.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                setBackgroundColorWithOpacity(el, color, 0.25);
            });
        }
    });

    document.addEventListener('mouseout', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const color = target.style.color;
            const elements = document.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                setBackgroundColorWithOpacity(el, color, 0.125);
            });
        }
    });
}

// Modify the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOMContentLoaded event fired');
    processLinksAndHighlight();
    addHoverEffect();
});

// Test suite
function runTests() {
    console.log("Running tests for syntax highlighting functions...");

    const testCases = [
        { input: "go(a, b)", pattern: "go(...)", name: "Simple go(...) match", expected: ["go(a, b)"] },
        { input: "func(a, func2(b, c), d)", pattern: "func(...)", name: "Nested brackets", expected: ["func(a, func2(b, c), d)"] },
        { input: "go(a, b) go(c, d) go(e, f)", pattern: "go(...)", name: "Multiple matches", expected: ["go(a, b)", "go(c, d)", "go(e, f)"] },
        { input: "function(a, b)", pattern: "go(...)", name: "No match", expected: [] },
        { input: "func(a, [b, c], {d: (e, f)})", pattern: "func(...)", name: "Complex nested brackets", expected: ["func(a, [b, c], {d: (e, f)})"] },
        { input: "func('a(b)', \"c)d\", e\\(f\\))", pattern: "func(...)", name: "Strings and escaped characters", expected: ["func('a(b)', \"c)d\", e\\(f\\))"] },
        { input: "a b c d e", pattern: "a...e", name: "Wildcard outside brackets", expected: ["a b c d e"] },
        { input: "goSomewhere()", pattern: "go...()", name: "Wildcard before brackets", expected: ["goSomewhere()"] },
        { input: "f(a, b, c, x)", pattern: "f(...x)", name: "Wildcard with specific end", expected: ["f(a, b, c, x)"] },
        { input: "abcdcde", pattern: "a...c...e", name: "Multiple wildcards", expected: ["abcdcde"] },
        { input: "if (x > 0) {doSomething();}", pattern: "if (...) {...}", name: "if statement", expected: ["if (x > 0) {doSomething();}"] },
        { input: "\"hello\"", pattern: "...", name: "String content", expected: ["\"hello\""] },
        { input: "[1, 2, 3]", pattern: "[...]", name: "Array content", expected: ["[1, 2, 3]"] },
    ];

    let passedTests = 0;
    let failedTests = 0;

    testCases.forEach(({ input, pattern, name, expected }) => {
        console.log(`\nTest: ${name}`);
        console.log(`Pattern: ${pattern}`);
        console.log(`Input: ${input}`);
        console.log(`Expected: ${JSON.stringify(expected)}`);

        try {
            const result = findMatches(input, pattern, true);
            const actualMatches = result.matches.map(([start, end]) => input.slice(start, end));
            const passed = JSON.stringify(actualMatches) === JSON.stringify(expected);

            if (!passed) {
                console.log("Test failed. Debug information:");
                console.log(`Actual: ${JSON.stringify(actualMatches)}`);
                console.log("\nDetailed matching process:");
                console.log(result.log);
                failedTests++;
            } else {
                console.log("Test passed.");
                passedTests++;
            }
        } catch (error) {
            console.log(`Test threw an error: ${error.message}`);
            console.log(error.stack);
            failedTests++;
        }
    });

    console.log(`\nTest summary: ${passedTests} passed, ${failedTests} failed`);
}

// Run tests
runTests();
