/*
Uplight: Flexible Syntax Highlighting for Documentation

Usage:
1. Include this script in your HTML.
2. Call `uplight({ target: 'body', debug: true })` to apply highlighting.
3. Use `<a href="uplight?match=pattern&dir=direction">` links in your documentation to define highlight patterns.
   These links will automatically apply to the nearest `<pre>` block based on the specified direction.

Key Features:
- Automatically associates highlight patterns with code blocks based on specified direction.
- Supports flexible pattern matching with wildcards.
- Provides interactive hover effects for highlighted code.

For detailed pattern syntax and behavior, see the documentation in the matchWildcard function.
*/

// Utility functions
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const Observable10Colors = [
    "#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
    "#d4af37", "#af7aa1", "#ff9da7", "#9c755f", "#9a908b"
];

let debug = false;
const log = (...args) => debug && console.log(...args);

// Wildcard matching functions
function matchWildcard(text, startIndex, nextLiteral) {
    /*
    Wildcard Matching Rules:
    1. '...' matches any sequence of characters, including none.
    2. Proper bracket nesting is ensured using a depth counter.
    3. Strings are handled by skipping to the end once encountered within a wildcard.
    4. Escaped characters and quotes within strings are properly handled.

    Examples:
    - 'func(...)' matches 'func(a, (b, c))', 'func(x)', 'func(a, [b, {c: d}])', etc.
    - 'a...c...e' matches 'abcdcde', 'ace', 'a123c456e', etc.
    - 'if (...) {...}' matches 'if (x > 0) {doSomething();}', 'if (true) {}', etc.
    - '"..."' matches any string content, including empty strings.
    - '[...]' matches any array content.

    Note: This function is sensitive to brackets, quotes, and escapes to ensure correct matching in complex scenarios.
    */

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

// Matching functions
function findMatches(text, pattern) {
    const isRegex = pattern.startsWith('/') && pattern.endsWith('/');
    if (isRegex) {
        return findRegexMatches(text, pattern.slice(1, -1));
    }

    let matches = [];
    let currentPosition = 0;

    while (currentPosition < text.length) {
        const match = findSingleMatch(text, pattern, currentPosition);

        if (match) {
            const [matchStart, matchEnd] = match;
            matches.push([matchStart, matchEnd]);
            currentPosition = matchEnd;
        } else {
            currentPosition++;
        }
    }

    return matches;
}

function findRegexMatches(text, pattern) {
    const regex = new RegExp(pattern, 'g');
    let matches = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
        if (match.length > 1) {
            // If there are capture groups, use the first one
            const captureGroup = match[1];
            const start = match.index + match[0].indexOf(captureGroup);
            const end = start + captureGroup.length;
            matches.push([start, end]);
        } else {
            // If no capture groups, use the whole match
            matches.push([match.index, regex.lastIndex]);
        }
    }

    return matches;
}

function findSingleMatch(text, pattern, startPosition) {
    let patternPosition = 0;
    let textPosition = startPosition;
    let matchStart = startPosition;

    while (textPosition < text.length && patternPosition < pattern.length) {
        if (pattern.substr(patternPosition, 3) === '...') {
            // Handle wildcard
            const nextCharacter = pattern[patternPosition + 3] || '';
            textPosition = matchWildcard(text, textPosition, nextCharacter);
            patternPosition += 3;
        } else if (text[textPosition] === pattern[patternPosition]) {
            // Characters match, move to next
            textPosition++;
            patternPosition++;
        } else {
            // No match found
            return null;
        }
    }

    // Check if we've matched the entire pattern
    if (patternPosition === pattern.length) {
        return [matchStart, textPosition];
    }

    return null;
}
// Highlighting functions
function findMatchesForPatterns(text, patterns, matchId) {
    return patterns.flatMap(pattern =>
        findMatches(text, pattern).map(match => [...match, matchId])
    );
}

function highlightPatterns(preElement, patterns, options = {}) {
    const text = preElement.textContent;
    const { matchId = `match-${Math.random().toString(36).slice(2, 11)}`, colorIndex = 0 } = options;
    const matches = findMatchesForPatterns(text, patterns, matchId);
    if (matches.length === 0) return;

    const color = Observable10Colors[colorIndex % Observable10Colors.length];
    const styleString = `--uplight-color: ${color};`;

    const allMatches = matches.map(match => [...match, styleString]);
    preElement.innerHTML = `<code>${applyHighlights(text, allMatches)}</code>`;
}

function applyHighlights(text, matches) {
    // Sort matches in reverse order based on their start index
    matches.sort((a, b) => b[0] - a[0]);

    return matches.reduce((result, [start, end, matchId, styleString]) => {
        const beforeMatch = result.slice(0, start);
        const matchContent = result.slice(start, end);
        const afterMatch = result.slice(end);

        return beforeMatch +
               `<span class="uplight-code" style="${styleString}" data-match-id="${matchId}">` +
               matchContent +
               '</span>' +
               afterMatch;
    }, text);
}

// Link processing and hover effect functions
function processLinksAndHighlight(targetElement) {
    const elements = targetElement.querySelectorAll('pre, a[href^="uplight"]');

    const preMap = new Map();
    const linkMap = new Map();
    const colorMap = new Map();
    let colorIndex = 0;

    // Process all elements
    elements.forEach((element, index) => {
        if (element.tagName === 'PRE') {
            preMap.set(element, []);
        } else if (element.tagName === 'A') {
            const url = new URL(element.href);
            const matchId = `match-${index}-${Math.random().toString(36).slice(2, 11)}`;
            const direction = url.searchParams.get('dir') || 'down';
            linkMap.set(element, {
                direction: direction === 'up' || direction === '-1' ? -1 :
                           direction === 'down' || direction === '1' ? 1 :
                           0, // 'all' or '0'
                patterns: (url.searchParams.get('match') || element.textContent).split(','),
                index,
                matchId
            });
            colorMap.set(matchId, colorIndex);
            colorIndex = (colorIndex + 1) % Observable10Colors.length;
            element.addEventListener('click', e => e.preventDefault());
        }
    });

    // Second pass: Process links and find matches in pre elements
    linkMap.forEach(({ direction, patterns, index, matchId }, linkElement) => {
        const findMatchingPre = (start, end, step) => {
            for (let i = start; i !== end; i += step) {
                if (elements[i].tagName === 'PRE') {
                    return [elements[i]];
                }
            }
            return [];
        };

        const matchingPres = direction === 0
            ? Array.from(preMap.keys())
            : direction === -1
                ? findMatchingPre(index - 1, -1, -1)
                : findMatchingPre(index + 1, elements.length, 1);

        matchingPres.forEach(matchingPre => {
            const text = matchingPre.textContent;
            const newMatches = findMatchesForPatterns(text, patterns, matchId);
            preMap.get(matchingPre).push(...newMatches);
        });
    });

    // Apply highlights to pre elements
    preMap.forEach((matches, preElement) => {
        if (matches.length > 0) {
            const text = preElement.textContent;
            const allMatches = matches.map(([start, end, matchId]) => [
                start,
                end,
                matchId,
                `--uplight-color: ${Observable10Colors[colorMap.get(matchId)]};`
            ]);
            preElement.innerHTML = `<code>${applyHighlights(text, allMatches)}</code>`;
        }
    });

    log(preMap);

    // Process links
    linkMap.forEach((linkData, linkElement) => {
        const { matchId } = linkData;
        const color = Observable10Colors[colorMap.get(matchId)];
        linkElement.dataset.matchId = matchId;
        linkElement.classList.add('uplight-link');
        linkElement.style.setProperty('--uplight-color', color);
    });
}

function addHoverEffect(targetElement) {
    targetElement.addEventListener('mouseover', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const elements = targetElement.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                el.classList.add('uplight-hover');
            });
        }
    });

    targetElement.addEventListener('mouseout', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const elements = targetElement.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                el.classList.remove('uplight-hover');
            });
        }
    });
}

// Main uplight function
function uplight({
    target = 'body',
    debugMode = false
}) {
    debug = debugMode;
    const targetElement = typeof target === 'string' ? document.querySelector(target) : target;

    if (!targetElement) {
        console.error(`Uplight: Target element not found - ${target}`);
        return;
    }

    processLinksAndHighlight(targetElement);
    addHoverEffect(targetElement);
}

// Test suite
function runTests() {
    log("Running tests for Uplight highlighting functions...");

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
        { input: "func(a, b)", pattern: "/func\\(.*?\\)/", name: "Regex match", expected: ["func(a, b)"] },
    ];

    let passedTests = 0;
    let failedTests = 0;

    testCases.forEach(({ input, pattern, name, expected }) => {
        log(`\nTest: ${name}`);
        log(`Pattern: ${pattern}`);
        log(`Input: ${input}`);
        log(`Expected: ${JSON.stringify(expected)}`);
        const debugMode = debug
        try {

            const result = findMatches(input, pattern);
            const actualMatches = result.map(([start, end]) => input.slice(start, end));
            const passed = JSON.stringify(actualMatches) === JSON.stringify(expected);

            if (!passed) {
                debug = true;
                log("Test failed. Debug information:");
                log(`Actual: ${JSON.stringify(actualMatches)}`);
                failedTests++;
            } else {
                log("Test passed.");
                passedTests++;
            }
        } catch (error) {
            log(`Test threw an error: ${error.message}`);
            log(error.stack);
            failedTests++;
        } finally {
            debug = false;
        }
    });

    log(`\nTest summary: ${passedTests} passed, ${failedTests} failed`);
}

// Modify the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function () {
    uplight({ target: 'body', debugMode: false });
});

// Run tests
runTests();
