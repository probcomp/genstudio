/*
Uplight: Flexible Syntax Highlighting for Documentation

Usage:
1. Include this script in your HTML.
2. Call `uplight({ target: 'body', debug: true })` to apply highlighting.
3. Use `<a href="uplight?match=pattern">` links in your documentation to define highlight patterns.
   These links will automatically apply to the nearest `<pre>` block above them.

Key Features:
- Automatically associates highlight patterns with the nearest code block.
- Supports flexible pattern matching with wildcards.
- Provides interactive hover effects for highlighted code.

For detailed pattern syntax and behavior, see the documentation in the matchWildcard function.
*/

const Observable10Colors = [
    "#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f",
    "#d4af37", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
];

let debug = false;
const log = (...args) => debug && console.log(...args);

// Utility function
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Core highlighting functionality
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
// Find all matches of a pattern in the text
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

// New function to find regex matches
function findRegexMatches(text, pattern) {
    const regex = new RegExp(pattern, 'g');
    let matches = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
        matches.push([match.index, regex.lastIndex]);
    }

    return matches;
}

// Find a single match of the pattern starting from a given position
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


function applyHighlights(text, matches) {
    // Sort matches in reverse order based on their start index
    matches.sort((a, b) => b[0] - a[0]);

    return matches.reduce((result, [start, end, styleString, matchId]) => {
        const beforeMatch = result.slice(0, start);
        const matchContent = result.slice(start, end);
        const afterMatch = result.slice(end);

        return beforeMatch +
               `<span class="uplight-highlight" style="${styleString}" data-match-id="${matchId}">` +
               matchContent +
               '</span>' +
               afterMatch;
    }, text);
}

function processLinksAndHighlight(targetElement) {

    const elements = targetElement.querySelectorAll('pre, a[href^="uplight"]');

    const preMap = new Map();
    const linkMap = new Map();
    const colorMap = new Map();
    let colorIndex = 0;

    // First pass: Process all pre elements and links
    elements.forEach((element, index) => {
        if (element.tagName === 'PRE') {
            preMap.set(element, []);
        } else if (element.tagName === 'A') {
            const url = new URL(element.href);
            const direction = url.searchParams.get('dir') || 'up';
            const patterns = (url.searchParams.get('match') || element.textContent).split(',');
            const matchId = `match-${index}-${Math.random().toString(36).substr(2, 9)}`;
            console.log(url.searchParams)
            log(patterns)
            linkMap.set(element, { direction, patterns, index, matchId });
            colorMap.set(matchId, colorIndex);
            colorIndex = (colorIndex + 1) % Observable10Colors.length;
        }
    });

    // Second pass: Process links and find matches in pre elements
    linkMap.forEach((linkData, linkElement) => {
        const { direction, patterns, index, matchId } = linkData;
        const searchForMatch = (preElement) => {
            const text = preElement.textContent;
            let allMatches = [];
            patterns.forEach(pattern => {
                const matches = findMatches(text, pattern);
                allMatches.push(...matches.map(match => [...match, matchId]));
            });
            return allMatches.length > 0 ? allMatches : null;
        };

        let matchingPres = [];
        if (direction === 'all') {
            preMap.forEach((_, preElement) => {
                const matches = searchForMatch(preElement);
                if (matches) {
                    matchingPres.push(preElement);
                }
            });
        } else if (direction === 'up') {
            for (let i = index - 1; i >= 0; i--) {
                if (elements[i].tagName === 'PRE') {
                    const matches = searchForMatch(elements[i]);
                    if (matches) {
                        matchingPres.push(elements[i]);
                        break;
                    }
                }
            }
        } else {
            for (let i = index + 1; i < elements.length; i++) {
                if (elements[i].tagName === 'PRE') {
                    const matches = searchForMatch(elements[i]);
                    if (matches) {
                        matchingPres.push(elements[i]);
                        break;
                    }
                }
            }
        }

        matchingPres.forEach(matchingPre => {
            const matches = searchForMatch(matchingPre);
            if (matches) {
                preMap.get(matchingPre).push(...matches);
            }
        });
    });

    // Remove preMaps that don't have any matches
    for (const [preElement, matches] of preMap.entries()) {
        if (matches.length === 0) {
            preMap.delete(preElement);
        }
    }

    log(preMap);

    // Process links
    linkMap.forEach((linkData, linkElement) => {
        const { matchId } = linkData;
        const colorIndex = colorMap.get(matchId);
        const color = Observable10Colors[colorIndex];
        const style = `color: ${color}; font-weight: bold; background-color: ${color}20;`;

        const span = document.createElement('span');
        span.textContent = linkElement.textContent;
        span.style.cssText = style;
        span.dataset.matchId = matchId;
        span.classList.add('uplight-reference');
        linkElement.parentNode.replaceChild(span, linkElement);
    });

    // Process each pre element
    preMap.forEach((matches, preElement) => {
        let text = preElement.textContent;
        let allMatches = [];

        matches.forEach(match => {
            const [start, end, matchId] = match;
            const colorIndex = colorMap.get(matchId);
            const color = Observable10Colors[colorIndex];
            const style = `color: ${color}; font-weight: bold; background-color: ${color}20;`;
            allMatches.push([start, end, style, matchId]);
        });

        // Only apply highlights if there are matches
        if (allMatches.length > 0) {
            const highlightedText = applyHighlights(text, allMatches);
            preElement.innerHTML = `<code class="uplight-code">${highlightedText}</code>`;
        } else {
            log('No matches found for this pre element');
        }
    });

    log('Finished processLinksAndHighlight');
}

function addHoverEffect(targetElement) {
    function setBackgroundColorWithOpacity(element, color, opacity) {
        const rgbaMatch = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*[\d.]+)?\)/);
        if (rgbaMatch) {
            const [, r, g, b] = rgbaMatch;
            element.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${opacity})`;
        } else {
            log('Color format not recognized:', color);
        }
    }

    targetElement.addEventListener('mouseover', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const color = target.style.color;
            const elements = targetElement.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                setBackgroundColorWithOpacity(el, color, 0.25);
            });
        }
    });

    targetElement.addEventListener('mouseout', (event) => {
        const target = event.target;
        if (target.dataset.matchId) {
            const matchId = target.dataset.matchId;
            const color = target.style.color;
            const elements = targetElement.querySelectorAll(`[data-match-id="${matchId}"]`);
            elements.forEach(el => {
                setBackgroundColorWithOpacity(el, color, 0.125);
            });
        }
    });
}

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

// Modify the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', function () {
    uplight({ target: 'body', debugMode: true });
});

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

// Run tests
runTests();
