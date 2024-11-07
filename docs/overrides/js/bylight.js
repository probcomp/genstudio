"use strict";
var bylight = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/index.ts
  var src_exports = {};
  __export(src_exports, {
    DefaultColors: () => DefaultColors,
    addHoverEffect: () => addHoverEffect,
    default: () => bylight,
    escapeRegExp: () => escapeRegExp,
    findMatches: () => findMatches,
    findRegexMatches: () => findRegexMatches,
    highlight: () => highlight,
    processLinksAndHighlight: () => processLinksAndHighlight
  });
  function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }
  var DefaultColors = [
    "#59a14f",
    "#b82efe",
    "#007bfe",
    "#6a6a6a",
    "#ff4245",
    "#7c2d00",
    "#76b7b2",
    "#d4af37",
    "#ff9da7",
    "#f28e2c"
  ];
  function matchWildcard(text, startIndex, nextLiteral) {
    let index = startIndex;
    let bracketDepth = 0;
    let inString = null;
    while (index < text.length) {
      if (inString) {
        if (text[index] === inString && text[index - 1] !== "\\") {
          inString = null;
        }
      } else if (text[index] === '"' || text[index] === "'") {
        inString = text[index];
      } else if (bracketDepth === 0 && text[index] === nextLiteral) {
        return index;
      } else if (text[index] === "(" || text[index] === "[" || text[index] === "{") {
        bracketDepth++;
      } else if (text[index] === ")" || text[index] === "]" || text[index] === "}") {
        if (bracketDepth === 0) {
          return index;
        }
        bracketDepth--;
      }
      index++;
    }
    return index;
  }
  function findMatches(text, pattern) {
    if (pattern.startsWith("/") && pattern.endsWith("/")) {
      return findRegexMatches(text, pattern.slice(1, -1));
    }
    const matches = [];
    let currentPosition = 0;
    while (currentPosition < text.length) {
      const match = findSingleMatch(text, pattern, currentPosition);
      if (match) {
        matches.push(match);
        currentPosition = match[1];
      } else {
        currentPosition++;
      }
    }
    return matches;
  }
  function findRegexMatches(text, pattern) {
    const regex = new RegExp(pattern, "g");
    let matches = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
      matches.push([match.index, regex.lastIndex]);
    }
    return matches;
  }
  function findSingleMatch(text, pattern, startPosition) {
    let patternPosition = 0;
    let textPosition = startPosition;
    while (textPosition < text.length && patternPosition < pattern.length) {
      if (pattern.startsWith("...", patternPosition)) {
        const nextCharacter = pattern[patternPosition + 3] || "";
        textPosition = matchWildcard(text, textPosition, nextCharacter);
        patternPosition += 3;
      } else if (text[textPosition] === pattern[patternPosition]) {
        textPosition++;
        patternPosition++;
      } else {
        return null;
      }
    }
    return patternPosition === pattern.length ? [startPosition, textPosition] : null;
  }
  function findMatchesForPatterns(text, patterns, matchId) {
    return patterns.flatMap(
      (pattern) => findMatches(text, pattern).map(
        (match) => [...match, matchId]
      )
    );
  }
  function generateUniqueId() {
    return `match-${Math.random().toString(36).slice(2, 11)}`;
  }
  function highlight(target, patterns, options = {}, colorScheme = DefaultColors) {
    if (!patterns || Array.isArray(patterns) && patterns.length === 0) {
      return;
    }
    const patternsArray = Array.isArray(patterns) ? patterns : [patterns];
    const elements = typeof target === "string" ? document.querySelectorAll(target) : target instanceof HTMLElement ? [target] : target;
    const { matchId = generateUniqueId() } = options;
    const processedPatterns = patternsArray.map((pattern, index) => {
      if (typeof pattern === "string") {
        return { match: pattern, color: colorScheme[index % colorScheme.length] };
      }
      return {
        match: pattern.match,
        color: pattern.color || colorScheme[index % colorScheme.length]
      };
    });
    elements.forEach((element) => {
      const text = element.textContent || "";
      const allMatches = processedPatterns.flatMap((pattern, index) => {
        const subPatterns = pattern.match.split(",").map((p) => p.trim()).filter((p) => p !== "");
        return findMatchesForPatterns(text, subPatterns, `${matchId}-${index}`).map((match) => [...match, `--bylight-color: ${pattern.color};`]);
      });
      if (allMatches.length > 0) {
        element.innerHTML = `<code>${applyHighlights(text, allMatches)}</code>`;
      }
    });
  }
  function applyHighlights(text, matches) {
    matches.sort((a, b) => b[0] - a[0]);
    return matches.reduce((result, [start, end, matchId, styleString]) => {
      const beforeMatch = result.slice(0, start);
      const matchContent = result.slice(start, end);
      const afterMatch = result.slice(end);
      return beforeMatch + `<span class="bylight-code" style="${styleString}" data-match-id="${matchId}">` + matchContent + "</span>" + afterMatch;
    }, text);
  }
  function processLinksAndHighlight(targetElement, colorScheme = DefaultColors) {
    const elements = targetElement.querySelectorAll('pre, a[href^="bylight"]');
    const preMap = /* @__PURE__ */ new Map();
    const linkMap = /* @__PURE__ */ new Map();
    const colorMap = /* @__PURE__ */ new Map();
    let colorIndex = 0;
    elements.forEach((element, index) => {
      if (element.tagName === "PRE") {
        preMap.set(element, []);
      } else if (element.tagName === "A") {
        const anchorElement = element;
        const linkData = processAnchorElement(anchorElement, index, colorScheme, colorIndex);
        linkMap.set(anchorElement, linkData);
        colorMap.set(linkData.matchId, colorIndex);
        colorIndex = (colorIndex + 1) % colorScheme.length;
      }
    });
    linkMap.forEach(
      ({ targetIndices, patterns, index, matchId, color }, linkElement) => {
        const findMatchingPres = (indices, index2) => {
          if (indices === "all") {
            return Array.from(preMap.keys());
          }
          if (indices === "up" || indices === "down") {
            return findPreElementsInDirection(elements, index2, indices, parseInt(indices));
          }
          return indices.map((offset) => findPreElementByOffset(elements, index2, offset)).filter((el) => el !== null);
        };
        const matchingPres = findMatchingPres(targetIndices, index);
        matchingPres.forEach((matchingPre) => {
          var _a;
          const text = matchingPre.textContent || "";
          const newMatches = findMatchesForPatterns(text, patterns, matchId);
          (_a = preMap.get(matchingPre)) == null ? void 0 : _a.push(...newMatches);
        });
      }
    );
    preMap.forEach((matches, preElement) => {
      if (matches.length > 0) {
        const text = preElement.textContent || "";
        const allMatches = matches.map(
          ([start, end, matchId]) => {
            const linkData = Array.from(linkMap.values()).find((data) => data.matchId === matchId);
            const color = (linkData == null ? void 0 : linkData.color) || colorScheme[colorMap.get(matchId) || 0];
            return [
              start,
              end,
              matchId,
              `--bylight-color: ${color};`
            ];
          }
        );
        preElement.innerHTML = `<code>${applyHighlights(text, allMatches)}</code>`;
      }
    });
    linkMap.forEach((linkData, linkElement) => {
      var _a;
      const { matchId, color } = linkData;
      const finalColor = color || colorScheme[colorMap.get(matchId) || 0];
      const spanElement = document.createElement("span");
      spanElement.innerHTML = linkElement.innerHTML;
      spanElement.dataset.matchId = matchId;
      spanElement.classList.add("bylight-link");
      spanElement.style.setProperty("--bylight-color", finalColor);
      (_a = linkElement.parentNode) == null ? void 0 : _a.replaceChild(spanElement, linkElement);
    });
  }
  function processAnchorElement(anchorElement, index, colorScheme, colorIndex) {
    const url = new URL(anchorElement.href);
    const matchId = generateUniqueId();
    const inParam = url.searchParams.get("in");
    const dirParam = url.searchParams.get("dir");
    const color = url.searchParams.get("color") || colorScheme[colorIndex];
    const targetIndices = getTargetIndices(inParam, dirParam);
    anchorElement.addEventListener("click", (e) => e.preventDefault());
    return {
      targetIndices,
      patterns: (url.searchParams.get("match") || anchorElement.textContent || "").split(","),
      index,
      matchId,
      color
    };
  }
  function getTargetIndices(inParam, dirParam) {
    if (inParam) {
      return inParam === "all" ? "all" : inParam.split(",").map(Number);
    } else if (dirParam) {
      return dirParam;
    }
    return [1];
  }
  function findPreElementsInDirection(elements, startIndex, direction, count) {
    const dir = direction === "up" ? -1 : 1;
    const matchingPres = [];
    let preCount = 0;
    for (let i = startIndex + dir; i >= 0 && i < elements.length; i += dir) {
      if (elements[i].tagName === "PRE") {
        matchingPres.push(elements[i]);
        preCount++;
        if (preCount === count) break;
      }
    }
    return matchingPres;
  }
  function findPreElementByOffset(elements, startIndex, offset) {
    let preCount = 0;
    const dir = Math.sign(offset);
    for (let i = startIndex + dir; i >= 0 && i < elements.length; i += dir) {
      if (elements[i].tagName === "PRE") {
        preCount++;
        if (preCount === Math.abs(offset)) {
          return elements[i];
        }
      }
    }
    return null;
  }
  function addHoverEffect(targetElement) {
    targetElement.addEventListener("mouseover", (event) => {
      const target = event.target;
      if (target.dataset.matchId) {
        const matchId = target.dataset.matchId;
        const elements = targetElement.querySelectorAll(
          `[data-match-id="${matchId}"]`
        );
        elements.forEach((el) => {
          el.classList.add("bylight-hover");
        });
      }
    });
    targetElement.addEventListener("mouseout", (event) => {
      const target = event.target;
      if (target.dataset.matchId) {
        const matchId = target.dataset.matchId;
        const elements = targetElement.querySelectorAll(
          `[data-match-id="${matchId}"]`
        );
        elements.forEach((el) => {
          el.classList.remove("bylight-hover");
        });
      }
    });
  }
  function bylight(options = {}) {
    const { target = "body", colorScheme = DefaultColors } = options;
    const targetElement = typeof target === "string" ? document.querySelector(target) : target;
    if (!targetElement) {
      console.error(`bylight: Target element not found - ${target}`);
      return;
    }
    processLinksAndHighlight(targetElement, colorScheme);
    addHoverEffect(targetElement);
  }
  bylight.highlight = highlight;
  bylight.processLinksAndHighlight = processLinksAndHighlight;
  bylight.addHoverEffect = addHoverEffect;
  bylight.findMatches = findMatches;
  bylight.findRegexMatches = findRegexMatches;
  bylight.escapeRegExp = escapeRegExp;
  bylight.DefaultColors = DefaultColors;
  return __toCommonJS(src_exports);
})();
window.bylight = bylight.default || bylight;
//# sourceMappingURL=index.global.js.map
