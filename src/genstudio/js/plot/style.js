// internal utils copied from @observablehq/plot/src/style.js

export function applyAttr(selection, name, value) {
    if (value != null) selection.attr(name, value);
}

export function applyStyle(selection, name, value) {
    if (value != null) selection.style(name, value);
}

export function applyTransform(selection, mark, { x, y }, tx = offset, ty = offset) {
    tx += mark.dx;
    ty += mark.dy;
    if (x?.bandwidth) tx += x.bandwidth() / 2;
    if (y?.bandwidth) ty += y.bandwidth() / 2;
    if (tx || ty) selection.attr("transform", `translate(${tx},${ty})`);
}

export function impliedString(value, impliedValue) {
    if ((value = string(value)) !== impliedValue) return value;
}

export function impliedNumber(value, impliedValue) {
    if ((value = number(value)) !== impliedValue) return value;
}

function applyHref(selection, href, target) {
    selection.each(function (i) {
        const h = href(i);
        if (h != null) {
            const a = this.ownerDocument.createElementNS(namespaces.svg, "a");
            a.setAttribute("fill", "inherit");
            a.setAttributeNS(namespaces.xlink, "href", h);
            if (target != null) a.setAttribute("target", target);
            this.parentNode.insertBefore(a, this).appendChild(this);
        }
    });
}

export function applyTitle(selection, L) {
    if (L)
        selection
            .filter((i) => nonempty(L[i]))
            .append("title")
            .call(applyText, L);
}

export function applyChannelStyles(
    selection,
    { target, tip },
    {
        ariaLabel: AL,
        title: T,
        fill: F,
        fillOpacity: FO,
        stroke: S,
        strokeOpacity: SO,
        strokeWidth: SW,
        opacity: O,
        href: H
    }
) {
    if (AL) applyAttr(selection, "aria-label", (i) => AL[i]);
    if (F) applyAttr(selection, "fill", (i) => F[i]);
    if (FO) applyAttr(selection, "fill-opacity", (i) => FO[i]);
    if (S) applyAttr(selection, "stroke", (i) => S[i]);
    if (SO) applyAttr(selection, "stroke-opacity", (i) => SO[i]);
    if (SW) applyAttr(selection, "stroke-width", (i) => SW[i]);
    if (O) applyAttr(selection, "opacity", (i) => O[i]);
    if (H) applyHref(selection, (i) => H[i], target);
    if (!tip) applyTitle(selection, T);
}

// Note: may mutate selection.node!
function applyClip(selection, mark, dimensions, context) {
    let clipUrl;
    const {clip = context.clip} = mark;
    switch (clip) {
      case "frame": {
        const {width, height, marginLeft, marginRight, marginTop, marginBottom} = dimensions;
        const id = getClipId();
        clipUrl = `url(#${id})`;
        selection = create("svg:g", context)
          .call((g) =>
            g
              .append("svg:clipPath")
              .attr("id", id)
              .append("rect")
              .attr("x", marginLeft)
              .attr("y", marginTop)
              .attr("width", width - marginRight - marginLeft)
              .attr("height", height - marginTop - marginBottom)
          )
          .each(function () {
            this.appendChild(selection.node());
            selection.node = () => this; // Note: mutation!
          });
        break;
      }
      case "sphere": {
        const {projection} = context;
        if (!projection) throw new Error(`the "sphere" clip option requires a projection`);
        const id = getClipId();
        clipUrl = `url(#${id})`;
        selection
          .append("clipPath")
          .attr("id", id)
          .append("path")
          .attr("d", geoPath(projection)({type: "Sphere"}));
        break;
      }
    }
    // Here we’re careful to apply the ARIA attributes to the outer G element when
    // clipping is applied, and to apply the ARIA attributes before any other
    // attributes (for readability).
    applyAttr(selection, "aria-label", mark.ariaLabel);
    applyAttr(selection, "aria-description", mark.ariaDescription);
    applyAttr(selection, "aria-hidden", mark.ariaHidden);
    applyAttr(selection, "clip-path", clipUrl);
  }

// Note: may mutate selection.node!
export function applyIndirectStyles(selection, mark, dimensions, context) {
    applyClip(selection, mark, dimensions, context);
    applyAttr(selection, "class", mark.className);
    applyAttr(selection, "fill", mark.fill);
    applyAttr(selection, "fill-opacity", mark.fillOpacity);
    applyAttr(selection, "stroke", mark.stroke);
    applyAttr(selection, "stroke-width", mark.strokeWidth);
    applyAttr(selection, "stroke-opacity", mark.strokeOpacity);
    applyAttr(selection, "stroke-linejoin", mark.strokeLinejoin);
    applyAttr(selection, "stroke-linecap", mark.strokeLinecap);
    applyAttr(selection, "stroke-miterlimit", mark.strokeMiterlimit);
    applyAttr(selection, "stroke-dasharray", mark.strokeDasharray);
    applyAttr(selection, "stroke-dashoffset", mark.strokeDashoffset);
    applyAttr(selection, "shape-rendering", mark.shapeRendering);
    applyAttr(selection, "filter", mark.imageFilter);
    applyAttr(selection, "paint-order", mark.paintOrder);
    const { pointerEvents = context.pointerSticky === false ? "none" : undefined } = mark;
    applyAttr(selection, "pointer-events", pointerEvents);
}

export function applyDirectStyles(selection, mark) {
    applyStyle(selection, "mix-blend-mode", mark.mixBlendMode);
    applyAttr(selection, "opacity", mark.opacity);
}
