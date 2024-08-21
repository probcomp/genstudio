(function () {
  const TIMEOUT = 2000; // ms
  let timeoutId;

  function scrollToHash() {
    const hash = window.location.hash.substring(1);
    const target = document.getElementById(hash);

    if (!target) {
      console.warn(`[ScrollFix] Target element not found for hash: ${hash}`);
      return;
    }

    function adjustScroll() {
      target.scrollIntoView({ behavior: "auto", block: "start" });
      console.log(`[ScrollFix] Scrolled to element with id: ${hash}`);
    }

    const observer = new MutationObserver(adjustScroll);
    const observerOptions = {
      childList: true,
      subtree: true,
      attributes: true,
      characterData: true,
    };

    observer.observe(document.body, observerOptions);

    function resetTimeout() {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        observer.disconnect();
        console.log("[ScrollFix] Stopped observing mutations after timeout");
      }, TIMEOUT);
    }

    resetTimeout();

    // Initial scroll attempt
    adjustScroll();

    // Set up onResourceLoad handler
    window.require.onResourceLoad = function () {
      console.log("[ScrollFix] Resource loaded, resetting timeout");
      resetTimeout();
    };
  }

  // Initialize the scroll fix when the DOM content is loaded
  document.addEventListener("DOMContentLoaded", () => {
    if (window.location.hash) {
      console.log(
        `[ScrollFix] Initializing scroll fix for hash: ${window.location.hash}`,
      );
      scrollToHash();
    } else {
      console.log("[ScrollFix] No hash in URL, scroll fix not initialized");
    }
  });
})();
