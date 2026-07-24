(() => {
  const track = document.getElementById("hero-track");
  const dotsContainer = document.getElementById("hero-dots");
  const input = document.getElementById("question");
  const heroViewport = track ? track.closest(".hero-viewport") : null;

  if (!track || !dotsContainer) return;

  const panels = Array.from(track.querySelectorAll(".hero-panel"));
  if (panels.length === 0) return;

  // Each era gets its own accent so a slide change is unmistakable even out
  // of the corner of your eye, not just a same-color crossfade.
  const ERA_ACCENTS = {
    "rome": "#B5533C",
    "silk-road": "#C08A2E",
    "feudal-japan": "#A6332C",
    "islamic-golden-age": "#2E8B84",
    "mesoamerica": "#C1652F",
    "medieval-europe": "#3B5BA5",
  };

  function applyAccent(panel) {
    if (!heroViewport) return;
    const color = ERA_ACCENTS[panel.dataset.era] || "";
    heroViewport.style.setProperty("--hero-accent", color || "var(--green)");
  }

  const AUTO_ADVANCE_MS = 4000;
  let activeIndex = 0;
  let timerId = null;

  // Build one dot per panel.
  const dots = panels.map((panel, i) => {
    const dot = document.createElement("button");
    dot.type = "button";
    dot.className = "hero-dot";
    dot.setAttribute("role", "tab");
    dot.setAttribute("aria-label", `Show ${panel.querySelector("h2")?.textContent || `slide ${i + 1}`}`);
    dot.addEventListener("click", () => goTo(i, { restartTimer: true }));
    dotsContainer.appendChild(dot);
    return dot;
  });

  function render() {
    panels.forEach((panel, i) => {
      panel.classList.toggle("is-active", i === activeIndex);
      panel.setAttribute("aria-hidden", i === activeIndex ? "false" : "true");
    });
    dots.forEach((dot, i) => {
      dot.classList.toggle("is-active", i === activeIndex);
      dot.setAttribute("aria-selected", i === activeIndex ? "true" : "false");
    });
    applyAccent(panels[activeIndex]);
  }

  function goTo(index, { restartTimer = false } = {}) {
    activeIndex = (index + panels.length) % panels.length;
    render();
    if (restartTimer) startTimer();
  }

  function next() {
    goTo(activeIndex + 1);
  }

  function startTimer() {
    stopTimer();
    // Note: we still auto-advance even when prefers-reduced-motion is set --
    // the CSS media query already removes the crossfade animation itself, and
    // pause-on-hover/focus below gives users a way to stop it, which covers
    // the accessibility concern without killing the slideshow outright.
    timerId = window.setInterval(next, AUTO_ADVANCE_MS);
  }

  function stopTimer() {
    if (timerId) {
      window.clearInterval(timerId);
      timerId = null;
    }
  }

  // Clicking a panel fills the question input with its suggested prompt,
  // but never auto-submits -- matches the comment already in index.html.
  panels.forEach((panel) => {
    panel.addEventListener("click", () => {
      const question = panel.dataset.question;
      if (question && input) {
        input.value = question;
        input.focus();
      }
    });
    panel.style.cursor = panel.dataset.question ? "pointer" : "";
  });

  // Pause auto-advance while the user is interacting with the carousel,
  // resume when they leave it alone.
  const heroSection = track.closest(".hero") || track;
  heroSection.addEventListener("mouseenter", stopTimer);
  heroSection.addEventListener("mouseleave", () => startTimer());
  heroSection.addEventListener("focusin", stopTimer);
  heroSection.addEventListener("focusout", () => startTimer());

  render();
  startTimer();
})();