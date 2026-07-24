(() => {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const transcript = document.getElementById("transcript");

  function scrollToLatest() {
    transcript.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  // Defensive fallback: the backend prompt asks Gemini for plain prose, but
  // models don't always obey formatting instructions perfectly. This strips
  // stray Markdown so it never leaks into the UI as literal asterisks, and
  // recasts stray bullets as em dashes to fit the manuscript voice.
  function formatAnswer(text) {
    return text
      .replace(/\*\*(.*?)\*\*/g, "$1")   // strip bold markers, keep the text
      .replace(/\*(.*?)\*/g, "$1")       // strip stray italic markers
      .replace(/^\s*[*-]\s+/gm, "— ")    // turn stray bullets into an em dash
      .trim();
  }

  function addEntry(kind, text) {
    const li = document.createElement("li");
    li.className = `entry entry--${kind}`;

    const bubble = document.createElement(kind === "user" ? "span" : "div");
    bubble.className = "bubble";

    if (kind === "user") {
      // User input is trusted as plain text only — textContent, never innerHTML,
      // so nothing typed here can ever be interpreted as markup.
      bubble.textContent = text;
    } else {
      // Split into real paragraphs instead of one text blob. Each paragraph
      // becomes its own <p>; CSS applies the illuminated drop-cap only to the
      // first one (see .entry--bot .bubble p:first-child::first-letter).
      const paragraphs = formatAnswer(text).split(/\n{2,}/).filter(Boolean);
      if (paragraphs.length === 0) {
        paragraphs.push(text);
      }
      paragraphs.forEach((para) => {
        const p = document.createElement("p");
        p.textContent = para;
        bubble.appendChild(p);
      });
    }

    li.appendChild(bubble);
    transcript.appendChild(li);
    scrollToLatest();
    return li;
  }

  function addPendingEntry() {
    const li = document.createElement("li");
    li.className = "entry entry--pending";
    li.innerHTML = `
      <div class="bubble">
        <span class="quill"></span>
        <span>Consulting the archive…</span>
      </div>`;
    transcript.appendChild(li);
    scrollToLatest();
    return li;
  }

  async function askQuestion(question) {
    submitBtn.disabled = true;
    const pending = addPendingEntry();

    try {
      const res = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await res.json();
      pending.remove();

      if (!res.ok) {
        addEntry("error", data.error || "Something went wrong reaching the archive.");
        return;
      }

      addEntry("bot", data.answer);
    } catch (err) {
      pending.remove();
      addEntry("error", "Couldn't reach the archive. Check your connection and try again.");
    } finally {
      submitBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    addEntry("user", question);
    input.value = "";
    askQuestion(question);
  });
})();