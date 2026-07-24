(() => {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question");
  const submitBtn = document.getElementById("submit-btn");
  const transcript = document.getElementById("transcript");

  function scrollToLatest() {
    transcript.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function addEntry(kind, text) {
    const li = document.createElement("li");
    li.className = `entry entry--${kind}`;

    const bubble = document.createElement(kind === "user" ? "span" : "div");
    bubble.className = "bubble";

    if (kind === "user") {
      bubble.textContent = text;
    } else {
      const p = document.createElement("p");
      p.textContent = text;
      bubble.appendChild(p);
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