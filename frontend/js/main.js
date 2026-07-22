/**
 * App entry point: holds the currently-selected provider, wires the
 * composer form to the chat + API modules, and initializes upload handling.
 */

(function () {
  let currentProvider = 'gemini';

  function initProviderSwitch() {
    const buttons = document.querySelectorAll('.provider-btn');
    const hint = document.getElementById('provider-hint');

    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        buttons.forEach((b) => {
          b.classList.remove('is-active');
          b.setAttribute('aria-checked', 'false');
        });
        btn.classList.add('is-active');
        btn.setAttribute('aria-checked', 'true');

        currentProvider = btn.dataset.provider;
        const label = currentProvider === 'groq' ? 'Groq' : 'Gemini';
        hint.textContent = `Every call for this turn runs on ${label}.`;
      });
    });
  }

  function initComposer() {
    const form = document.getElementById('composer');
    const input = document.getElementById('query-input');
    const sendBtn = document.getElementById('send-btn');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const text = input.value.trim();
      if (!text) return;

      GroundwireChat.addUserTurn(text);
      input.value = '';
      input.disabled = true;
      sendBtn.disabled = true;
      GroundwireChat.addThinkingTurn();

      try {
        const result = await GroundwireAPI.query(text, currentProvider);
        GroundwireChat.removeThinkingTurn();
        GroundwireChat.addAgentTurn(result);
      } catch (err) {
        GroundwireChat.removeThinkingTurn();
        GroundwireChat.addErrorTurn(`Request failed: ${err.message}`
        );
        console.error('Query error:', err);
      } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    GroundwireChat.init();
    GroundwireUpload.init();
    initProviderSwitch();
    initComposer();
  });
})();
