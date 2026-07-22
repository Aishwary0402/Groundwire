/**
 * Renders the chat trace: a user turn, then an agent turn styled according
 * to which of the four Decision Router outcomes actually fired. This is
 * where the self-correction behavior becomes visible, not just logged.
 */

const GroundwireChat = {
  traceEl: null,
  emptyStateEl: null,

  init() {
    this.traceEl = document.getElementById('chat-trace');
    this.emptyStateEl = document.getElementById('empty-state');
  },

  _hideEmptyState() {
    if (this.emptyStateEl) {
      this.emptyStateEl.remove();
      this.emptyStateEl = null;
    }
  },

  addUserTurn(text) {
    this._hideEmptyState();

    const turn = document.createElement('div');
    turn.className = 'turn turn-user';
    turn.innerHTML = `
      <div class="turn-node glow-user"></div>
      <div class="bubble">${this._escape(text)}</div>
    `;
    this.traceEl.appendChild(turn);
    this._scrollToBottom();
  },

  addThinkingTurn() {
  const turn = document.createElement('div');
  turn.className = 'turn turn-thinking';
  turn.id = 'thinking-turn';
  turn.innerHTML = `
    <div class="turn-node"></div>
    <div class="thinking-scan"></div>
  `;
  this.traceEl.appendChild(turn);
  this._scrollToBottom();
},

  removeThinkingTurn() {
    const el = document.getElementById('thinking-turn');
    if (el) el.remove();
  },

  addAgentTurn(result) {
    const decision = result.decision || 'final_answer';
    const config = this._configFor(decision, result.retry_count);

    const turn = document.createElement('div');
    turn.className = 'turn';
    turn.innerHTML = `
  <div class="turn-node ${config.nodeClass}" data-tooltip="${this._escape(result.decision_reasoning || '')}"></div>
  <div class="bubble ${config.bubbleClass}">
        <span class="badge ${config.badgeClass}">${config.badgeLabel}</span>
        ${this._renderBody(decision, result)}
        ${this._renderCitations(result.citations)}
        ${this._renderRetryNote(result.retry_count)}
      </div>
    `;
    this.traceEl.appendChild(turn);
    this._scrollToBottom();
  },

  addErrorTurn(message) {
    const turn = document.createElement('div');
    turn.className = 'turn';
    turn.innerHTML = `
      <div class="turn-node glow-low"></div>
      <div class="bubble bubble-answer is-low">
        <span class="badge badge-low">Connection issue</span>
        ${this._escape(message)}
      </div>
    `;
    this.traceEl.appendChild(turn);
    this._scrollToBottom();
  },

  _configFor(decision, retryCount) {
    if (decision === 'clarify') {
      return {
        nodeClass: 'glow-clarify',
        bubbleClass: 'bubble-clarify',
        badgeClass: 'badge-clarify',
        badgeLabel: 'Clarification needed',
      };
    }
    if (decision === 'low_confidence') {
      return {
        nodeClass: 'glow-low',
        bubbleClass: 'bubble-answer is-low',
        badgeClass: 'badge-low',
        badgeLabel: 'Low confidence',
      };
    }
    if (retryCount && retryCount > 0) {
      return {
        nodeClass: 'glow-retry',
        bubbleClass: 'bubble-answer is-retry',
        badgeClass: 'badge-retry',
        badgeLabel: `Verified · reformulated ${retryCount}×`,
      };
    }
    return {
      nodeClass: 'glow-final',
      bubbleClass: 'bubble-answer',
      badgeClass: 'badge-final',
      badgeLabel: 'Verified',
    };
  },

  _renderBody(decision, result) {
  if (decision === 'clarify') {
    return `<p>${this._escape(result.clarification_question || 'Could you narrow that down?')}</p>`;
  }

  const answerText = result.answer || '(no answer returned)';
  let body = marked.parse(answerText);

  if (decision === 'low_confidence' && result.low_confidence_caveat) {
    body += `<div class="caveat">⚠ ${this._escape(result.low_confidence_caveat)}</div>`;
  }

  return body;
},

  _renderCitations(citations) {
    if (!citations || citations.length === 0) return '';
    const chips = citations.map(id => `<span class="citation-chip">${this._escape(id)}</span>`).join('');
    return `<div class="citations">${chips}</div>`;
  },

  _renderRetryNote(retryCount) {
    if (!retryCount || retryCount === 0) return '';
    return `<div class="retry-note">↻ query reformulated ${retryCount} time${retryCount > 1 ? 's' : ''} before this response</div>`;
  },

  _scrollToBottom() {
    const scrollEl = document.getElementById('chat-scroll');
    scrollEl.scrollTop = scrollEl.scrollHeight;
  },

  _escape(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};
