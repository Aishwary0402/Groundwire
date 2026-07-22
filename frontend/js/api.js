/**
 * All network calls to the Groundwire backend live here, and nowhere else.
 * If the backend URL or auth ever changes, this is the only file to touch.
 */

const API_BASE = '';

const GroundwireAPI = {
  async uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Upload failed (${response.status}): ${errorBody}`);
    }

    return response.json(); // { filename, chunks_indexed }
  },

  async query(queryText, llmProvider) {
    const response = await fetch(`${API_BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: queryText, llm_provider: llmProvider }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Query failed (${response.status}): ${errorBody}`);
    }

    return response.json();
    // { decision, decision_reasoning, answer, clarification_question,
    //   low_confidence_caveat, citations, retry_count }
  },
};
