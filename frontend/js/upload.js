/**
 * Handles document upload: file picker + drag-drop, both funnel into the
 * same upload flow. Shows chunks_indexed as confirmation the document was
 * actually OCR'd/parsed and made it into the vector store.
 */

const GroundwireUpload = {
  dropzoneEl: null,
  fileInputEl: null,
  docListEl: null,

  init() {
    this.dropzoneEl = document.getElementById('dropzone');
    this.fileInputEl = document.getElementById('file-input');
    this.docListEl = document.getElementById('doc-list');

    this.fileInputEl.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) this._handleFile(file);
      this.fileInputEl.value = ''; // allow re-uploading the same filename later
    });

    this.dropzoneEl.addEventListener('dragover', (e) => {
      e.preventDefault();
      this.dropzoneEl.classList.add('is-dragover');
    });

    this.dropzoneEl.addEventListener('dragleave', () => {
      this.dropzoneEl.classList.remove('is-dragover');
    });

    this.dropzoneEl.addEventListener('drop', (e) => {
      e.preventDefault();
      this.dropzoneEl.classList.remove('is-dragover');
      const file = e.dataTransfer.files[0];
      if (file) this._handleFile(file);
    });
  },

  async _handleFile(file) {
    const itemEl = this._addDocItem(file.name, 'Uploading…', 'is-uploading');

    try {
      const result = await GroundwireAPI.uploadDocument(file);
      this._updateDocItem(itemEl, `${result.chunks_indexed} chunks`, '');
    } catch (err) {
      this._updateDocItem(itemEl, 'Failed', 'is-error');
      console.error('Upload error:', err);
    }
  },

  _addDocItem(name, metaText, stateClass) {
    const li = document.createElement('li');
    li.className = `doc-item ${stateClass}`;
    li.innerHTML = `
      <span class="doc-name" title="${this._escape(name)}">${this._escape(name)}</span>
      <span class="doc-meta">${this._escape(metaText)}</span>
    `;
    this.docListEl.prepend(li);
    return li;
  },

  _updateDocItem(itemEl, metaText, stateClass) {
    itemEl.className = `doc-item ${stateClass}`;
    itemEl.querySelector('.doc-meta').textContent = metaText;
  },

  _escape(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  },
};
