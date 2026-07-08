const chatArea = document.getElementById('chatArea');
const questionInput = document.getElementById('questionInput');
const sendBtn = document.getElementById('sendBtn');
const faqList = document.getElementById('faqList');
const faqSearch = document.getElementById('faqSearch');

let allFaqs = [];

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function appendUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'message user-message';
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatArea.appendChild(div);
  scrollToBottom();
}

function appendBotMessage(data) {
  const div = document.createElement('div');
  div.className = 'message bot-message' + (data.emergency ? ' emergency' : '');

  let confClass = 'confidence-low';
  if (data.confidence_level === 'high') confClass = 'confidence-high';
  else if (data.confidence_level === 'medium') confClass = 'confidence-medium';

  let html = `<div class="bubble">`;
  html += `${data.emergency ? '🚨 ' : '🤖 '}${escapeHtml(data.answer)}`;
  html += `<div class="meta">Confidence: ${data.confidence}%
            <span class="confidence-tag ${confClass}">${data.confidence_level.toUpperCase()}</span></div>`;
  if (data.matched_question) {
    html += `<div class="meta">Matched: "${escapeHtml(data.matched_question)}"</div>`;
  }
  if (data.related_questions && data.related_questions.length > 0) {
    html += `<div class="related">You might also ask:`;
    data.related_questions.forEach(r => {
      html += `<span class="related-item" data-question="${escapeHtml(r.question)}">${escapeHtml(r.question)} (${r.confidence}%)</span>`;
    });
    html += `</div>`;
  }
  html += `</div>`;
  div.innerHTML = html;
  chatArea.appendChild(div);

  div.querySelectorAll('.related-item').forEach(el => {
    el.addEventListener('click', () => {
      questionInput.value = el.dataset.question;
      sendQuestion();
    });
  });

  scrollToBottom();
}

function appendTyping() {
  const div = document.createElement('div');
  div.className = 'message bot-message typing-indicator';
  div.id = 'typingIndicator';
  div.innerHTML = `<div class="bubble">🤖 Analyzing query...</div>`;
  chatArea.appendChild(div);
  scrollToBottom();
}

function removeTyping() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendQuestion() {
  const question = questionInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionInput.value = '';
  sendBtn.disabled = true;
  appendTyping();

  try {
    const resp = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    });
    const data = await resp.json();
    removeTyping();

    if (!resp.ok) {
      appendBotMessage({
        answer: data.error || 'Something went wrong.',
        confidence: 0,
        confidence_level: 'low',
        matched_question: null,
        related_questions: [],
        emergency: false
      });
      return;
    }
    appendBotMessage(data);
  } catch (err) {
    removeTyping();
    appendBotMessage({
      answer: 'Connection to station AI lost. Please try again.',
      confidence: 0,
      confidence_level: 'low',
      matched_question: null,
      related_questions: [],
      emergency: false
    });
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', sendQuestion);
questionInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') sendQuestion();
});

document.getElementById('clearChatBtn').addEventListener('click', () => {
  chatArea.innerHTML = `<div class="message bot-message"><div class="bubble">🤖 Chat cleared. How can I help, crew member?</div></div>`;
});

async function loadFaqs() {
  try {
    const resp = await fetch('/faqs');
    const data = await resp.json();
    allFaqs = data.faqs;
    renderFaqs(allFaqs);
  } catch (e) {
    faqList.innerHTML = '<p>Could not load FAQs.</p>';
  }
}

function renderFaqs(faqs) {
  faqList.innerHTML = faqs.map(f =>
    `<div class="faq-item" data-question="${escapeHtml(f.q)}">${escapeHtml(f.q)}</div>`
  ).join('');

  faqList.querySelectorAll('.faq-item').forEach(el => {
    el.addEventListener('click', () => {
      questionInput.value = el.dataset.question;
      sendQuestion();
    });
  });
}

faqSearch.addEventListener('input', () => {
  const term = faqSearch.value.toLowerCase();
  const filtered = allFaqs.filter(f => f.q.toLowerCase().includes(term));
  renderFaqs(filtered);
});

loadFaqs();
