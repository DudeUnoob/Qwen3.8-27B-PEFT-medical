const $ = (id) => document.getElementById(id);
let csrf = '', messages = [], controller = null;
const setStatus = (text) => { $('status').textContent = text; };
const showError = (text) => { $('page-error').textContent = text; $('page-error').hidden = !text; };
function signedOut() {
  $('login-view').hidden = false;
  for (const id of ['chat-view', 'account', 'new-chat']) $(id).hidden = true;
  messages = []; $('messages').replaceChildren();
}
function addMessage(role, text) {
  const article = document.createElement('article'); article.className = `message ${role}`;
  const label = document.createElement('div'); label.className = 'label'; label.textContent = role === 'user' ? 'YOU' : 'QWEN MEDICAL';
  const content = document.createElement('div'); content.className = 'content'; content.textContent = text;
  article.append(label, content); $('messages').append(article); return content;
}
function scrollChat() { $('messages').scrollTop = $('messages').scrollHeight; }
function reset() {
  if (controller) return;
  messages = []; $('messages').replaceChildren(); $('welcome').hidden = false;
  showError(''); setStatus('Ready when you are'); $('prompt').focus();
}
async function initialize() {
  try {
    const response = await fetch('/api/me');
    if (response.status === 401) { signedOut(); return; }
    if (!response.ok) throw new Error('Unable to connect. Please refresh the page.');
    const user = await response.json(); csrf = user.csrf; $('username').textContent = user.login;
    for (const option of $('api-version').options) option.disabled = !user.api_versions.includes(option.value);
    if ($('api-version').selectedOptions[0]?.disabled) $('api-version').value = user.api_versions[0] || 'v1';
    $('login-view').hidden = true;
    for (const id of ['chat-view', 'account', 'new-chat']) $(id).hidden = false;
    $('prompt').focus();
  } catch (error) { showError(error.message); }
}
$('composer').addEventListener('submit', async (event) => {
  event.preventDefault();
  const prompt = $('prompt').value.trim(); if (!prompt || controller) return;
  showError(''); $('welcome').hidden = true;
  messages.push({role: 'user', content: prompt}); addMessage('user', prompt);
  const output = addMessage('assistant', '');
  $('prompt').value = ''; $('prompt').style.height = ''; $('prompt').disabled = true;
  $('send').hidden = true; $('stop').hidden = false; $('api-version').disabled = true;
  const apiVersion = $('api-version').value;
  controller = new AbortController(); const started = performance.now(); let first = null, answer = '', done = false;
  setStatus('Connecting to your model… The first response may take a few minutes.'); scrollChat();
  try {
    const response = await fetch('/api/chat', {method: 'POST', signal: controller.signal,
      headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf}, body: JSON.stringify({messages, api_version: apiVersion})});
    if (response.status === 401) { signedOut(); throw new Error('Your session expired. Please sign in again.'); }
    if (!response.ok) throw new Error((await response.json()).detail || 'Unable to send message.');
    const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = '';
    while (true) {
      const chunk = await reader.read(); if (chunk.done) break;
      pending += decoder.decode(chunk.value, {stream: true});
      let newline;
      while ((newline = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, newline); pending = pending.slice(newline + 1); if (!line.trim()) continue;
        const item = JSON.parse(line);
        if (item.type === 'error') throw new Error(item.message);
        if (item.type === 'token') {
          if (first === null) first = (performance.now() - started) / 1000;
          answer += item.text; output.textContent = answer; setStatus('Generating…'); scrollChat();
        }
        if (item.type === 'done') {
          done = true; messages.push({role: 'assistant', content: answer});
          const seconds = ((performance.now() - started) / 1000).toFixed(1);
          setStatus(`API ${apiVersion} · ${seconds}s total${first === null ? '' : ` · ${first.toFixed(1)}s to first token`}${item.usage ? ` · ${item.usage.completion_tokens} output tokens` : ''}`);
        }
      }
    }
    if (!done) throw new Error('The connection ended early. Please try again.');
  } catch (error) {
    if (!done) messages.pop();
    setStatus(error.name === 'AbortError' ? 'Generation stopped' : 'Unable to complete response');
    if (error.name !== 'AbortError') showError(error.message);
    if (!answer) output.textContent = error.name === 'AbortError' ? 'Stopped.' : 'No response received.';
  } finally {
    controller = null; $('prompt').disabled = false; $('api-version').disabled = false; $('send').hidden = false; $('stop').hidden = true; $('prompt').focus();
  }
});
$('prompt').addEventListener('input', () => { $('prompt').style.height = ''; $('prompt').style.height = Math.min($('prompt').scrollHeight, 150) + 'px'; });
$('prompt').addEventListener('keydown', (event) => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } });
$('stop').addEventListener('click', () => controller?.abort());
$('new-chat').addEventListener('click', reset);
for (const button of document.querySelectorAll('[data-prompt]')) button.addEventListener('click', () => { $('prompt').value = button.dataset.prompt; $('composer').requestSubmit(); });
$('logout').addEventListener('click', async () => {
  controller?.abort();
  try {
    const response = await fetch('/logout', {method: 'POST', headers: {'X-CSRF-Token': csrf}});
    if (!response.ok && response.status !== 401) throw new Error('Sign out failed. Please try again.');
    csrf = ''; signedOut();
  } catch (error) { showError(error.message); }
});
initialize();
