import { PipecatClient } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';

const $ = id => document.getElementById(id);
const labels = {listening:'Слушаю тебя', queued:'Реплика в очереди', verify:'Проверяю голос', whisper:'Распознаю слова', recognized:'Текст распознан', codex:'Жду ответ Codex', tts:'Готовлю голос', speaking:'Передаю звук', done:'Готова слушать', interrupted:'Ответ остановлен', rejected:'Реплика отклонена', error:'Нужна проверка'};
const metricNames = {pause:'Ожидание паузы',queue:'Очередь',verify:'Проверка голоса',whisper:'Whisper',codex:'Codex',tts:'Первое аудио'};
const traces = new Map();
let client, connected = false, muted = false, busy = false, latest = null, detailId = null;
let stageStarted = performance.now(), stageName = '', sessionVoice = 'piper';
const time = ms => Number.isFinite(ms) ? (ms < 1000 ? `${Math.round(ms)} мс` : `${(ms/1000).toFixed(1)} с`) : '—';
const clock = timestamp => new Date(timestamp).toLocaleTimeString('ru-RU', {hour:'2-digit',minute:'2-digit',second:'2-digit'});

function notice(text) { $('notice').textContent = text; $('notice').hidden = !text; }
function stage(name, reason = '') {
  if (stageName !== name) { stageStarted = performance.now(); stageName = name; }
  $('stage').textContent = labels[name] || name;
  $('stage-reason').textContent = reason;
}
function controls(ready) {
  connected = ready;
  $('verify-owner').disabled = !ready;
  for (const id of ['mute','stop','send','text-input']) $(id).disabled = !ready;
  $('connection').textContent = ready ? 'Подключена' : 'Не подключена';
  $('connection').classList.toggle('online', ready);
  $('connect').textContent = ready ? 'Отключить' : 'Подключить';
  $('listen-state').textContent = ready ? (muted ? 'Микрофон выключен' : '● Микрофон включён') : 'Микрофон отключён';
  if (!ready) { $('meter').value = -80; $('db').textContent = '—'; }
}
function devices(id, list) {
  const select = $(id), previous = select.value;
  select.replaceChildren(...list.map((d, index) => new Option(d.label || `Устройство ${index + 1}`, d.deviceId)));
  if ([...select.options].some(o => o.value === previous)) select.value = previous;
}
function message(role, text, traceId) {
  if (!text) return;
  $('empty')?.remove();
  const key = `${traceId}-${role}`;
  if (document.getElementById(key)) return;
  const list = $('messages'), atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 90;
  const article = document.createElement('article'); article.id = key; article.className = `message ${role}`;
  const label = document.createElement('div'); label.className = 'message-label';
  const name = document.createElement('strong'); name.textContent = role === 'user' ? 'Ты' : 'Диана';
  const stamp = document.createElement('span'); stamp.textContent = clock(Date.now()); label.append(name, stamp);
  const body = document.createElement('div'); body.className = 'message-text'; body.textContent = text;
  const link = document.createElement('button'); link.className = 'trace-link'; link.textContent = 'Причины и время ↗'; link.onclick = () => showDetails(traceId);
  article.append(label, body, link); list.append(article);
  while (list.children.length > 200) list.firstElementChild.remove();
  if (atBottom) list.scrollTop = list.scrollHeight;
}
function drawMetrics(trace) {
  const values = trace.metrics || {}, max = Math.max(1, ...Object.keys(metricNames).map(k => values[k] || 0));
  $('total').parentElement.nextElementSibling.textContent = trace.source === 'keyboard'
    ? 'От получения текста сервером до первого аудио в выходном тракте. Не включает задержку динамиков.'
    : 'От последнего обнаруженного участка речи до первого аудио в выходном тракте. Не включает задержку динамиков.';
  $('metrics').replaceChildren();
  for (const [key, name] of Object.entries(metricNames)) {
    const row = document.createElement('div'); row.className = 'metric';
    const label = document.createElement('span'); label.textContent = name;
    const value = document.createElement('span'); value.className = 'mono'; value.textContent = time(values[key]);
    const track = document.createElement('div'); track.className = 'metric-track';
    const fill = document.createElement('div'); fill.className = 'metric-fill'; fill.style.width = `${(values[key] || 0)/max*100}%`;
    track.append(fill); row.append(label,value,track); $('metrics').append(row);
  }
  $('total').textContent = time(values.to_audio);
}
function receive(data) {
  if (data.kind === 'verification') {
    $('verify-owner').checked = data.enabled;
    $('verify-owner').disabled = false;
    $('verification-status').textContent = data.enabled ? 'Проверка владельца включена' : 'Проверка выключена · принимается любая речь';
    return;
  }
  if (data.kind === 'meter') { $('meter').value = muted ? -80 : data.db; $('db').textContent = muted ? 'выкл.' : `${data.db.toFixed(0)} дБ`; return; }
  if (data.kind === 'session') {
    sessionVoice = data.voice;
    $('voice-label').textContent = data.voice === 'edge' ? 'Светлана · онлайн' : 'Ирина · локально';
    $('settings-voice').textContent = data.voice === 'edge' ? 'Светлана / запасной Piper' : 'Piper · Ирина';
    $('route-voice').textContent = data.voice === 'edge' ? 'Светлана' : 'Piper';
    $('thread').textContent = data.thread ? `Диалог …${data.thread.slice(-8)}` : 'Режим проверки';
    $('thread').title = data.thread || '';
    return;
  }
  if (data.kind !== 'trace') return;
  if (!traces.has(data.id)) latest = data.id;
  traces.set(data.id, data);
  if (traces.size > 100) traces.delete(traces.keys().next().value);
  message('user', data.text, data.id); message('assistant', data.answer, data.id);
  if (data.id === latest) {
    stage(data.stage, data.reason);
    $('trace-id').textContent = `…${data.id.slice(-6)}`;
    $('details-open').disabled = false;
    $('identity').textContent = data.source === 'keyboard' ? 'Введено с клавиатуры' : data.verify_owner === false ? 'Без проверки владельца' : data.score == null ? 'Голос ещё не проверен' : data.score >= data.threshold ? 'Голос подтверждён' : 'Голос не подтверждён';
    $('score').textContent = data.score == null ? '—' : data.score.toFixed(3);
    $('identity').parentElement.className = `identity ${data.score == null ? '' : data.score >= data.threshold ? 'accepted' : 'rejected'}`;
    drawMetrics(data);
    if (data.fallback) notice(data.fallback);
  }
  if ($('details').open && detailId === data.id) renderDetails(data);
}
function addFact(dl, key, value) {
  const dt = document.createElement('dt'), dd = document.createElement('dd');
  dt.textContent = key; dd.textContent = value; dl.append(dt,dd);
}
function renderDetails(trace) {
  $('detail-title').textContent = `Реплика …${trace.id.slice(-6)}`;
  const content = $('detail-content'); content.replaceChildren();
  const facts = document.createElement('dl');
  addFact(facts, 'Состояние', labels[trace.stage] || trace.stage);
  addFact(facts, 'Причина', trace.reason || '—');
  addFact(facts, 'Источник', trace.source === 'keyboard' ? 'Клавиатура · без проверки голоса' : 'Микрофон');
  if (trace.verify_owner === false) addFact(facts, 'Проверка владельца', 'Выключена для этой реплики');
  addFact(facts, 'Сходство / порог', trace.score == null ? 'Не измерено' : `${trace.score.toFixed(3)} / ${trace.threshold}`);
  addFact(facts, 'Диалог Codex', trace.thread || 'Запрос не отправлен');
  addFact(facts, 'Озвучка', trace.voice === 'edge' ? 'Светлана · Microsoft' : trace.voice === 'piper' ? 'Ирина · Piper' : 'Ещё не получена');
  if (trace.fallback) addFact(facts, 'Запасной голос', trace.fallback);
  if (trace.truncated) addFact(facts, 'Ограничение', 'Обработаны первые 20 секунд; хвост записи пропущен');
  addFact(facts, 'Передача данных', trace.thread ? 'Codex: текст реплики. Онлайн-озвучка: текст ответа, если выбран Edge.' : 'До отправки в Codex: обработка на ноутбуке.');
  content.append(facts);
  if (trace.text) { const text = document.createElement('p'); text.className = 'detail-text'; text.textContent = trace.text; content.append(text); }
  const list = document.createElement('ol'); list.className = 'timeline';
  for (const event of trace.events) {
    const li = document.createElement('li'), stamp = document.createElement('time'), description = document.createElement('span');
    stamp.textContent = clock(event.at * 1000); description.textContent = event.reason || labels[event.stage]; li.append(stamp,description); list.append(li);
  }
  content.append(list);
}
function showDetails(id) { const trace = traces.get(id); if (!trace) { notice('Эта реплика уже вышла из истории пульта (последние 100).'); return; } detailId = id; renderDetails(trace); if (!$('details').open) $('details').showModal(); }

function createClient() {
  return new PipecatClient({transport: new SmallWebRTCTransport(), enableCam:false, enableMic:true, callbacks:{
    onTransportStateChanged(state) {
      if (['connecting','connected','authenticating','authenticated'].includes(state)) stage('Подключаюсь…', 'Устанавливаю WebRTC и загружаю модели.');
      if (state === 'disconnected') { controls(false); stage('Отключена', 'Диалог Codex сохранён. Можно подключиться снова.'); }
    },
    onBotReady() { controls(true); notice(''); stage('Готова слушать', 'Скажи фразу и сделай паузу.'); },
    onDisconnected() { controls(false); },
    onError(error) { notice(`Ошибка соединения: ${error.message || JSON.stringify(error)}`); },
    onDeviceError(error) { notice(`Проверь доступ браузера к микрофону: ${error.message || error.type || 'ошибка устройства'}`); },
    onServerMessage: receive,
    onBotOutput(data) { if (/не поступает|не удалось|недоступна|Ошибка|Слишком коротко/i.test(data.text)) notice(data.text); },
    onTrackStarted(track, participant) {
      if (track.kind === 'audio' && !participant?.local) {
        $('audio').srcObject = new MediaStream([track]);
        $('audio').play().catch(() => { $('audio-unlock').hidden = false; notice('Браузер заблокировал звук. Нажми «Разрешить звук».'); });
      }
    },
    onAvailableMicsUpdated: list => devices('mic-select',list),
    onAvailableSpeakersUpdated: list => devices('speaker-select',list),
    onMicUpdated: mic => { $('mic-select').value = mic.deviceId; },
    onSpeakerUpdated: speaker => { $('speaker-select').value = speaker.deviceId; },
  }});
}
$('connect').onclick = async () => {
  if (busy) return; busy = true; $('connect').disabled = true;
  try {
    if (connected) { await client.disconnect(); controls(false); stage('Отключена'); }
    else {
      if (client) await client.disconnect().catch(() => {});
      client = createClient(); muted = false; $('mute').textContent = 'Выключить микрофон';
      stage('Подключаюсь…', 'Разреши доступ к микрофону.');
      await client.connect({webrtcUrl:`${location.origin}/api/offer`});
    }
  } catch (error) { notice(`Не удалось подключиться: ${error.message}. Если открыта /client/, отключи её.`); await client?.disconnect().catch(() => {}); controls(false); }
  finally { busy = false; $('connect').disabled = false; }
};
$('audio-unlock').onclick = async () => { try { await $('audio').play(); $('audio-unlock').hidden = true; notice(''); } catch { notice('Браузер всё ещё блокирует воспроизведение. Проверь разрешение звука для этой страницы.'); } };
$('mute').onclick = () => { muted = !muted; client.enableMic(!muted); $('mute').textContent = muted ? 'Включить микрофон' : 'Выключить микрофон'; controls(connected); };
$('stop').onclick = () => { client.sendClientMessage('diana.stop'); };
$('text-form').onsubmit = event => { event.preventDefault(); const text = $('text-input').value.trim(); if (!connected || !text) return; client.sendClientMessage('diana.text', {text}); $('text-input').value = ''; notice(''); };
$('text-input').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('text-form').requestSubmit(); } };
$('settings-open').onclick = () => $('settings').showModal();
$('verify-owner').onchange = () => { if (connected) { client.sendClientMessage('diana.verify', {enabled:$('verify-owner').checked}); $('verify-owner').disabled = true; } };
$('details-open').onclick = () => showDetails(latest);
for (const button of document.querySelectorAll('[data-close]')) button.onclick = () => $(button.dataset.close).close();
$('mic-select').onchange = async event => { try { await client?.updateMic(event.target.value); } catch (error) { notice(`Не удалось выбрать микрофон: ${error.message}`); } };
$('speaker-select').onchange = async event => { try { await $('audio').setSinkId(event.target.value); } catch (error) { notice(`Не удалось выбрать динамики: ${error.message}`); } };
if (!('setSinkId' in $('audio'))) { $('speaker-select').disabled = true; $('speaker-help').textContent = 'Этот браузер использует системный выбор динамиков.'; }
$('export').onclick = () => {
  const trace = traces.get(detailId); if (!trace) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(trace,null,2)], {type:'application/json'}));
  const link = document.createElement('a'); link.href = url; link.download = `diana-${trace.id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
};
setInterval(() => { $('stage-time').textContent = ['queued','verify','whisper','codex','tts','speaking'].includes(stageName) ? time(performance.now()-stageStarted) : ''; },200);
drawMetrics({});
