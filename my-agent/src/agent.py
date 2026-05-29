import asyncio
import json
import logging
import os
from datetime import date, datetime, timezone

import httpx
from dotenv import load_dotenv
from livekit import rtc
from livekit import api as lkapi
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    InterruptionOptions,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import sarvam, deepgram, groq, silero, dtln
from livekit.agents.voice import room_io

# ── Sarvam TTS bug fix ────────────────────────────────────────────────────────
# bulbul:v2 declares mime_type="audio/wav" but streams MP3 bytes → patch to mpeg.
# bulbul:v3/v3-beta send real WAV — skip patch or WAV gets decoded as MP3 (stuttering).
def _patch_sarvam_mime() -> None:
    try:
        import livekit.plugins.sarvam.tts as _st
        _orig = _st.SynthesizeStream._run
        async def _fixed_run(self, output_emitter):  # type: ignore[override]
            # Detect model from stream object to decide whether patch is needed
            _raw_model = ""
            for _attr in ("_model", "_opts"):
                _obj = getattr(self, _attr, None)
                if _obj is None:
                    continue
                _raw_model = str(
                    getattr(_obj, "model", "") or getattr(_obj, "_model", "") or ""
                )
                if _raw_model:
                    break
            # Patch only for v2 (known MP3-as-WAV bug). v3+ sends real WAV.
            # Unknown model → default to patching (safe backward-compat).
            _needs_patch = ("v3" not in _raw_model)
            _orig_init = output_emitter.initialize
            def _fixed_init(*a, **kw):
                if _needs_patch and kw.get("mime_type") == "audio/wav" and kw.get("stream"):
                    kw["mime_type"] = "audio/mpeg"
                return _orig_init(*a, **kw)
            output_emitter.initialize = _fixed_init
            return await _orig(self, output_emitter)
        _st.SynthesizeStream._run = _fixed_run
    except Exception:
        pass  # non-fatal

_patch_sarvam_mime()
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.local"))

from instructions import SYSTEM_PROMPT, SURVEY_QUESTIONS, QUESTION_KEYWORDS  # noqa: E402
from tools import CallState, build_tools, _add_event  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("agent")

SURVEY_NAME = os.getenv("SURVEY_NAME", "ग्राम विकास सर्वे")
GCS_BUCKET  = os.getenv("GCS_BUCKET", "anthrovoice-recordings")
GREETING    = os.getenv(
    "GREETING",
    f"नमस्ते! मैं {SURVEY_NAME} की तरफ से बात कर रहा हूँ। क्या आप कुछ मिनट दे सकते हैं?",
)

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.45,
        activation_threshold=0.45,
        min_speech_duration=0.15,
        max_buffered_speech=45.0,
    )


server.setup_fnc = prewarm


def _build_sarvam_tts(model: str, speaker: str) -> sarvam.TTS:
    """Build a Sarvam TTS instance with model-appropriate settings.

    v2  – streaming=True (WebSocket), 8 kHz.  Unchanged from original.
    v3/v3-beta – streaming=False (HTTP synthesize path).
                 The v3 WebSocket silently returns no audio for short Hindi
                 sentences regardless of min_buffer_size.  The HTTP path is
                 reliable: it sends speech_sample_rate to the server and gets
                 back proper WAV with embedded sample-rate headers.
                 22050 Hz is v3's native rate; livekit resamples to 8 kHz for SIP.
    """
    is_v2 = model == "bulbul:v2"
    tts_instance = sarvam.TTS(
        model=model,
        speaker=speaker,
        target_language_code="hi-IN",
        pace=1.0,
        speech_sample_rate=8000 if is_v2 else 22050,
        min_buffer_size=50 if is_v2 else 30,  # v3: 30 = plugin-enforced minimum
    )
    if not is_v2:
        # Force v3 onto the HTTP synthesize() path instead of WebSocket stream().
        # Disable the streaming capability so livekit calls tts.synthesize(text)
        # (ChunkedStream / HTTP REST) rather than tts.stream() (WebSocket).
        from livekit.agents import tts as _lk_tts
        tts_instance._capabilities = _lk_tts.TTSCapabilities(streaming=False)
        # Safety: also patch min_buffer_size in case stream() is called anyway.
        tts_instance._opts.min_buffer_size = 1
    return tts_instance


@server.rtc_session(agent_name="my-agent")
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    is_sip = participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
    phone = (
        participant.attributes.get("sip.phoneNumber", "unknown")
        if is_sip
        else participant.identity
    )

    meta_raw = getattr(ctx.job, "metadata", "") or ""
    call_meta: dict[str, str] = {}
    if meta_raw:
        try:
            call_meta = json.loads(meta_raw)
        except Exception:
            logger.warning(f"[call] failed to parse metadata: {meta_raw!r}")

    from_number      = call_meta.get("from_number") or phone
    to_number        = call_meta.get("to_number") or phone
    agent_id         = call_meta.get("agent_id") or "my-agent"
    custom_prompt    = (call_meta.get("custom_prompt") or "").strip()
    webhook_url      = (call_meta.get("webhook_url") or "").strip()
    if webhook_url and not (
        webhook_url.startswith("http://") or webhook_url.startswith("https://")
    ):
        webhook_url = "https://" + webhook_url.lstrip("/")
    external_call_id = call_meta.get("external_call_id") or ctx.room.name

    # Sarvam TTS model + speaker from trigger metadata
    # sarvam_model: "bulbul:v2" | "bulbul:v3" | "bulbul:v3-beta"  (default v2)
    sarvam_model = (call_meta.get("sarvam_model") or "bulbul:v2").strip()
    if sarvam_model not in ("bulbul:v2", "bulbul:v3", "bulbul:v3-beta"):
        sarvam_model = "bulbul:v2"  # safe fallback

    # Speaker: use caller-supplied value, else pick a sensible default per model
    _caller_speaker = (call_meta.get("tts_speaker") or "").strip().lower() or None
    if _caller_speaker:
        sarvam_speaker: str | None = _caller_speaker
    elif sarvam_model == "bulbul:v2":
        sarvam_speaker = "anushka"  
    else:
        sarvam_speaker = "kavya" 

    logger.info(
        f"[call] {'SIP' if is_sip else 'web'} from:{from_number} to:{to_number} "
        f"room:{ctx.room.name} agent_id:{agent_id} call_id:{external_call_id} "
        f"tts:{sarvam_model} speaker:{sarvam_speaker}"
    )
    if webhook_url:
        logger.info(f"[call] webhook_url:{webhook_url}")

    state = CallState()
    today = date.today().strftime("%A, %B %d, %Y")
    instructions = (
        SYSTEM_PROMPT
        + f"\n\nआज की तारीख: {today}। आप {SURVEY_NAME} का सर्वे कर रहे हैं।"
    )
    if custom_prompt:
        instructions += f"\n\nCall-specific instructions (follow exactly):\n{custom_prompt}"

    transcript_lines: list[str] = []
    call_started_at = datetime.now(timezone.utc)
    state.events.append(
        {
            "type": "call_started",
            "ts": call_started_at.isoformat(),
            "room_name": ctx.room.name,
            "from_number": from_number,
            "to_number": to_number,
            "agent_id": agent_id,
            "external_call_id": external_call_id,
        }
    )

    tools = build_tools(
        ctx=ctx,
        session=None,
        participant=participant,
        state=state,
    )

    session = AgentSession(
        stt=deepgram.STT(
            model="nova-2",
            language="hi",
            punctuate=True,
            smart_format=True,
            filler_words=False,
        ),
        llm=groq.LLM(
            model="llama-3.3-70b-versatile",  # 8B made wrong tool calls; 70B proven correct
            temperature=0.3,
        ),
        tts=_build_sarvam_tts(sarvam_model, sarvam_speaker),
        vad=ctx.proc.userdata["vad"],
        tools=tools,
        preemptive_generation=True,
        turn_handling=TurnHandlingOptions(
            min_endpointing_delay=0.2,   # detect end-of-speech faster
            max_endpointing_delay=1.5,   # don't wait long — survey answers are short
            allow_interruptions=True,
            interruption=InterruptionOptions(mode="vad"),
        ),
    )

    webhook_task: asyncio.Task[None] | None = None

    async def _flush_webhook_task() -> None:
        if webhook_task is None:
            return
        try:
            await asyncio.wait_for(webhook_task, timeout=6)
        except Exception:
            return

    ctx.add_shutdown_callback(_flush_webhook_task)

    # ── Recording ─────────────────────────────────────────────────────────
    async def _start_recording() -> None:
        try:
            lk = lkapi.LiveKitAPI()
            safe = to_number.replace("+", "").replace(" ", "").replace("-", "")
            filename = f"{safe}-{int(call_started_at.timestamp())}.ogg"
            await lk.egress.start_room_composite_egress(
                lkapi.RoomCompositeEgressRequest(
                    room_name=ctx.room.name,
                    audio_only=True,
                    file=lkapi.EncodedFileOutput(
                        file_type=lkapi.EncodedFileType.OGG,
                        filepath=f"recordings/{filename}",
                        gcp=lkapi.GCPUpload(
                            bucket=GCS_BUCKET,
                        ),
                    ),
                )
            )
            await lk.aclose()
            logger.info(f"[recording] started: {filename}")
            state.events.append({
                "type": "recording_started",
                "ts": datetime.now(timezone.utc).isoformat(),
                "filename": filename,
                "bucket": GCS_BUCKET,
            })
        except Exception as e:
            logger.warning(f"[recording] failed to start: {e}")

    # Non-answers: caller confusion sounds that shouldn't be saved as survey responses
    _NON_ANSWERS = {"hello", "हेलो", "हैलो", "हाय", "hi"}

    @session.on("conversation_item_added")
    def on_item_added(ev):
        try:
            item = ev.item
            text = (item.text_content or "").strip()
            if not text:
                return

            if item.role == "assistant":
                # Detect if agent just asked the current survey question
                if state.q_idx < len(SURVEY_QUESTIONS):
                    kws = QUESTION_KEYWORDS[state.q_idx]
                    if any(kw in text for kw in kws):
                        state.pending_question_key = SURVEY_QUESTIONS[state.q_idx][0]
                label = "Agent"

            else:  # caller
                label = "Caller"
                if state.pending_question_key:
                    _clean = text.lower().strip("।?!, .")
                    if _clean not in _NON_ANSWERS and _clean:
                        # Valid answer — save it
                        state.survey_responses[state.pending_question_key] = text
                        _add_event(state, "survey_response_saved",
                                   question_key=state.pending_question_key, answer=text)
                        logger.info(f"[survey] '{state.pending_question_key}': {text[:60]}")
                        state.q_idx += 1
                        state.pending_question_key = None
                    # If "Hello?" → don't save, keep pending so agent re-asks

            line = f"{label}: {text}"
            transcript_lines.append(line)
            logger.info(f"[turn] {line[:100]}")
        except Exception:
            pass

    @session.on("close")
    def on_close():
        ended_at = datetime.now(timezone.utc)
        duration_s = (ended_at - call_started_at).total_seconds()
        questions_answered = len(state.survey_responses)
        logger.info(
            f"[call] ended room:{ctx.room.name} turns:{len(transcript_lines)} "
            f"duration_s:{duration_s:.1f} survey_answers:{questions_answered}"
        )
        if transcript_lines:
            logger.info("[transcript]\n" + "\n".join(transcript_lines))
        if state.survey_responses:
            logger.info(f"[survey] responses: {json.dumps(state.survey_responses, ensure_ascii=False)}")

        if webhook_url:
            payload = {
                # ── Call metadata ──────────────────────────────────────────
                "call_id":         external_call_id,
                "agent_id":        agent_id,
                "survey_name":     SURVEY_NAME,
                "room_name":       ctx.room.name,
                "from":            from_number,
                "to":              to_number,
                "start_time":      call_started_at.isoformat(),
                "end_time":        ended_at.isoformat(),
                "duration_seconds": round(duration_s, 1),
                # ── Survey results ─────────────────────────────────────────
                "survey_responses": state.survey_responses,
                "questions_answered": questions_answered,
                # ── Full transcript ────────────────────────────────────────
                "transcript": transcript_lines,
                # ── Event log ─────────────────────────────────────────────
                "events": state.events,
            }

            async def _send_webhook() -> None:
                async with httpx.AsyncClient(timeout=8) as client:
                    for attempt in range(1, 4):
                        try:
                            res = await client.post(webhook_url, json=payload)
                            if 200 <= res.status_code < 300:
                                logger.info(
                                    f"[webhook] sent to {webhook_url} status={res.status_code}"
                                )
                                return
                            logger.warning(
                                f"[webhook] non-2xx status={res.status_code} attempt={attempt} "
                                f"url={webhook_url}"
                            )
                        except Exception as exc:
                            logger.warning(
                                f"[webhook] failed attempt={attempt} url={webhook_url}: {exc}"
                            )
                        await asyncio.sleep(0.5 * attempt)

            nonlocal webhook_task
            webhook_task = asyncio.create_task(_send_webhook())
            webhook_task.add_done_callback(lambda _: None)

    _t_stt: list[float] = []  # mutable container for closure

    @session.on("agent_state_changed")
    def on_state_changed(ev):
        now = datetime.now()
        state_name = str(ev.new_state).split(".")[-1]
        if state_name == "THINKING":   # LLM just started (STT done)
            gap = f"{(now.timestamp() - _t_stt[0]):.2f}s after STT" if _t_stt else ""
            logger.info(f"[timing] → THINKING (LLM started) {gap}")
        elif state_name == "SPEAKING":  # TTS first chunk ready (LLM done)
            gap = f"{(now.timestamp() - _t_stt[0]):.2f}s after STT" if _t_stt else ""
            logger.info(f"[timing] → SPEAKING (TTS playing) {gap}")
        elif state_name == "LISTENING":
            logger.info(f"[timing] → LISTENING")
            _t_stt.clear()

    @session.on("user_input_transcribed")
    def on_transcribed(ev):
        if ev.is_final:
            t = datetime.now()
            _t_stt.clear()
            _t_stt.append(t.timestamp())
            logger.info(f"[timing] STT final: '{ev.transcript[:30]}' at {t.strftime('%H:%M:%S.%f')[:-3]}")

    await session.start(
        agent=Agent(
            instructions=instructions,
            use_tts_aligned_transcript=False,
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(strength=0.35),
            ),
        ),
    )

    logger.info(f"[agent] ready room:{ctx.room.name}")

    asyncio.ensure_future(_start_recording())

    # Greeting: no LLM, pure TTS. allow_interruptions=False prevents line noise from cutting it.
    await session.say(GREETING, allow_interruptions=False)


if __name__ == "__main__":
    cli.run_app(server)