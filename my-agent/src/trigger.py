import os
import time
import json
import logging
import asyncio
import pathlib
from collections import deque
from aiohttp import web
from livekit import api as lkapi
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env.local")

# In-memory log buffer — last 500 lines
log_buffer = deque(maxlen=500)

class BufferHandler(logging.Handler):
    def emit(self, record):
        log_buffer.append(self.format(record))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        BufferHandler(),
    ],
)
logger = logging.getLogger("trigger")

TRIGGER_SECRET         = os.getenv("TRIGGER_SECRET", "change-me")
LIVEKIT_SIP_TRUNK_ID   = os.getenv("LIVEKIT_SIP_TRUNK_ID", "")
AGENT_NAME             = "my-agent"
DEFAULT_WEBHOOK_URL    = os.getenv("DEFAULT_WEBHOOK_URL", "")

routes = web.RouteTableDef()


@routes.get("/health")
async def health(_: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "agent": AGENT_NAME})


@routes.get("/logs")
async def get_logs(request: web.Request) -> web.Response:
    if request.headers.get("x-api-key") != TRIGGER_SECRET:
        return web.json_response({"error": "Unauthorized"}, status=401)
    n = int(request.query.get("n", 100))
    lines = list(log_buffer)[-n:]
    return web.Response(
        text="\n".join(lines),
        content_type="text/plain",
    )


@routes.post("/trigger")
async def trigger_call(request: web.Request) -> web.Response:
    if request.headers.get("x-api-key") != TRIGGER_SECRET:
        logger.warning("[trigger] unauthorized")
        return web.json_response({"error": "Unauthorized"}, status=401)

    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Invalid JSON"}, status=400)

    # Backwards compatible: support legacy {"phone_number": "..."}
    legacy_phone = (body.get("phone_number") or "").strip()

    from_number = (body.get("from") or body.get("from_number") or "").strip()
    to_number = (body.get("to") or body.get("to_number") or legacy_phone).strip()

    if not to_number:
        return web.json_response(
            {"error": "to/to_number or phone_number required"}, status=400
        )

    agent_id = (body.get("agent_id") or "").strip()
    custom_prompt = (body.get("prompt") or body.get("custom_prompt") or "").strip()
    webhook_url = (body.get("webhook") or body.get("webhook_url") or DEFAULT_WEBHOOK_URL).strip()
    external_call_id = (body.get("call_id") or body.get("request_id") or "").strip()

    # Sarvam TTS version / model
    # Accept shorthand "v2", "v3", "v3-beta" OR full model string "bulbul:v2" etc.
    raw_tts = (body.get("tts_version") or body.get("sarvam_version") or "v2").strip().lower()
    _model_map = {
        "v2": "bulbul:v2",
        "v3": "bulbul:v3",
        "v3-beta": "bulbul:v3-beta",
        "bulbul:v2": "bulbul:v2",
        "bulbul:v3": "bulbul:v3",
        "bulbul:v3-beta": "bulbul:v3-beta",
    }
    sarvam_model = _model_map.get(raw_tts, "bulbul:v2")
    tts_version = sarvam_model  # keep for response/logging

    # Optional speaker override — must be compatible with the chosen model
    tts_speaker = (body.get("speaker") or "").strip().lower() or None

    safe_to = to_number.replace("+", "").replace(" ", "").replace("-", "")
    room_name = f"call-{safe_to}-{int(time.time() * 1000)}"

    metadata = {
        "agent_id": agent_id or AGENT_NAME,
        "from_number": from_number,
        "to_number": to_number,
        "custom_prompt": custom_prompt,
        "webhook_url": webhook_url,
        "external_call_id": external_call_id or room_name,
        "sarvam_model": sarvam_model,
        "tts_speaker": tts_speaker,  # None = let plugin pick the default for the model
    }

    try:
        lk = lkapi.LiveKitAPI()
        await lk.room.create_room(lkapi.CreateRoomRequest(name=room_name))
        await lk.agent_dispatch.create_dispatch(
            lkapi.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )
        await lk.sip.create_sip_participant(
            lkapi.CreateSIPParticipantRequest(
                sip_trunk_id=LIVEKIT_SIP_TRUNK_ID,
                sip_call_to=to_number,
                room_name=room_name,
                participant_identity=f"sip-{safe_to}",
                participant_name="Caller",
            )
        )
        await lk.aclose()
        logger.info(
            f"[trigger] call started room:{room_name} from:{from_number or 'n/a'} to:{to_number}"
        )
        return web.json_response(
            {
                "room_name": room_name,
                "from": from_number,
                "to": to_number,
                "agent": AGENT_NAME,
                "agent_id": metadata["agent_id"],
                "call_id": metadata["external_call_id"],
                "sarvam_model": sarvam_model,
                "tts_speaker": tts_speaker,
            }
        )

    except Exception as e:
        logger.error(f"[trigger] failed: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def main():
    app = web.Application()
    app.add_routes(routes)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.info(f"[trigger] HTTP on :8080 agent:{AGENT_NAME} trunk:{LIVEKIT_SIP_TRUNK_ID}")
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())